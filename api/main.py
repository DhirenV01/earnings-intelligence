"""
api/main.py

FastAPI application for the earnings-intelligence RAG pipeline.

Endpoints:
    GET  /            Service info and available endpoints
    GET  /health      Liveness check (is the process running?)
    GET  /ready       Readiness check (can I serve traffic?)
    GET  /demo        Interactive demo UI
    GET  /metrics     Aggregate query statistics
    POST /ingest      Multipart upload: .txt file + ticker -> ingest_transcript()
    POST /query       JSON body: question + optional filters -> query_transcripts()

Local dev:
    uvicorn api.main:app --reload --port 8000

Lambda deployment:
    Mangum handler at the bottom wraps the ASGI app for API Gateway.
"""

from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import json
import logging
import os
import sys
import time
import uuid

# ---------------------------------------------------------------------------
# Repo-root sys.path injection (same pattern as ingest_upload.py / query.py)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env for local dev only. override=False ensures Railway's injected env
# vars are never clobbered -- os.environ values always win. No-op if no .env.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
_log = logging.getLogger("earnings_intelligence")

# ---------------------------------------------------------------------------
# DynamoDB setup -- optional; falls back to JSONL if boto3 missing or no creds
# ---------------------------------------------------------------------------

_DYNAMO_TABLE_NAME = "earnings-query-logs"
_DYNAMO_TABLE      = None  # set below if boto3 is available

try:
    import boto3
    _AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")
    _dynamo_res  = boto3.resource("dynamodb", region_name=_AWS_REGION)
    _DYNAMO_TABLE = _dynamo_res.Table(_DYNAMO_TABLE_NAME)
    _log.info("DynamoDB query logging enabled (table=%s, region=%s)", _DYNAMO_TABLE_NAME, _AWS_REGION)
except ImportError:
    _log.info("boto3 not installed -- query logging falls back to JSONL")


def _to_dynamo(obj):
    """Recursively convert Python floats to Decimal (required by boto3 resource)."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dynamo(i) for i in obj]
    return obj


def _from_dynamo(obj):
    """Recursively convert DynamoDB Decimal back to int or float."""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: _from_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_dynamo(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# FastAPI imports
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Run: pip install fastapi python-multipart")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENV_VARS = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"]

_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
]
# In production, append your real frontend domain:
# _ALLOWED_ORIGINS.append("https://earnings-agent.yourdomain.com")

# For portfolio/demo mode, allow all origins. Swap to _ALLOWED_ORIGINS for prod.
_CORS_ORIGINS = ["*"]

_DEMO_HTML = Path(__file__).with_name("demo.html")
_LOG_FILE  = _REPO_ROOT / "logs" / "query_log.jsonl"

_QUERY_TIMEOUT_SECONDS = 30.0

# Rate limiting: requests per user per window
_RATE_LIMIT_WINDOW  = 60   # seconds
_RATE_LIMIT_MAX     = 30   # max requests per window
_rate_limit_tracker: dict[str, list[float]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic. Resources loaded here are available
    on app.state throughout the application lifetime."""

    # -- Startup --
    present = [v for v in _ENV_VARS if os.getenv(v)]
    missing = [v for v in _ENV_VARS if not os.getenv(v)]
    _log.info("ENV CHECK: present=%s missing=%s", present, missing)
    if missing:
        _log.error(
            "STARTUP WARNING: required env vars not found: %s "
            "-- check Railway variable definitions and redeploy.",
            missing,
        )
    app.state.ready = not missing  # only ready if all env vars are present

    yield

    # -- Shutdown --
    _log.info("Shutting down gracefully.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="earnings-intelligence",
    description="RAG pipeline for querying earnings call transcripts.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request latency tracking
# ---------------------------------------------------------------------------

@app.middleware("http")
async def track_latency(request: Request, call_next):
    """Attach response time header to every response for observability."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = str(round(duration_ms, 1))
    _log.info(
        "%s %s - %.1fms - %s",
        request.method,
        request.url.path,
        duration_ms,
        response.status_code,
    )
    return response


# ---------------------------------------------------------------------------
# Middleware: rate limiting (in-memory, per-process)
# ---------------------------------------------------------------------------

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Simple sliding-window rate limiter. Skips health/readiness checks.
    In production with multiple replicas, replace with Redis-backed limiter."""
    skip_paths = {"/health", "/ready", "/docs", "/openapi.json"}
    if request.url.path in skip_paths:
        return await call_next(request)

    client_id = request.headers.get("X-User-Id", request.client.host if request.client else "unknown")
    now = time.time()

    # Prune expired entries, check limit
    _rate_limit_tracker[client_id] = [
        t for t in _rate_limit_tracker[client_id] if now - t < _RATE_LIMIT_WINDOW
    ]
    if len(_rate_limit_tracker[client_id]) >= _RATE_LIMIT_MAX:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again shortly."},
            headers={"Retry-After": str(_RATE_LIMIT_WINDOW)},
        )

    _rate_limit_tracker[client_id].append(now)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    ticker:   Optional[str] = None
    quarter:  Optional[str] = None
    role:     Optional[str] = None
    section:  Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers: query logging
# ---------------------------------------------------------------------------

def _log_query(entry: dict) -> None:
    """Write a query log entry to DynamoDB, falling back to JSONL. Never raises."""
    entry = {"query_id": str(uuid.uuid4()), **entry}

    if _DYNAMO_TABLE is not None:
        try:
            _DYNAMO_TABLE.put_item(Item=_to_dynamo(entry))
            return
        except Exception as exc:
            _log.warning("DynamoDB write failed, falling back to JSONL: %s", exc)

    # JSONL fallback
    try:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        _log.warning("Query logging failed (non-fatal): %s", exc)


def _get_log_entries() -> list:
    """Return all log entries from DynamoDB (preferred) or JSONL fallback."""
    if _DYNAMO_TABLE is not None:
        try:
            items = []
            resp = _DYNAMO_TABLE.scan()
            items.extend(resp.get("Items", []))
            # DynamoDB paginates at 1 MB; follow LastEvaluatedKey until exhausted
            while "LastEvaluatedKey" in resp:
                resp = _DYNAMO_TABLE.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
                items.extend(resp.get("Items", []))
            return [_from_dynamo(item) for item in items]
        except Exception as exc:
            _log.warning("DynamoDB scan failed, falling back to JSONL: %s", exc)

    # JSONL fallback
    if not _LOG_FILE.exists():
        return []
    entries = []
    try:
        with _LOG_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as exc:
        _log.warning("Failed to read query log: %s", exc)
    return entries


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict:
    """Service info and available endpoints."""
    return {
        "name":      "Earnings Call Intelligence Agent",
        "version":   "0.2.0",
        "endpoints": ["/health", "/ready", "/ingest", "/query", "/metrics"],
        "docs":      "/docs",
    }


# ---------------------------------------------------------------------------
# GET /health  (liveness)
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Liveness check: is the process alive and responding?
    Load balancers and container orchestrators use this to detect crashed processes."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /ready  (readiness)
# ---------------------------------------------------------------------------

@app.get("/ready")
def ready() -> dict:
    """Readiness check: is the service ready to handle traffic?
    Returns 503 if required env vars are missing. Load balancers use this
    to decide whether to route requests to this instance."""
    missing = [v for v in _ENV_VARS if not os.getenv(v)]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Not ready: missing env vars {missing}",
        )
    return {
        "status":       "ready",
        "env_vars_set": [v for v in _ENV_VARS if os.getenv(v)],
    }


# ---------------------------------------------------------------------------
# GET /demo
# ---------------------------------------------------------------------------

@app.get("/demo", response_class=FileResponse)
def demo():
    """Serve the interactive demo UI."""
    if not _DEMO_HTML.exists():
        raise HTTPException(status_code=404, detail="demo.html not found")
    return FileResponse(_DEMO_HTML, media_type="text/html")


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@app.post("/ingest", status_code=200)
async def ingest(
    file:   UploadFile = File(..., description="Raw .txt earnings call transcript"),
    ticker: str        = Form(..., description="Equity ticker, e.g. AAPL"),
    force:  bool       = Form(False, description="Overwrite if transcript already exists"),
) -> dict:
    """
    Accept a .txt transcript file and a ticker symbol, run the full
    normalize -> embed -> upsert pipeline, and return the ingest summary.

    Returns the dict from ingest_transcript():
        transcript_id, chunks_upserted, tokens_estimated, elapsed_seconds, skipped
    """
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(status_code=422, detail="File must be a .txt transcript")

    raw_bytes = await file.read()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"File is not valid UTF-8: {exc}")

    try:
        from ingestion.ingest_upload import ingest_transcript
        result = ingest_transcript(ticker=ticker, raw_text=raw_text, force=force)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        _log.exception("Ingest pipeline error")
        raise HTTPException(status_code=500, detail=f"Ingest pipeline error: {exc}")

    return result


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@app.post("/query", status_code=200)
async def query(request: QueryRequest) -> dict:
    """
    Accept a natural language question and optional metadata filters,
    run the RAG query pipeline, and return the grounded answer with sources.

    Includes a timeout to prevent long-running queries from blocking the server.
    """
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    import asyncio

    try:
        from query.query import QueryFilter, query_transcripts
        filters = QueryFilter(
            ticker  = request.ticker,
            quarter = request.quarter,
            role    = request.role,
            section = request.section,
        )

        _t0 = time.monotonic()

        # Run the sync pipeline in a thread with a timeout so one slow query
        # can't block the event loop or hang indefinitely.
        result = await asyncio.wait_for(
            asyncio.to_thread(query_transcripts, request.question, filters),
            timeout=_QUERY_TIMEOUT_SECONDS,
        )

        _elapsed = round(time.monotonic() - _t0, 3)

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Query timed out after {_QUERY_TIMEOUT_SECONDS}s",
        )
    except Exception as exc:
        _log.exception("Query pipeline error")
        raise HTTPException(status_code=500, detail=f"Query pipeline error: {exc}")

    _log_query({
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "question":       result.question,
        "ticker_filter":  request.ticker  or "none",
        "quarter_filter": request.quarter or "none",
        "role_filter":    request.role    or "none",
        "section_filter": request.section or "none",
        "found":          result.found,
        "confidence":     result.confidence,
        "chunks_used":    result.chunks_used,
        "model":          result.model,
        "elapsed_seconds": _elapsed,
    })

    return {
        "question":    result.question,
        "answer":      result.answer,
        "found":       result.found,
        "confidence":  result.confidence,
        "model":       result.model,
        "chunks_used": result.chunks_used,
        "sources": [
            {
                "chunk_id": s.chunk_id,
                "speaker":  s.speaker,
                "role":     s.role,
                "firm":     s.firm,
                "ticker":   s.ticker,
                "quarter":  s.quarter,
                "section":  s.section,
                "score":    s.score,
                "text":     s.text,
            }
            for s in result.sources
        ],
    }


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

@app.get("/metrics")
def metrics() -> dict:
    """
    Aggregate statistics derived from DynamoDB (or logs/query_log.jsonl fallback).
    Returns zeroed-out metrics if no log entries exist yet.
    """
    _ZERO = {
        "total_queries": 0,
        "queries_by_confidence": {"high": 0, "medium": 0, "low": 0},
        "queries_by_ticker": {},
        "average_chunks_used": 0.0,
        "found_rate": 0.0,
        "average_latency_seconds": 0.0,
        "last_10_queries": [],
    }

    entries = _get_log_entries()
    if not entries:
        return _ZERO

    confidence_counts: dict = {"high": 0, "medium": 0, "low": 0}
    ticker_counts: dict     = {}
    total_chunks             = 0
    total_latency            = 0.0
    found_count              = 0

    for e in entries:
        conf = e.get("confidence", "medium")
        if conf in confidence_counts:
            confidence_counts[conf] += 1

        ticker = e.get("ticker_filter")
        if ticker:
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

        total_chunks  += e.get("chunks_used", 0)
        total_latency += e.get("elapsed_seconds", 0.0)

        if e.get("found", False):
            found_count += 1

    n = len(entries)
    last_10 = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)[:10]
    return {
        "total_queries":           n,
        "queries_by_confidence":   confidence_counts,
        "queries_by_ticker":       ticker_counts,
        "average_chunks_used":     round(total_chunks / n, 2),
        "found_rate":              round(found_count / n * 100, 1),
        "average_latency_seconds": round(total_latency / n, 3),
        "last_10_queries":         last_10,
    }


# ---------------------------------------------------------------------------
# Lambda handler (Mangum wraps the ASGI app for API Gateway / Function URLs)
# ---------------------------------------------------------------------------

try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    handler = None  # Mangum is optional for local dev; required for Lambda