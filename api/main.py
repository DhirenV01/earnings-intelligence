"""
api/main.py

FastAPI application for the earnings-intelligence RAG pipeline.

Endpoints:
    POST /ingest   Multipart upload: .txt file + ticker → ingest_transcript()
    POST /query    JSON body: question + optional filters → query_transcripts()
    GET  /health   Env-var presence check

Local dev:
    uvicorn api.main:app --reload --port 8000

Lambda deployment:
    Mangum handler at the bottom wraps the ASGI app for API Gateway.
"""

from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Repo-root sys.path injection (same pattern as ingest_upload.py / query.py)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env for local dev only. override=False ensures Railway's injected env
# vars are never clobbered — os.environ values always win. No-op if no .env.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
_log = logging.getLogger("earnings_intelligence")

# ---------------------------------------------------------------------------
# DynamoDB setup — optional; falls back to JSONL if boto3 missing or no creds
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
    _log.info("boto3 not installed — query logging falls back to JSONL")


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
# FastAPI imports — fail loudly so the dev knows what to install
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Run: pip install fastapi python-multipart")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_ENV_VARS = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"]

app = FastAPI(
    title="earnings-intelligence",
    description="RAG pipeline for querying earnings call transcripts.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _log_startup_env() -> None:
    """Log env-var presence at boot so Railway / server logs make it obvious
    whether the required keys were injected into the process environment."""
    present = [v for v in _ENV_VARS if os.getenv(v)]
    missing = [v for v in _ENV_VARS if not os.getenv(v)]
    _log.info("ENV CHECK — present=%s missing=%s", present, missing)
    if missing:
        _log.error(
            "STARTUP WARNING: required env vars not found in os.environ: %s — "
            "check Railway variable definitions and redeploy.",
            missing,
        )


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
# GET /
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict:
    """Service info and available endpoints."""
    return {
        "name":      "Earnings Call Intelligence Agent",
        "version":   "1.0",
        "endpoints": ["/health", "/ingest", "/query"],
        "docs":      "/docs",
    }


# ---------------------------------------------------------------------------
# GET /demo
# ---------------------------------------------------------------------------

_DEMO_HTML = Path(__file__).with_name("demo.html")
_LOG_FILE  = _REPO_ROOT / "logs" / "query_log.jsonl"


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


@app.get("/demo", response_class=FileResponse)
def demo():
    """Serve the interactive demo UI."""
    if not _DEMO_HTML.exists():
        raise HTTPException(status_code=404, detail="demo.html not found")
    return FileResponse(_DEMO_HTML, media_type="text/html")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Returns service status and which required env vars are set."""
    return {
        "status":       "ok",
        "env_vars_set": [v for v in _ENV_VARS if os.getenv(v)],
    }


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
    normalize → embed → upsert pipeline, and return the ingest summary.

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
        # ingest_transcript raises ValueError for bad inputs (e.g. empty ticker)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest pipeline error: {exc}")

    return result


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@app.post("/query", status_code=200)
def query(request: QueryRequest) -> dict:
    """
    Accept a natural language question and optional metadata filters,
    run the RAG query pipeline, and return the grounded answer with sources.

    The response mirrors QueryResponse from query.py, serialised to a dict:
        question, answer, found, confidence, model, chunks_used, sources[]
    """
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    try:
        from query.query import QueryFilter, query_transcripts
        filters = QueryFilter(
            ticker  = request.ticker,
            quarter = request.quarter,
            role    = request.role,
            section = request.section,
        )
        _t0 = time.monotonic()
        result = query_transcripts(request.question, filters)
        _elapsed = round(time.monotonic() - _t0, 3)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query pipeline error: {exc}")

    _log_query({
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "question":       result.question,
        "ticker_filter":  request.ticker  or "",
        "quarter_filter": request.quarter or "",
        "role_filter":    request.role    or "",
        "section_filter": request.section or "",
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
        "last_10_queries": [],
    }

    entries = _get_log_entries()
    if not entries:
        return _ZERO

    confidence_counts: dict = {"high": 0, "medium": 0, "low": 0}
    ticker_counts: dict     = {}
    total_chunks             = 0
    found_count              = 0

    for e in entries:
        conf = e.get("confidence", "medium")
        if conf in confidence_counts:
            confidence_counts[conf] += 1

        ticker = e.get("ticker_filter")
        if ticker:
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

        total_chunks += e.get("chunks_used", 0)

        if e.get("found", False):
            found_count += 1

    n = len(entries)
    last_10 = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)[:10]
    return {
        "total_queries":          n,
        "queries_by_confidence":  confidence_counts,
        "queries_by_ticker":      ticker_counts,
        "average_chunks_used":    round(total_chunks / n, 2),
        "found_rate":             round(found_count / n * 100, 1),
        "last_10_queries":        last_10,
    }


# ---------------------------------------------------------------------------
# Lambda handler (Mangum wraps the ASGI app for API Gateway / Function URLs)
# ---------------------------------------------------------------------------

try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    handler = None  # Mangum is optional for local dev; required for Lambda
