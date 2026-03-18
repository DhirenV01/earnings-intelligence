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

import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Repo-root sys.path injection (same pattern as ingest_upload.py / query.py)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# FastAPI imports — fail loudly so the dev knows what to install
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Run: pip install fastapi python-multipart")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="earnings-intelligence",
    description="RAG pipeline for querying earnings call transcripts.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

_ENV_VARS = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"]


class QueryRequest(BaseModel):
    question: str
    ticker:   Optional[str] = None
    quarter:  Optional[str] = None
    role:     Optional[str] = None
    section:  Optional[str] = None


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
        result = query_transcripts(request.question, filters)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query pipeline error: {exc}")

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
# Lambda handler (Mangum wraps the ASGI app for API Gateway / Function URLs)
# ---------------------------------------------------------------------------

try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    handler = None  # Mangum is optional for local dev; required for Lambda
