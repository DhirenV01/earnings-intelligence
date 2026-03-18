"""
ingest_upload.py

Manual ingest path: .txt transcript → normalize → embed → Pinecone upsert.

CLI usage:
    python ingest_upload.py data/samples/aapl_q1_2026.txt AAPL
    python ingest_upload.py data/samples/msft_q2_2026.txt MSFT --force

Callable as a function (for Lambda, FastAPI, ingest_scheduled.py):
    from ingestion.ingest_upload import ingest_transcript
    result = ingest_transcript(ticker="AAPL", file_path="path/to/file.txt")
    result = ingest_transcript(ticker="AAPL", raw_text="...")  # Lambda variant

Environment variables (loaded from .env automatically):
    OPENAI_API_KEY
    PINECONE_API_KEY
    PINECONE_INDEX  (default: earnings-intelligence)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure repo root (parent of ingestion/) is on sys.path so that
# `pipeline` and `query` packages are importable when this script is
# run directly (python ingestion/ingest_upload.py ...) or imported
# from a different working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Logging — structured format so CloudWatch can parse fields
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ingest_upload")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_transcript(
    ticker: str,
    file_path: Optional[str] = None,
    raw_text: Optional[str] = None,
    force: bool = False,
) -> dict:
    """
    Normalize → embed → upsert a single earnings call transcript.

    Parameters
    ----------
    ticker    : str   Equity ticker, e.g. "AAPL".
    file_path : str   Path to raw .txt transcript file (mutually exclusive with raw_text).
    raw_text  : str   Raw transcript text (for Lambda / HTTP upload paths).
    force     : bool  Overwrite if transcript_id already exists in Pinecone.

    Returns
    -------
    dict  Summary with keys: transcript_id, chunks_upserted, tokens_estimated,
          elapsed_seconds, skipped.
    """
    t_start = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    _validate_inputs(ticker, file_path, raw_text)

    # ------------------------------------------------------------------
    # 2. Load raw text
    # ------------------------------------------------------------------
    if raw_text is None:
        log.info("Reading transcript from %s", file_path)
        raw_text = Path(file_path).read_text(encoding="utf-8")
        log.info("Loaded %d characters", len(raw_text))

    # ------------------------------------------------------------------
    # 3. Normalize → Silver JSON
    # ------------------------------------------------------------------
    log.info("Normalizing transcript for ticker=%s", ticker.upper())
    try:
        from pipeline.normalize import normalize_transcript
        silver = normalize_transcript(raw_text, ticker=ticker)
    except Exception as exc:
        log.error("Normalization failed: %s", exc)
        raise RuntimeError(f"Normalization failed: {exc}") from exc

    transcript_id = silver["transcript_id"]
    log.info(
        "Normalized: transcript_id=%s  segments=%d  quarter=%s",
        transcript_id, len(silver["segments"]), silver["quarter"],
    )

    # ------------------------------------------------------------------
    # 4. Duplicate check
    # ------------------------------------------------------------------
    log.info("Checking Pinecone for existing transcript_id=%s", transcript_id)
    try:
        already_exists = _transcript_exists_in_pinecone(transcript_id)
    except Exception as exc:
        log.error("Pinecone duplicate check failed: %s", exc)
        raise RuntimeError(f"Pinecone duplicate check failed: {exc}") from exc

    if already_exists:
        if not force:
            log.warning(
                "transcript_id=%s already exists in Pinecone. "
                "Use --force to overwrite. Skipping.",
                transcript_id,
            )
            return {
                "transcript_id":    transcript_id,
                "chunks_upserted":  0,
                "tokens_estimated": 0,
                "elapsed_seconds":  round(time.monotonic() - t_start, 2),
                "skipped":          True,
            }
        log.info("--force set: overwriting existing vectors for %s", transcript_id)

    # ------------------------------------------------------------------
    # 5. Embed + upsert
    # ------------------------------------------------------------------
    log.info("Embedding and upserting chunks for %s", transcript_id)
    try:
        from pipeline.embed import embed_transcript
        chunks = embed_transcript(silver, dry_run=False)
    except Exception as exc:
        log.error("Embed/upsert failed: %s", exc)
        raise RuntimeError(f"Embed/upsert failed: {exc}") from exc

    tokens_estimated = sum(c.token_count for c in chunks)
    elapsed = round(time.monotonic() - t_start, 2)

    log.info(
        "Ingest complete: transcript_id=%s  chunks=%d  tokens~=%d  elapsed=%.2fs",
        transcript_id, len(chunks), tokens_estimated, elapsed,
    )

    return {
        "transcript_id":    transcript_id,
        "chunks_upserted":  len(chunks),
        "tokens_estimated": tokens_estimated,
        "elapsed_seconds":  elapsed,
        "skipped":          False,
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_inputs(ticker: str, file_path: Optional[str], raw_text: Optional[str]) -> None:
    """Raise ValueError early so callers get clear messages before any API call."""
    if not ticker or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")

    if file_path is None and raw_text is None:
        raise ValueError("Provide either file_path or raw_text")

    if file_path is not None and raw_text is not None:
        raise ValueError("Provide file_path OR raw_text, not both")

    if file_path is not None:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Transcript file not found: {file_path}")
        if p.suffix.lower() != ".txt":
            raise ValueError(f"Expected a .txt file, got: {p.suffix}")

    _validate_env_keys()


def _validate_env_keys() -> None:
    """Load .env if present, then verify required keys exist."""
    _load_dotenv()

    missing = [k for k in ("OPENAI_API_KEY", "PINECONE_API_KEY") if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Set them in your .env file or shell environment."
        )


def _load_dotenv() -> None:
    """Load .env from repo root (or any parent) if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        # Walk up from CWD looking for a .env file
        search = Path.cwd()
        for _ in range(5):
            candidate = search / ".env"
            if candidate.exists():
                load_dotenv(candidate, override=False)
                log.debug("Loaded .env from %s", candidate)
                return
            search = search.parent
    except ImportError:
        pass  # dotenv not installed; env vars must be set externally


# ---------------------------------------------------------------------------
# Pinecone duplicate check
# ---------------------------------------------------------------------------

def _transcript_exists_in_pinecone(transcript_id: str) -> bool:
    """
    Query Pinecone for any vector whose metadata.transcript_id matches.
    Returns True if at least one chunk already exists for this transcript.
    """
    try:
        from pinecone import Pinecone
    except ImportError:
        raise ImportError("Run: pip install pinecone-client")

    index_name = os.getenv("PINECONE_INDEX", "earnings-intelligence")
    pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)

    # Use a zero vector — we only care about the metadata filter match, not ANN ranking
    from pipeline.embed import EMBEDDING_DIM
    dummy_vector = [0.0] * EMBEDDING_DIM

    result = index.query(
        vector=dummy_vector,
        filter={"transcript_id": {"$eq": transcript_id}},
        top_k=1,
        include_metadata=False,
    )
    return len(result.get("matches", [])) > 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest an earnings call transcript into the vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ingest_upload.py data/samples/aapl_q1_2026.txt AAPL\n"
            "  python ingest_upload.py data/samples/msft_q2_2026.txt MSFT --force\n"
        ),
    )
    parser.add_argument("file", help="Path to raw .txt transcript file")
    parser.add_argument("ticker", help="Equity ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite if transcript already exists in Pinecone",
    )
    return parser


def _print_summary(result: dict) -> None:
    print("\n" + "=" * 55)
    print("  INGEST SUMMARY")
    print("=" * 55)
    if result["skipped"]:
        print(f"  Status           : SKIPPED (already exists)")
        print(f"  transcript_id    : {result['transcript_id']}")
        print(f"  Tip              : re-run with --force to overwrite")
    else:
        print(f"  Status           : OK")
        print(f"  transcript_id    : {result['transcript_id']}")
        print(f"  Chunks upserted  : {result['chunks_upserted']}")
        print(f"  Tokens estimated : ~{result['tokens_estimated']:,}")
        print(f"  Elapsed          : {result['elapsed_seconds']:.2f}s")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    try:
        summary = ingest_transcript(
            ticker=args.ticker,
            file_path=args.file,
            force=args.force,
        )
    except (ValueError, FileNotFoundError, EnvironmentError) as exc:
        log.error("Input error: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        log.error("Pipeline error: %s", exc)
        sys.exit(2)

    _print_summary(summary)
    sys.exit(0)
