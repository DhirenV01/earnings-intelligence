"""
ingest_scheduled.py

Scheduled ingest path: poll the Financial Modeling Prep (FMP) stable API for
earnings call transcripts for each ticker in watchlist.json, skip anything
already in Pinecone, and call ingest_transcript() for new ones.

Local usage:
    python ingestion/ingest_scheduled.py
    python ingestion/ingest_scheduled.py --tickers AAPL MSFT
    python ingestion/ingest_scheduled.py --tickers GOOGL --lookback 2 --dry-run

crontab (daily at 06:00):
    0 6 * * * cd /path/to/earnings-intelligence && python ingestion/ingest_scheduled.py >> logs/scheduled.log 2>&1

Lambda / EventBridge usage:
    Entry point: ingestion.ingest_scheduled.lambda_handler
    Optional event payload (all fields override watchlist.json):
        {
            "tickers":           ["AAPL", "MSFT"],
            "lookback_quarters": 2,
            "dry_run":           false
        }

Required environment variables:
    FMP_API_KEY      — Financial Modeling Prep API key (paid plan required for transcripts)
    OPENAI_API_KEY   — used by ingest_transcript()
    PINECONE_API_KEY — used by ingest_transcript()
    PINECONE_INDEX   — optional, default: earnings-intelligence

FMP endpoint used:
    GET https://financialmodelingprep.com/stable/earning-call-transcript
        ?symbol={TICKER}&limit={N}&apikey={KEY}

    Response array (newest first):
    [
      {
        "symbol":     "AAPL",
        "period":     "Q1",       # "Q1" | "Q2" | "Q3" | "Q4"
        "fiscalYear": "2026",     # string
        "date":       "2026-01-29",
        "content":    "Operator: Ladies and gentlemen..."
      },
      ...
    ]

Header synthesis:
    FMP transcripts contain only the raw call body — no Motley Fool preamble.
    _build_synthetic_text() prepends a minimal header so normalize.py can
    extract quarter, date, and company name using its existing regex patterns.
    Nothing is written to disk; the synthesized text lives only in memory.
"""

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Repo-root sys.path injection — same pattern as ingest_upload.py / query.py
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging — structured so CloudWatch can parse fields
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ingest_scheduled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FMP_TRANSCRIPT_URL = "https://financialmodelingprep.com/stable/earning-call-transcript"
WATCHLIST_PATH     = _REPO_ROOT / "watchlist.json"
DEFAULT_LOOKBACK   = 4   # most-recent quarters to consider per ticker


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

def _load_watchlist(
    override_tickers:  Optional[list] = None,
    override_lookback: Optional[int]  = None,
) -> tuple:
    """
    Load tickers and lookback_quarters from watchlist.json.
    CLI / Lambda overrides always win over the file.

    Returns (tickers: list[str], lookback_quarters: int).
    """
    tickers  = []
    lookback = DEFAULT_LOOKBACK

    if WATCHLIST_PATH.exists():
        try:
            data     = json.loads(WATCHLIST_PATH.read_text())
            tickers  = [t.upper() for t in data.get("tickers", [])]
            lookback = int(data.get("lookback_quarters", DEFAULT_LOOKBACK))
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("Could not parse %s: %s — using defaults", WATCHLIST_PATH, exc)
    else:
        log.warning(
            "%s not found. Pass --tickers or create the file.", WATCHLIST_PATH.name
        )

    if override_tickers:
        tickers = [t.upper() for t in override_tickers]
    if override_lookback is not None:
        lookback = override_lookback

    return tickers, lookback


# ---------------------------------------------------------------------------
# FMP API
# ---------------------------------------------------------------------------

def _fetch_fmp_transcripts(ticker: str, lookback: int) -> list:
    """
    Fetch the most recent `lookback` transcript records for `ticker` from FMP.

    Returns a list of raw dicts. Returns [] and logs on any error.

    Raises EnvironmentError for hard failures (missing key, plan too low)
    so the caller can abort the run rather than silently produce nothing.
    """
    api_key = os.environ.get("FMP_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "FMP_API_KEY is not set. Add it to your .env file or shell environment."
        )

    url = (
        f"{FMP_TRANSCRIPT_URL}"
        f"?symbol={ticker.upper()}&limit={lookback}&apikey={api_key}"
    )

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 402:
            raise EnvironmentError(
                f"FMP plan does not include transcript access (HTTP 402). "
                f"Upgrade at https://financialmodelingprep.com/developer/docs"
            ) from exc
        if exc.code == 401:
            raise EnvironmentError(
                f"FMP API key is invalid (HTTP 401). Check FMP_API_KEY in .env."
            ) from exc
        # 404 / 5xx — log and return empty so one bad ticker doesn't abort the run
        log.error("FMP HTTP %d for %s: %s", exc.code, ticker, body[:200])
        return []
    except urllib.error.URLError as exc:
        log.error("FMP network error for %s: %s", ticker, exc.reason)
        return []
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.error("FMP response parse error for %s: %s", ticker, exc)
        return []

    if not isinstance(data, list):
        log.error("Unexpected FMP response type for %s: %r", ticker, type(data))
        return []

    log.info("FMP: %d transcript(s) available for %s (limit=%d)", len(data), ticker, lookback)
    return data   # FMP returns newest-first; limit is applied server-side


# ---------------------------------------------------------------------------
# Field normalisation
# ---------------------------------------------------------------------------

def _parse_fmp_record(record: dict) -> Optional[dict]:
    """
    Normalise a raw FMP record into a canonical shape regardless of which
    API version returned it.

    Stable API uses:   period="Q1"  fiscalYear="2026"  date="2026-01-29"
    Legacy v3 used:    quarter=1    year=2026           date="2026-01-29 17:00:00"

    Returns None and logs a warning if required fields are missing.
    """
    symbol  = str(record.get("symbol", "")).upper()
    content = record.get("content", "").strip()
    date    = str(record.get("date", ""))[:10]  # take YYYY-MM-DD from any format

    # Quarter string — prefer stable "Q1" format, fall back to int "quarter" field
    period_raw = record.get("period") or record.get("quarter")
    if period_raw is None:
        log.warning("FMP record for %s missing period/quarter field: %r", symbol, record)
        return None
    # Normalise: int 1 → "Q1",  str "Q1" → "Q1",  str "1" → "Q1"
    period_str = str(period_raw).strip()
    if period_str.isdigit():
        quarter_str = f"Q{period_str}"
    elif period_str.upper().startswith("Q"):
        quarter_str = period_str.upper()
    else:
        log.warning("Unrecognised period value %r for %s", period_raw, symbol)
        return None

    # Fiscal year string
    fiscal_year_raw = record.get("fiscalYear") or record.get("year")
    if fiscal_year_raw is None:
        log.warning("FMP record for %s missing fiscalYear/year field: %r", symbol, record)
        return None
    fiscal_year = str(fiscal_year_raw).strip()

    if not content:
        log.warning("FMP record %s_%s_%s has empty content — skipping", symbol, quarter_str, fiscal_year)
        return None

    return {
        "symbol":      symbol,
        "quarter_str": quarter_str,   # "Q1"
        "fiscal_year": fiscal_year,   # "2026"
        "date":        date,          # "2026-01-29"
        "content":     content,
    }


# ---------------------------------------------------------------------------
# Transcript-ID derivation
# ---------------------------------------------------------------------------

def _derive_transcript_id(symbol: str, quarter_str: str, fiscal_year: str) -> str:
    """
    Build the transcript_id we expect normalize.py to produce from the
    synthesized header, so we can pre-check Pinecone before embedding.

    Mirrors normalize.py's _build_transcript_id:
        {TICKER}_{Q}_{YEAR}   →   AAPL_Q1_2026
    """
    return f"{symbol.upper()}_{quarter_str}_{fiscal_year}"


# ---------------------------------------------------------------------------
# Synthetic Motley Fool header
# ---------------------------------------------------------------------------

def _build_synthetic_text(record: dict) -> str:
    """
    Prepend a minimal Motley Fool-style preamble to the raw FMP content so
    normalize.py can extract quarter, date, and company name using its existing
    patterns without any changes to that module.

    Patterns satisfied:
    • Quarter  — title line contains "Q1 2026" → _infer_quarter Pattern 1
    • Date     — "Weekday, Month D, YYYY"      → _extract_metadata date regex
    • Company  — "AAPL (AAPL)"                 → company_pattern regex
                  _normalize_company_name then expands AAPL → "Apple Inc." etc.

    The synthesized string is only ever held in memory.
    """
    symbol      = record["symbol"]
    quarter_str = record["quarter_str"]   # "Q1"
    fiscal_year = record["fiscal_year"]   # "2026"
    date        = record["date"]          # "2026-01-29"
    content     = record["content"]

    try:
        dt        = datetime.strptime(date, "%Y-%m-%d")
        date_line = dt.strftime("%A, %B %-d, %Y")   # "Wednesday, January 29, 2026"
    except (ValueError, TypeError):
        date_line = ""
        log.warning("Could not parse date %r for %s — call_date will be empty", date, symbol)

    header = (
        f"{symbol} ({symbol}) {quarter_str} {fiscal_year} Earnings Call\n"
        f"Date\n"
        f"{date_line}\n"
        f"\n"
        f"Call participants\n"
        f"\n"
        f"Full Conference Call Transcript\n"
    )
    return header + content


# ---------------------------------------------------------------------------
# Pinecone pre-check (imported from ingest_upload to avoid duplicating logic)
# ---------------------------------------------------------------------------

def _already_in_pinecone(transcript_id: str) -> bool:
    """Thin wrapper around ingest_upload's private duplicate-check function."""
    from ingestion.ingest_upload import _transcript_exists_in_pinecone
    return _transcript_exists_in_pinecone(transcript_id)


# ---------------------------------------------------------------------------
# Core orchestrator
# ---------------------------------------------------------------------------

def run_scheduled(
    tickers:  Optional[list] = None,
    lookback: Optional[int]  = None,
    dry_run:  bool           = False,
) -> dict:
    """
    Main entry point. For each ticker in the watchlist (or override list):
      1. Fetch the most recent `lookback_quarters` transcripts from FMP
      2. Derive the expected transcript_id for each
      3. Skip any already present in Pinecone
      4. Ingest new ones via ingest_transcript()

    Parameters
    ----------
    tickers  : list[str] | None   Override watchlist.json tickers.
    lookback : int | None         Override lookback_quarters.
    dry_run  : bool               Fetch + check Pinecone but do not embed/upsert.

    Returns
    -------
    dict  Run summary with per-transcript detail list.
    """
    t_start = time.monotonic()
    watchlist_tickers, lookback_quarters = _load_watchlist(tickers, lookback)

    if not watchlist_tickers:
        log.warning("No tickers to process. Add them to watchlist.json or pass --tickers.")
        return {
            "tickers_processed": 0, "transcripts_found": 0,
            "ingested": 0, "skipped": 0, "errors": 0, "results": [],
        }

    log.info(
        "Starting scheduled ingest — tickers=%s  lookback=%d  dry_run=%s",
        watchlist_tickers, lookback_quarters, dry_run,
    )

    from ingestion.ingest_upload import ingest_transcript

    summary = {
        "tickers_processed": len(watchlist_tickers),
        "transcripts_found": 0,
        "ingested":          0,
        "skipped":           0,
        "errors":            0,
        "results":           [],
    }

    for ticker in watchlist_tickers:
        try:
            raw_records = _fetch_fmp_transcripts(ticker, lookback_quarters)
        except EnvironmentError:
            raise   # re-raise config errors — no point continuing

        if not raw_records:
            log.warning("No transcripts returned for %s — skipping ticker", ticker)
            continue

        for raw in raw_records:
            record = _parse_fmp_record(raw)
            if record is None:
                summary["errors"] += 1
                continue

            summary["transcripts_found"] += 1
            tid = _derive_transcript_id(
                record["symbol"], record["quarter_str"], record["fiscal_year"]
            )

            # --- Pre-check Pinecone to skip the embed call entirely ---
            try:
                exists = _already_in_pinecone(tid)
            except Exception as exc:
                log.error("Pinecone check failed for %s: %s", tid, exc)
                summary["errors"] += 1
                summary["results"].append({"transcript_id": tid, "status": "error", "error": str(exc)})
                continue

            if exists:
                log.info("Already in Pinecone — skipping %s", tid)
                summary["skipped"] += 1
                summary["results"].append({"transcript_id": tid, "status": "skipped",
                                            "chunks_upserted": 0, "tokens_estimated": 0,
                                            "elapsed_seconds": 0})
                continue

            if dry_run:
                log.info("[dry-run] Would ingest %s", tid)
                summary["results"].append({"transcript_id": tid, "status": "dry_run"})
                continue

            # --- Ingest ---
            try:
                raw_text = _build_synthetic_text(record)
                result   = ingest_transcript(
                    ticker   = record["symbol"],
                    raw_text = raw_text,
                    force    = False,
                )
            except Exception as exc:
                log.error("Ingest failed for %s: %s", tid, exc)
                summary["errors"] += 1
                summary["results"].append({"transcript_id": tid, "status": "error", "error": str(exc)})
                continue

            log.info(
                "Ingested %s — %d chunks  ~%d tokens  %.2fs",
                result["transcript_id"],
                result["chunks_upserted"],
                result["tokens_estimated"],
                result["elapsed_seconds"],
            )
            summary["ingested"] += 1
            summary["results"].append({
                "transcript_id":    result["transcript_id"],
                "status":           "ingested",
                "chunks_upserted":  result["chunks_upserted"],
                "tokens_estimated": result["tokens_estimated"],
                "elapsed_seconds":  result["elapsed_seconds"],
            })

    elapsed = round(time.monotonic() - t_start, 2)
    summary["elapsed_seconds"] = elapsed
    log.info(
        "Run complete — ingested=%d  skipped=%d  errors=%d  elapsed=%.2fs",
        summary["ingested"], summary["skipped"], summary["errors"], elapsed,
    )
    return summary


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point — triggered by EventBridge cron rule.

    A scheduled EventBridge event carries no payload; the handler then reads
    everything from watchlist.json and env vars. You can also invoke manually
    with an override payload to test or backfill specific quarters.
    """
    tickers  = event.get("tickers")            # None → use watchlist.json
    lookback = event.get("lookback_quarters")   # None → use watchlist.json
    dry_run  = bool(event.get("dry_run", False))

    try:
        result = run_scheduled(tickers=tickers, lookback=lookback, dry_run=dry_run)
        return {"statusCode": 200, "body": result}
    except EnvironmentError as exc:
        log.error("Configuration error: %s", exc)
        return {"statusCode": 500, "body": {"error": str(exc)}}
    except Exception as exc:
        log.error("Unexpected error: %s", exc)
        return {"statusCode": 500, "body": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch new earnings transcripts from FMP and upsert to Pinecone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ingestion/ingest_scheduled.py\n"
            "  python ingestion/ingest_scheduled.py --tickers AAPL MSFT\n"
            "  python ingestion/ingest_scheduled.py --tickers GOOGL --lookback 2 --dry-run\n"
        ),
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Override watchlist.json tickers for this run",
    )
    parser.add_argument(
        "--lookback", type=int, metavar="N",
        help=f"Most recent quarters to consider per ticker (default: {DEFAULT_LOOKBACK})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check Pinecone for new transcripts but do not embed or upsert",
    )
    args = parser.parse_args()

    try:
        result = run_scheduled(
            tickers  = args.tickers,
            lookback = args.lookback,
            dry_run  = args.dry_run,
        )
    except EnvironmentError as exc:
        log.error("%s", exc)
        sys.exit(1)

    print("\n" + "=" * 58)
    print("  SCHEDULED INGEST SUMMARY")
    print("=" * 58)
    print(f"  Tickers processed  : {result['tickers_processed']}")
    print(f"  Transcripts found  : {result['transcripts_found']}")
    print(f"  Ingested           : {result['ingested']}")
    print(f"  Skipped (exists)   : {result['skipped']}")
    print(f"  Errors             : {result['errors']}")
    print(f"  Elapsed            : {result.get('elapsed_seconds', '?')}s")
    if result["results"]:
        print("\n  Per-transcript:")
        for r in result["results"]:
            tid    = r["transcript_id"]
            status = r["status"].upper()
            if r["status"] == "ingested":
                print(f"    [{status}]   {tid}  —  {r['chunks_upserted']} chunks  ~{r['tokens_estimated']} tokens")
            elif r["status"] == "error":
                print(f"    [{status}]     {tid}  —  {r['error']}")
            else:
                print(f"    [{status}]   {tid}")
    print("=" * 58 + "\n")
