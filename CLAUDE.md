# earnings-intelligence — Claude Context

## What this project is

A RAG pipeline for querying earnings call transcripts. Raw .txt files are normalized into structured JSON, chunked and embedded, upserted to Pinecone, and served through a query layer that returns GPT-4o answers grounded in transcript excerpts with citations.

---

## Project structure

```
earnings-intelligence/
├── pipeline/
│   ├── normalize.py        # Bronze → Silver: raw .txt → structured JSON
│   └── embed.py            # Silver → Gold: JSON → chunks → embeddings → Pinecone
├── ingestion/
│   ├── ingest_upload.py    # Manual ingest CLI + callable function (COMPLETE)
│   └── ingest_scheduled.py # Scheduled/automated ingest (EMPTY — not started)
├── query/
│   └── query.py            # RAG query layer: question → Pinecone → GPT-4o → answer
├── data/
│   └── samples/
│       ├── aapl_q1_2026.txt  # Motley Fool format with full header
│       └── msft_q2_2026.txt  # Header manually prepended (see gotchas)
├── infra/
│   └── template.yaml       # SAM/CloudFormation template (EMPTY — not started)
├── tests/
│   ├── test_normalize.py   # EMPTY
│   ├── test_embed.py       # EMPTY
│   └── test_query.py       # EMPTY
├── README.md               # EMPTY
├── requirements.txt        # EMPTY
├── .env                    # OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX
└── CLAUDE.md               # this file
```

---

## Medallion architecture

The design follows Bronze → Silver → Gold, currently running locally. S3 is the intended future home for each layer.

| Layer  | Current location     | Future location        | Content                              |
|--------|----------------------|------------------------|--------------------------------------|
| Bronze | `data/samples/*.txt` | `s3://bucket/bronze/`  | Raw Motley Fool .txt transcript files |
| Silver | In-memory dict       | `s3://bucket/silver/`  | Normalized JSON (segments, metadata) |
| Gold   | Pinecone index       | Pinecone + S3 backup   | Embedded chunks with metadata        |

The Silver layer is not currently persisted to disk between pipeline stages — it flows in-memory from `normalize_transcript()` directly into `embed_transcript()`. If you want to inspect or cache Silver output, `normalized_output.json` is written by `normalize.py`'s `__main__` block.

---

## Pipeline flow

```
raw .txt
   │
   ▼
normalize_transcript(raw_text, ticker)          ← pipeline/normalize.py
   │  returns Silver dict:
   │    transcript_id, ticker, company, quarter,
   │    fiscal_year, call_date, participants, segments[]
   │
   ▼
embed_transcript(silver)                        ← pipeline/embed.py
   │  1. merge short segments (< 80 tokens folded into next)
   │  2. sliding window chunking (512 tokens, 50 overlap)
   │  3. embed via text-embedding-3-small (OpenAI)
   │  4. upsert to Pinecone in batches of 100
   │
   ▼
Pinecone index: earnings-intelligence
   │  each vector carries full metadata (see Metadata schema below)
   │
   ▼
query_transcripts(question, filters)            ← query/query.py
   │  1. embed question (same model — must match)
   │  2. Pinecone query with metadata pre-filter
   │  3. assemble context block with [SOURCE N] labels
   │  4. GPT-4o call (temp=0.1, max_tokens=1000)
   │  returns QueryResponse with answer + SourceChunk citations
```

---

## Dual ingest design

Two paths both funnel into the same `ingest_transcript()` core function in `ingest_upload.py`.

**Manual / upload path** (`ingest_upload.py`) — fully implemented:
- CLI: `python ingestion/ingest_upload.py data/samples/aapl_q1_2026.txt AAPL`
- `--force` flag overwrites existing vectors for the same `transcript_id`
- Callable as a function for FastAPI and Lambda:
  ```python
  from ingestion.ingest_upload import ingest_transcript
  result = ingest_transcript(ticker="AAPL", file_path="path/to/file.txt")
  result = ingest_transcript(ticker="AAPL", raw_text="...")   # Lambda / HTTP variant
  ```
- Returns a summary dict: `{transcript_id, chunks_upserted, tokens_estimated, elapsed_seconds, skipped}`

**Scheduled path** (`ingest_scheduled.py`) — empty, not started:
- Will import and call `ingest_transcript()` on a schedule
- Intended trigger: EventBridge cron on AWS Lambda, or a local cron job
- Will need to fetch new transcripts from a source (TBD: Motley Fool scrape, paid data feed, or manual S3 drop)

---

## Metadata schema in Pinecone

Every vector stored in the `earnings-intelligence` index carries these metadata fields. All fields must be non-null — `embed.py` coerces `None` to `""` or `0` before upsert (Pinecone rejects null values).

| Field           | Type    | Example                   | Notes                                     |
|-----------------|---------|---------------------------|-------------------------------------------|
| `transcript_id` | string  | `AAPL_Q1_2026`            | Primary grouping key; used for dedup check |
| `ticker`        | string  | `AAPL`                    | Always uppercase                          |
| `quarter`       | string  | `Q1 2026`                 | Extracted from transcript header          |
| `fiscal_year`   | integer | `2026`                    | Extracted from call date                  |
| `call_date`     | string  | `2026-01-29`              | ISO 8601                                  |
| `speaker`       | string  | `Timothy D. Cook`         | Normalized from transcript                |
| `role`          | string  | `CEO`                     | CEO / CFO / COO / IR Director / Analyst / Operator |
| `firm`          | string  | `Morgan Stanley`          | Company for internal speakers; analyst firm for external |
| `section`       | string  | `prepared_remarks`        | `prepared_remarks` or `qa`                |
| `segment_index` | integer | `3`                       | Index of parent segment in Silver JSON    |
| `chunk_index`   | integer | `0`                       | Position of chunk within its parent segment |
| `text`          | string  | `"Revenue grew 16%..."`   | Full chunk text, stored for retrieval display |
| `token_count`   | integer | `487`                     | Approximate (len(text) // 4)              |

---

## Chunk ID convention

```
{transcript_id}_seg{segment_index}_c{chunk_index}
```

Example: `AAPL_Q1_2026_seg3_c0`, `AAPL_Q1_2026_seg3_c1`

- Globally unique across all transcripts in the index
- Deterministic: re-running the same transcript produces the same IDs
- `--force` relies on this — Pinecone upsert with the same ID overwrites

---

## Normalization conventions

**Preamble stripping**: `normalize.py` looks for the literal string `"Full Conference Call Transcript"` as the dividing line. Everything before it (Takeaways, Risks, Summary, Industry Glossary, Call Participants block) is used for metadata extraction only and is not embedded. The actual transcript starts at `match.end()` after that marker.

**Participant extraction** (two passes):
1. Parses the `Call participants` block for internal speakers (CEO, CFO, etc.) and known analysts
2. Scans the transcript body for operator introduction lines (`"Our next question is from NAME of FIRM"`) to catch analysts not listed in the header

**Section tagging**: Segments are tagged `prepared_remarks` until either (a) the Operator announces Q&A ("open the call for questions"), or (b) a speaker with `role=Analyst` appears — whichever comes first.

**Company name normalization**: Short names extracted from the text (e.g. "Apple") are expanded to formal registered names (e.g. "Apple Inc.") via `_FORMAL_COMPANY_NAMES` dict in `normalize.py`. Add new companies there as needed.

**Quarter inference** (priority order):
1. Explicit `Q1 2026` in the title line (first 5 lines)
2. Named month in the transcript body (`"December quarter"`)
3. Current-quarter language (`"our Q1 2026 results"`)
4. Calendar fallback from the call date

---

## What is complete vs. empty

| File                          | Status    | Notes                                              |
|-------------------------------|-----------|---------------------------------------------------|
| `pipeline/normalize.py`       | Complete  | Tested against AAPL Q1 2026 (85 segments)         |
| `pipeline/embed.py`           | Complete  | Tested; coerces None metadata before upsert       |
| `ingestion/ingest_upload.py`  | Complete  | CLI + callable function; dedup check; --force     |
| `query/query.py`              | Complete  | Lambda handler stub included                      |
| `ingestion/ingest_scheduled.py` | Empty   | Not started                                       |
| `infra/template.yaml`         | Empty     | SAM template not started                          |
| `tests/test_normalize.py`     | Empty     | Not started                                       |
| `tests/test_embed.py`         | Empty     | Not started                                       |
| `tests/test_query.py`         | Empty     | Not started                                       |
| `README.md`                   | Empty     | Architecture diagram planned here                 |
| `requirements.txt`            | Empty     | Known deps: openai, pinecone-client, python-dotenv |

**Pinecone index state**: `earnings-intelligence` currently holds AAPL Q1 2026 (51 chunks) and MSFT Q2 2026 (30 chunks).

---

## Upcoming work this feeds into

- **`ingest_scheduled.py`** — imports `ingest_transcript()` from `ingest_upload.py`, runs on schedule
- **FastAPI `/upload` endpoint** — calls `ingest_transcript(ticker, raw_text=...)` after receiving a file via HTTP POST
- **Lambda handler** — wraps `ingest_transcript()` triggered by S3 ObjectCreated on `bronze/` prefix
- **README** — architecture diagram referencing `ingest_upload.py` as the manual ingest path in the dual-ingest design

---

## Known gotchas

**Python < 3.10 type hint syntax**: `str | None` union syntax fails at runtime on Python 3.9 (which is what this repo runs on via Anaconda). Use `Optional[str]` from `typing` everywhere. All current code is fixed, but watch for this when adding new functions.

**Pinecone rejects None metadata values**: If `normalize.py` can't extract a field (e.g. `call_date` when the transcript header is missing), it returns `None`. Pinecone raises a 400 Bad Request on upsert if any metadata value is null. `embed.py`'s `_upsert_to_pinecone()` coerces all nullable fields with `or ""` / `or 0` before building the vector payload. Don't remove those guards.

**Transcript files must have the Motley Fool header**: `normalize.py` depends on the header block for date, quarter, company name, and participants. `msft_q2_2026.txt` was originally just the raw transcript body (starting at `"Full Conference Call Transcript"`) with no header — it normalized to `transcript_id=MSFT_UNKNOWN` and `call_date=None`. The header was manually prepended. Any new transcript file that is missing the header will exhibit the same behavior silently — the pipeline will succeed but produce empty/UNKNOWN metadata fields.

**Duplicate check uses a zero vector**: `ingest_upload.py` detects duplicates by querying Pinecone with a `[0.0] * 1536` dummy vector and a `transcript_id` metadata filter. This works because we only care about filter hits, not ANN ranking. If Pinecone changes behavior around zero-vector queries this may need to change.

**`sys.path` injection in `ingest_upload.py`**: The script inserts the repo root into `sys.path` at import time so `pipeline` is importable regardless of working directory. This is intentional — don't remove it.

**Embedding model must match at query time**: `embed.py` uses `text-embedding-3-small`. `query.py` must use the same model. Both constants are named `EMBEDDING_MODEL`. If you ever re-embed with a different model, you must reindex everything — mixing models in the same Pinecone index produces garbage similarity scores.
