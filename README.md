# earnings-intelligence

![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-000000?logo=pinecone&logoColor=white)

A production-style RAG pipeline for querying earnings call transcripts. Transcripts are normalized into structured speaker segments, embedded, and stored in a vector index with rich metadata — enabling natural language questions like *"What did the CFO say about gross margin guidance?"* or *"What questions did analysts ask about memory pricing?"* to return grounded, cited answers from actual speaker quotes.

Built to demonstrate data engineering (medallion architecture, structured ingest, deduplication), AI engineering (RAG, vector search, metadata filtering), and financial domain knowledge (speaker attribution, earnings call structure, cross-company analysis).

---

## Architecture

```
Raw .txt transcript
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  BRONZE LAYER  (local data/samples/ → S3 planned)     │
│  Raw Motley Fool transcript files, unmodified         │
└───────────────────────────┬───────────────────────────┘
                            │
              pipeline/normalize.py
              • Strip preamble, parse header
              • Extract speaker segments w/ role/firm
              • Tag section: prepared_remarks | qa
              • Extract Industry Glossary terms
                            │
                            ▼
┌───────────────────────────────────────────────────────┐
│  SILVER LAYER  (in-memory dict → S3 planned)          │
│  Structured JSON: segments[], glossary[], metadata    │
└───────────────────────────┬───────────────────────────┘
                            │
               pipeline/embed.py
               • Merge short segments
               • Sliding window chunking (512 tok, 50 overlap)
               • Embed via text-embedding-3-small
               • Upsert to Pinecone w/ full metadata
                            │
                            ▼
┌───────────────────────────────────────────────────────┐
│  GOLD LAYER  (Pinecone serverless index)              │
│  Vectors + metadata: ticker, quarter, speaker,        │
│  role, firm, section, chunk_id, text                  │
└───────────────────────────┬───────────────────────────┘
                            │
               query/query.py
               • Embed question (same model)
               • Pinecone metadata pre-filter
               • Assemble context w/ [SOURCE N] labels
               • GPT-4o grounded answer + citations
                            │
                            ▼
                    QueryResponse
             answer | sources[] | confidence


INGEST PATHS
────────────

Manual / API upload                  Scheduled (FMP API)
        │                                     │
ingest_upload.py                   ingest_scheduled.py
  CLI or function call               FMP stable API poll
  POST /ingest (FastAPI)             watchlist.json tickers
        │                                     │
        └─────────────┬───────────────────────┘
                      ▼
          normalize → embed → Pinecone
              (same pipeline, both paths)
```

---

## Key Design Decisions

**Medallion architecture** — Each layer has a defined schema and clear transformation responsibility. Bronze is immutable source files. Silver is normalized JSON with speaker-level structure. Gold is the queryable vector index. Adding a new transcript source (e.g. SEC EDGAR) only requires adapting the Bronze → Silver step; downstream is untouched.

**Speaker-level metadata on every chunk** — Each Pinecone vector carries `ticker`, `quarter`, `speaker`, `role`, `firm`, and `section`. This enables metadata pre-filtering before the ANN search, so a query filtered to `role=CFO, section=qa` only ranks CFO Q&A chunks — not semantically similar CEO remarks or analyst questions. Filtered queries are both more accurate and cheaper.

**Glossary extraction with data provenance** — Industry Glossary terms are stored as separate chunks with `section="glossary"` and `speaker="editorial"`. The system never conflates an editorial definition with an actual speaker quote. In financial contexts, attribution accuracy is material — a definition that surfaces as a CEO quote would be a meaningful error.

**Dual ingest paths converging at a single pipeline** — `ingest_upload.py` handles manual uploads and the FastAPI `/ingest` endpoint. `ingest_scheduled.py` polls the FMP API for a watchlist of tickers and builds synthetic Motley Fool-style headers so `normalize.py` runs unchanged for both sources. Both paths call the same `normalize → embed → upsert` functions.

**Duplicate detection before embedding** — `ingest_upload.py` queries Pinecone for the `transcript_id` using a zero vector and metadata filter before any OpenAI call is made. `ingest_scheduled.py` replicates this check at the FMP record level before constructing the synthetic text. Prevents redundant embed costs and index pollution on re-runs.

**Lambda/local parity via Mangum** — The FastAPI app is wrapped with a Mangum handler at the bottom of `api/main.py`. Local dev runs via `uvicorn`. Lambda runs via API Gateway with the same handler. No branching, no environment flags.

---

## Project Structure

```
earnings-intelligence/
│
├── pipeline/
│   ├── normalize.py          # Bronze → Silver: raw .txt → structured JSON w/ speaker segments
│   └── embed.py              # Silver → Gold: JSON → chunks → embeddings → Pinecone upsert
│
├── ingestion/
│   ├── ingest_upload.py      # Manual ingest: CLI + callable function for FastAPI / Lambda
│   └── ingest_scheduled.py   # Scheduled ingest: FMP API poll → normalize → embed pipeline
│
├── query/
│   └── query.py              # RAG query layer: question → Pinecone → GPT-4o → cited answer
│
├── api/
│   └── main.py               # FastAPI app: POST /ingest, POST /query, GET /health + Mangum
│
├── data/
│   └── samples/
│       ├── aapl_q1_2026.txt  # Apple Q1 FY2026 earnings call (Motley Fool format)
│       └── msft_q2_2026.txt  # Microsoft Q2 FY2026 earnings call
│
├── infra/
│   └── template.yaml         # SAM/CloudFormation template (Lambda + API Gateway + EventBridge)
│
├── tests/
│   ├── test_normalize.py     # Unit tests for normalization pipeline
│   ├── test_embed.py         # Unit tests for chunking and embed logic
│   └── test_query.py         # Integration tests for query layer
│
├── scripts/
│   └── cleanup_pinecone.py   # Utility: delete all vectors for a given transcript_id
│
├── watchlist.json            # Tickers + lookback_quarters for ingest_scheduled.py
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed)
└── CLAUDE.md                 # Project context for Claude Code
```

---

## Pinecone Metadata Schema

Every vector in the index carries these fields, enabling precise pre-filtered retrieval:

| Field           | Type    | Example                 | Notes                                               |
|-----------------|---------|-------------------------|-----------------------------------------------------|
| `transcript_id` | string  | `AAPL_Q1_2026`          | Primary dedup key                                   |
| `ticker`        | string  | `AAPL`                  | Always uppercase                                    |
| `quarter`       | string  | `Q1 2026`               | Extracted from transcript header                    |
| `fiscal_year`   | integer | `2026`                  | From call date                                      |
| `call_date`     | string  | `2026-01-29`            | ISO 8601                                            |
| `speaker`       | string  | `Timothy D. Cook`       | Normalized speaker name; `"editorial"` for glossary |
| `role`          | string  | `CEO`                   | CEO / CFO / COO / Analyst / Operator / editorial    |
| `firm`          | string  | `Morgan Stanley`        | Company for internal speakers; analyst firm for external |
| `section`       | string  | `prepared_remarks`      | `prepared_remarks` / `qa` / `glossary`              |
| `segment_index` | integer | `3`                     | Parent segment index in Silver JSON                 |
| `chunk_index`   | integer | `0`                     | Position within parent segment                      |
| `text`          | string  | `"Revenue grew 16%..."` | Full chunk text, stored for retrieval display       |
| `token_count`   | integer | `487`                   | Approximate (len(text) // 4)                        |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourhandle/earnings-intelligence.git
cd earnings-intelligence
pip install openai pinecone-client python-dotenv fastapi uvicorn python-multipart mangum
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in:
# OPENAI_API_KEY=sk-...
# PINECONE_API_KEY=pcsk_...
# PINECONE_INDEX=earnings-intelligence
# FMP_API_KEY=...          # Financial Modeling Prep (paid plan required for transcripts)
```

### 3. Ingest a transcript

```bash
# Ingest Apple Q1 2026 (runs normalize → embed → Pinecone upsert)
python ingestion/ingest_upload.py data/samples/aapl_q1_2026.txt AAPL

# Re-ingest and overwrite existing vectors
python ingestion/ingest_upload.py data/samples/aapl_q1_2026.txt AAPL --force
```

```
transcript_id=AAPL_Q1_2026  segments=85  quarter=Q1 2026
Upserted 58/58 vectors... (51 transcript + 7 glossary)
```

### 4. Query from the CLI

```bash
python query/query.py --question "What is Apple Intelligence?"
python query/query.py --question "What did the CFO say about gross margin?" --ticker AAPL --role CFO
python query/query.py --question "What questions did analysts ask about memory pricing?" --section qa
```

### 5. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

```bash
# Upload a transcript
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/samples/aapl_q1_2026.txt" \
  -F "ticker=AAPL"

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Apple Intelligence?", "ticker": "AAPL"}'

# Health check
curl http://localhost:8000/health
```

### 6. Scheduled ingest

```bash
# Check watchlist.json tickers against FMP, ingest anything new
python ingestion/ingest_scheduled.py

# Dry-run: show what would be ingested without making any embed calls
python ingestion/ingest_scheduled.py --dry-run

# One-off backfill for specific tickers
python ingestion/ingest_scheduled.py --tickers GOOGL NVDA --lookback 2
```

---

## Example Queries

### Glossary + transcript blending

**Q:** *What is Apple Intelligence?*

```
ANSWER  (found=True, confidence=high)

Apple Intelligence is a suite of AI features and foundational models integrated
into Apple devices, encompassing on-device and private cloud computation for
personalized user experiences [SOURCE 1]. These AI experiences are designed to
be personal, private, and integrated across Apple's platforms [SOURCE 2].
Apple is collaborating with Google to develop the next generation of Apple
foundation models to power future Apple Intelligence features [SOURCE 4].

SOURCES
  [0.7225] editorial (glossary)                  | AAPL Q1 2026
  [0.6018] Timothy D. Cook (CEO, Apple Inc.)     | AAPL Q1 2026 | prepared_remarks
  [0.5453] Richard Kramer (Analyst, Arete)       | AAPL Q1 2026 | qa
  [0.5372] Timothy D. Cook (CEO, Apple Inc.)     | AAPL Q1 2026 | prepared_remarks
```

The glossary chunk surfaces as SOURCE 1 at score 0.7225 — the highest-ranked result — because the query is definitional. The system correctly distinguishes the editorial definition from Cook's remarks rather than blending them.

---

### Cross-company synthesis

**Q:** *What did Apple and Microsoft both say about AI monetization this quarter?*

```
ANSWER  (found=True, confidence=medium)

For Microsoft, Amy Hood (CFO) noted AI infrastructure investment is pressuring
gross margins, partially offset by efficiency gains in Azure and Microsoft 365
Copilot [SOURCE 1]. Satya Nadella (CEO) highlighted that their AI business is
now larger than some of their biggest franchises [SOURCE 2].

For Apple, Timothy D. Cook (CEO) discussed collaboration with Google on
next-generation foundation models to power future Apple Intelligence features
[SOURCE 4]. Suhasini Chandramouli (IR Director) acknowledged added costs from
AI initiatives without detailing specific monetization timelines [SOURCE 5].

SOURCES
  [0.5824] Amy Hood (CFO, Microsoft Corporation)       | MSFT Q2 2026 | prepared_remarks
  [0.5685] Satya Nadella (CEO, Microsoft Corporation)  | MSFT Q2 2026 | prepared_remarks
  [0.5684] Satya Nadella (CEO, Microsoft Corporation)  | MSFT Q2 2026 | prepared_remarks
  [0.5641] Timothy D. Cook (CEO, Apple Inc.)           | AAPL Q1 2026 | prepared_remarks
  [0.5623] Suhasini Chandramouli (IR, Apple Inc.)      | AAPL Q1 2026 | qa
```

No ticker filter applied — the retrieval layer surfaces relevant chunks from both companies and GPT-4o synthesizes across them with per-company attribution.

---

### Metadata-filtered analyst attribution

**Q:** *What questions did analysts ask about memory pricing?*  `--section qa`

```
ANSWER  (found=True, confidence=high)

1. Wamsi Mohan (Bank of America) asked whether Apple would use pricing as a
   lever in response to unprecedented moves in memory prices [SOURCE 1].

2. Amit Daryanani (Evercore) asked about Apple's comfort securing necessary
   memory for shipments and how memory inflation might affect Apple's model
   over time [SOURCE 2].

3. Krish Sankar asked whether Apple could gain iPhone/Mac market share due to
   its purchasing power amid memory constraints [SOURCE 4].

4. David Locke (UBS) asked whether Apple is pursuing long-term agreements or
   spot-based options for memory procurement [SOURCE 5].

5. Ben Reitzes (Melius) questioned how Apple maintains 48-49% gross margin
   despite rising NAND prices — services mix vs. hardware [SOURCE 6].

SOURCES
  [0.6091] Wamsi Mohan (Analyst, Bank of America)  | AAPL Q1 2026 | qa
  [0.5657] Amit Daryanani (Analyst, Evercore)       | AAPL Q1 2026 | qa
  [0.5309] Timothy D. Cook (CEO, Apple Inc.)        | AAPL Q1 2026 | qa
  [0.5088] David Locke (Analyst, UBS)               | AAPL Q1 2026 | qa
  [0.4767] Ben Reitzes (Analyst, Melius)            | AAPL Q1 2026 | qa
```

`--section qa` pre-filters the Pinecone candidate pool to Q&A chunks only before semantic ranking. Five distinct analysts with firm attribution — the role map built by `normalize.py` from the call participants block and operator introduction lines is what enables this.

---

## API Reference

### `POST /ingest`

Multipart form upload. Runs the full normalize → embed → upsert pipeline.

| Field    | Type    | Required | Description                             |
|----------|---------|----------|-----------------------------------------|
| `file`   | file    | yes      | `.txt` earnings call transcript         |
| `ticker` | string  | yes      | Equity ticker symbol (e.g. `AAPL`)      |
| `force`  | boolean | no       | Overwrite if transcript already exists  |

```json
{
  "transcript_id":    "AAPL_Q1_2026",
  "chunks_upserted":  58,
  "tokens_estimated": 12223,
  "elapsed_seconds":  6.5,
  "skipped":          false
}
```

### `POST /query`

| Field      | Type   | Required | Description                                      |
|------------|--------|----------|--------------------------------------------------|
| `question` | string | yes      | Natural language question                        |
| `ticker`   | string | no       | Filter to a single company (e.g. `AAPL`)         |
| `quarter`  | string | no       | Filter to a quarter (e.g. `Q1 2026`)             |
| `role`     | string | no       | Filter by speaker role (`CEO`, `CFO`, `Analyst`) |
| `section`  | string | no       | Filter by section (`prepared_remarks`, `qa`)     |

```json
{
  "question":    "What is Apple Intelligence?",
  "answer":      "Apple Intelligence is a suite of AI features...",
  "found":       true,
  "confidence":  "high",
  "model":       "gpt-4o",
  "chunks_used": 8,
  "sources": [
    {
      "chunk_id": "AAPL_Q1_2026_glossary_c0",
      "speaker":  "editorial",
      "role":     "editorial",
      "section":  "glossary",
      "ticker":   "AAPL",
      "quarter":  "Q1 2026",
      "score":    0.7225,
      "text":     "Apple Intelligence: Suite of AI features..."
    }
  ]
}
```

### `GET /health`

```json
{
  "status": "ok",
  "env_vars_set": ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"]
}
```

---

## Roadmap

- [ ] **EventBridge trigger** — weekly Lambda invocation of `ingest_scheduled.py` post-earnings season
- [ ] **S3 Bronze layer** — replace local `.txt` files with S3 ObjectCreated trigger on `bronze/` prefix
- [ ] **Full SAM deployment** — Lambda functions for ingest and query behind API Gateway
- [ ] **SEC EDGAR integration** — ingest 10-K and 10-Q filings as an additional transcript source
- [ ] **Frontend query UI** — lightweight interface for interactive transcript Q&A

---

## Tech Stack

| Component         | Technology                                           |
|-------------------|------------------------------------------------------|
| Language          | Python 3.9                                           |
| API framework     | FastAPI + Uvicorn                                    |
| Lambda adapter    | Mangum                                               |
| Embeddings        | OpenAI `text-embedding-3-small` (1536-dim)           |
| LLM               | OpenAI `gpt-4o`                                      |
| Vector store      | Pinecone serverless                                  |
| Transcript source | Financial Modeling Prep API                          |
| Config            | python-dotenv                                        |
| Infra (planned)   | AWS Lambda + API Gateway + S3 + EventBridge          |
| IaC (planned)     | AWS SAM / CloudFormation                             |
