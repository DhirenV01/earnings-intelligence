"""
query.py

RAG query layer — the intelligence core of the system.

Takes a natural language question, retrieves relevant chunks from Pinecone
using semantic + metadata filtering, and calls GPT-4o to produce a grounded
answer with source citations.

Usage (local):
    python query.py --question "What did Tim Cook say about China?"
    python query.py --question "What is gross margin guidance?" --quarter "Q1 2026"
    python query.py --question "What did analysts ask about memory?" --role Analyst

Environment variables:
    OPENAI_API_KEY   -- required
    PINECONE_API_KEY -- required
    PINECONE_INDEX   -- default: earnings-intelligence
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Load .env from repo root so the script works when run directly
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env", override=False)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PINECONE_INDEX     = os.getenv("PINECONE_INDEX", "earnings-intelligence")
EMBEDDING_MODEL    = "text-embedding-3-small"  # must match what embed.py used
GPT_MODEL          = "gpt-4o"
TOP_K              = 8      # chunks to retrieve from Pinecone
MAX_CONTEXT_TOKENS = 6000   # soft cap on context sent to GPT-4o


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class QueryFilter:
    """
    Optional metadata filters applied BEFORE the vector similarity search.

    Pinecone evaluates these as exact-match conditions against stored chunk
    metadata, narrowing the candidate set before semantic ranking happens.

    This means a query for AAPL will never surface MSFT chunks even if
    they are semantically similar -- the filter eliminates them first.
    All fields are optional. Omitting a field means no restriction on it.
    """
    ticker:  Optional[str] = None   # e.g. "AAPL"
    quarter: Optional[str] = None   # e.g. "Q1 2026"
    role:    Optional[str] = None   # "CEO" | "CFO" | "Analyst"
    section: Optional[str] = None   # "prepared_remarks" | "qa"


@dataclass
class SourceChunk:
    """A single retrieved chunk with its metadata -- used for citations."""
    chunk_id: str
    speaker:  str
    role:     str
    firm:     str
    section:  str
    quarter:  str
    ticker:   str
    text:     str
    score:    float   # cosine similarity score from Pinecone (0-1)


@dataclass
class QueryResponse:
    """
    Structured response returned to the caller.

    Returning a typed object instead of a raw string means the API layer,
    a frontend, or a test can all work with the data reliably.
    """
    question:    str
    answer:      str
    sources:     list[SourceChunk]
    found:       bool    # False if the answer could not be grounded in chunks
    confidence:  str     # "high" | "medium" | "low"
    model:       str
    chunks_used: int


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def query_transcripts(
    question: str,
    filters:  Optional[QueryFilter] = None,
    top_k:    int = TOP_K,
) -> QueryResponse:
    """
    Main entry point for the RAG query pipeline.

    Orchestrates four steps:
      1. Embed the question into a vector
      2. Retrieve top-k semantically similar chunks from Pinecone
      3. Assemble those chunks into a formatted context block
      4. Call GPT-4o with the question + context and parse the response

    Parameters
    ----------
    question : str          Natural language question from the user
    filters  : QueryFilter  Optional metadata pre-filters (ticker, quarter, role, section)
    top_k    : int          How many chunks to retrieve

    Returns
    -------
    QueryResponse  Structured answer with citations and confidence
    """
    filters = filters or QueryFilter()

    # Step 1 -- embed the question into the same vector space as our chunks
    query_vector = _embed_query(question)

    # Step 2 -- hit Pinecone: filter first, then rank by semantic similarity
    chunks = _retrieve_chunks(query_vector, filters, top_k)

    # If nothing comes back (index empty, or filters too narrow), return early
    if not chunks:
        return QueryResponse(
            question    = question,
            answer      = "No relevant content found for this question in the indexed transcripts.",
            sources     = [],
            found       = False,
            confidence  = "low",
            model       = GPT_MODEL,
            chunks_used = 0,
        )

    # Step 3 -- format chunks into a readable context block with source labels
    context = _assemble_context(chunks)

    # Step 4 -- call GPT-4o, grounded strictly on the retrieved context
    answer, found, confidence = _call_gpt(question, context)

    return QueryResponse(
        question    = question,
        answer      = answer,
        sources     = chunks,
        found       = found,
        confidence  = confidence,
        model       = GPT_MODEL,
        chunks_used = len(chunks),
    )


# ---------------------------------------------------------------------------
# Step 1 -- Embed the query
# ---------------------------------------------------------------------------

def _embed_query(question: str) -> list[float]:
    """
    Convert the user's question into a vector using the same embedding model
    that was used to embed the transcript chunks in embed.py.

    This is non-negotiable: if the query and chunk vectors come from different
    models, cosine similarity scores are meaningless. They must share the
    same vector space.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    client   = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(
        model = EMBEDDING_MODEL,
        input = [question],
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Step 2 -- Retrieve from Pinecone with metadata filters
# ---------------------------------------------------------------------------

def _build_pinecone_filter(filters: QueryFilter) -> Optional[dict]:
    """
    Translate a QueryFilter into Pinecone's metadata filter syntax.

    Pinecone uses MongoDB-style operators. We only add conditions for fields
    that were explicitly set -- omitting a field means "match anything."

    Example output for QueryFilter(ticker="AAPL", role="CEO"):
        {"ticker": {"$eq": "AAPL"}, "role": {"$eq": "CEO"}}

    This pre-filter runs before ANN search, so it's fast and doesn't degrade
    retrieval quality -- it just narrows the candidate pool.
    """
    conditions = {}

    if filters.ticker:
        conditions["ticker"] = {"$eq": filters.ticker.upper()}
    if filters.quarter:
        conditions["quarter"] = {"$eq": filters.quarter}
    if filters.role:
        conditions["role"] = {"$eq": filters.role}
    if filters.section:
        conditions["section"] = {"$eq": filters.section}

    return conditions if conditions else None


def _retrieve_chunks(
    query_vector: list[float],
    filters:      QueryFilter,
    top_k:        int,
) -> list[SourceChunk]:
    """
    Query Pinecone for the top-k most semantically similar chunks,
    filtered by metadata before ranking.

    include_metadata=True is critical -- without it Pinecone only returns
    the vector IDs and scores. We need the stored text and speaker info
    to build citations and assemble context for GPT-4o.
    """
    try:
        from pinecone import Pinecone
    except ImportError:
        raise ImportError("Run: pip install pinecone-client")

    pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(PINECONE_INDEX)

    pinecone_filter = _build_pinecone_filter(filters)

    response = index.query(
        vector           = query_vector,
        top_k            = top_k,
        include_metadata = True,
        filter           = pinecone_filter,
    )

    chunks = []
    for match in response.matches:
        meta = match.metadata
        chunks.append(SourceChunk(
            chunk_id = match.id,
            speaker  = meta.get("speaker", "Unknown"),
            role     = meta.get("role", "Unknown"),
            firm     = meta.get("firm", ""),
            section  = meta.get("section", ""),
            quarter  = meta.get("quarter", ""),
            ticker   = meta.get("ticker", ""),
            text     = meta.get("text", ""),
            score    = round(match.score, 4),
        ))

    return chunks


# ---------------------------------------------------------------------------
# Step 3 -- Assemble context block
# ---------------------------------------------------------------------------

def _assemble_context(chunks: list[SourceChunk]) -> str:
    """
    Format retrieved chunks into a labeled context block for GPT-4o.

    Each chunk gets a [SOURCE N] header with full provenance metadata so
    GPT-4o can produce accurate citations. Chunks are ordered by relevance
    score (highest first) so the most relevant content leads.

    We enforce a soft token cap to avoid blowing the context window.
    Chunks that would exceed the cap are truncated rather than dropped,
    so we always pass some content rather than none.
    """
    # Sort highest relevance first
    chunks_sorted = sorted(chunks, key=lambda c: c.score, reverse=True)

    context_parts = []
    total_chars   = 0
    char_cap      = MAX_CONTEXT_TOKENS * 4  # ~4 chars per token for English prose

    for i, chunk in enumerate(chunks_sorted):
        # Build a rich source header so GPT-4o knows exactly who said what
        firm_label = f", {chunk.firm}" if chunk.firm else ""
        header = (
            f"[SOURCE {i+1}] "
            f"{chunk.ticker} {chunk.quarter} | "
            f"{chunk.speaker} ({chunk.role}{firm_label}) | "
            f"{chunk.section.replace('_', ' ').title()} | "
            f"score={chunk.score}"
        )
        block = f"{header}\n{chunk.text}"

        if total_chars + len(block) > char_cap:
            remaining = char_cap - total_chars
            if remaining > 300:   # only include if there's enough room to be useful
                block = block[:remaining] + "... [truncated]"
                context_parts.append(block)
            break

        context_parts.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Step 4 -- Call GPT-4o
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a financial analyst assistant specializing in earnings call analysis.

You answer questions strictly based on the provided transcript excerpts.
Each excerpt is labeled [SOURCE N] with the company, quarter, speaker, role, and relevance score.

Rules you must always follow:
1. Answer ONLY from the provided excerpts. Do not use outside knowledge or training data.
2. Always cite sources using [SOURCE N] labels inline in your answer.
3. When referencing what someone said, include their name and role.
4. If the excerpts do not contain enough information to answer, say so clearly and explain what is missing.
5. Be precise and concise -- this is financial analysis, not a summary.
6. At the very end of your response, output this JSON block and nothing after it:
   ```json
   {"found": true, "confidence": "high"}
   ```
   - found: true if the excerpts contained relevant information, false if not
   - confidence: "high" if directly addressed, "medium" if partially, "low" if inferred
"""


def _call_gpt(
    question: str,
    context:  str,
) -> tuple[str, bool, str]:
    """
    Call GPT-4o with the user question and assembled context.

    Returns
    -------
    answer     : str   The model's grounded answer
    found      : bool  Whether the answer was grounded in retrieved chunks
    confidence : str   "high" | "medium" | "low"
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    user_message = f"""
TRANSCRIPT EXCERPTS:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model    = GPT_MODEL,
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature = 0.1,   # low temp for factual financial analysis
        max_tokens  = 1000,
    )

    raw_answer = response.choices[0].message.content.strip()

    # Parse the JSON metadata block the model appends
    found, confidence = _parse_response_metadata(raw_answer)

    # Strip the JSON block from the displayed answer
    answer = _strip_json_block(raw_answer)

    return answer, found, confidence


def _parse_response_metadata(raw: str) -> tuple[bool, str]:
    """
    Extract the found/confidence JSON block from the model's response.
    Falls back to sensible defaults if parsing fails.
    """
    import re
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            data       = json.loads(match.group(1))
            found      = bool(data.get("found", True))
            confidence = data.get("confidence", "medium")
            return found, confidence
        except json.JSONDecodeError:
            pass
    return True, "medium"


def _strip_json_block(raw: str) -> str:
    """Remove the trailing JSON metadata block from the answer text."""
    import re
    return re.sub(r"```json\s*\{.*?\}\s*```", "", raw, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point. Sits behind API Gateway POST /query.

    Expected event body:
    {
        "question": "What did Tim Cook say about China?",
        "ticker":   "AAPL",
        "quarter":  "Q1 2026",
        "role":     "CEO",
        "section":  "qa"
    }
    """
    try:
        body     = json.loads(event.get("body", "{}"))
        question = body.get("question", "").strip()

        if not question:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "question is required"})
            }

        filters = QueryFilter(
            ticker  = body.get("ticker"),
            quarter = body.get("quarter"),
            role    = body.get("role"),
            section = body.get("section"),
        )

        result = query_transcripts(question, filters)

        # Serialize response -- SourceChunk objects need manual conversion
        response_body = {
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
                    "text":     s.text[:300] + "..." if len(s.text) > 300 else s.text,
                }
                for s in result.sources
            ]
        }

        return {
            "statusCode": 200,
            "headers":    {"Content-Type": "application/json"},
            "body":       json.dumps(response_body),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


# ---------------------------------------------------------------------------
# CLI entry point -- test locally without deploying
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query earnings call transcripts")
    parser.add_argument("--question", required=True, help="Your question")
    parser.add_argument("--ticker",   help="Filter by ticker, e.g. AAPL")
    parser.add_argument("--quarter",  help='Filter by quarter, e.g. "Q1 2026"')
    parser.add_argument("--role",     help="Filter by role: CEO | CFO | Analyst")
    parser.add_argument("--section",  help="Filter by section: prepared_remarks | qa")
    parser.add_argument("--top-k",   type=int, default=TOP_K, help="Chunks to retrieve")
    args = parser.parse_args()

    filters = QueryFilter(
        ticker  = args.ticker,
        quarter = args.quarter,
        role    = args.role,
        section = args.section,
    )

    print(f"\nQuestion : {args.question}")
    print(f"Filters  : ticker={filters.ticker} | quarter={filters.quarter} | role={filters.role} | section={filters.section}")
    print(f"Retrieving top {args.top_k} chunks...\n")

    result = query_transcripts(args.question, filters, top_k=args.top_k)

    print("=" * 60)
    print(f"ANSWER (found={result.found}, confidence={result.confidence})")
    print("=" * 60)
    print(result.answer)

    print(f"\n--- Sources ({result.chunks_used} chunks used) ---")
    for s in result.sources:
        firm = f", {s.firm}" if s.firm else ""
        print(f"  [{s.score}] {s.speaker} ({s.role}{firm}) | {s.ticker} {s.quarter} | {s.section}")
