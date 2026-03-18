"""
embed.py

Silver → Gold layer.

Takes a normalized Silver JSON record (output of normalize.py), splits each
segment into overlapping token-aware chunks, embeds them via OpenAI, and
upserts to Pinecone.

Usage (local):
    python embed.py normalized_output.json --dry-run   # chunk + print, no API calls
    python embed.py normalized_output.json             # full embed + upsert

Environment variables:
    OPENAI_API_KEY   — required unless --dry-run
    PINECONE_API_KEY — required unless --dry-run
    PINECONE_INDEX   — Pinecone index name (default: earnings-intelligence)
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE         = 512    # target tokens per chunk
CHUNK_OVERLAP      = 50     # overlap tokens between consecutive chunks
MIN_SEGMENT_TOKENS = 80     # segments below this get merged with their neighbor
EMBEDDING_MODEL    = "text-embedding-3-small"
EMBEDDING_DIM      = 1536
PINECONE_INDEX     = os.getenv("PINECONE_INDEX", "earnings-intelligence")
BATCH_SIZE         = 100    # vectors per Pinecone upsert call


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id:      str
    transcript_id: str
    ticker:        str
    quarter:       str
    fiscal_year:   int
    call_date:     str
    speaker:       str
    role:          str
    firm:          str
    section:       str       # prepared_remarks | qa
    segment_index: int
    chunk_index:   int       # position within the parent segment
    text:          str
    token_count:   int
    embedding:     list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def embed_transcript(silver: dict, dry_run: bool = False) -> list[Chunk]:
    """
    Parameters
    ----------
    silver  : dict  Silver-layer record from normalize.py
    dry_run : bool  Skip OpenAI + Pinecone calls entirely (for local testing)

    Returns
    -------
    list[Chunk]  All chunks with embeddings populated (unless dry_run)
    """
    # Step 1 — merge short segments so tiny utterances aren't standalone chunks
    segments = _merge_short_segments(silver["segments"])

    # Step 2 — split every segment into overlapping token-aware chunks
    chunks = []
    for seg in segments:
        seg_chunks = _chunk_segment(seg, silver)
        chunks.extend(seg_chunks)

    # Step 2b — one chunk per glossary term (section="glossary")
    for i, entry in enumerate(silver.get("glossary", [])):
        text = f"{entry['term']}: {entry['definition']}"
        chunks.append(Chunk(
            chunk_id      = f"{silver['transcript_id']}_glossary_c{i}",
            transcript_id = silver["transcript_id"],
            ticker        = silver["ticker"],
            quarter       = silver["quarter"],
            fiscal_year   = silver["fiscal_year"] or 0,
            call_date     = silver["call_date"] or "",
            speaker       = "editorial",
            role          = "editorial",
            firm          = "",
            section       = "glossary",
            segment_index = -1,
            chunk_index   = i,
            text          = text,
            token_count   = _count_tokens(text),
        ))

    if dry_run:
        _print_dry_run_summary(chunks)
        return chunks

    # Step 3 — embed in batches (OpenAI allows up to 2048 inputs per call)
    chunks = _embed_chunks(chunks)

    # Step 4 — upsert to Pinecone in batches
    _upsert_to_pinecone(chunks)

    return chunks


# ---------------------------------------------------------------------------
# Step 1 — Merge short segments
# ---------------------------------------------------------------------------

def _merge_short_segments(segments: list[dict]) -> list[dict]:
    """
    Segments under MIN_SEGMENT_TOKENS tokens get merged with the next segment
    if they share the same section (prepared_remarks | qa).

    This prevents tiny analyst questions like "Thank you" from becoming
    standalone chunks with no useful retrieval signal.

    Merging preserves the speaker of the FIRST segment in the merged group,
    and appends a [SPEAKER: text] inline label for subsequent speakers so
    the context is never lost.
    """
    merged = []
    i = 0
    while i < len(segments):
        seg = dict(segments[i])  # copy so we don't mutate Silver data
        tok = _count_tokens(seg["text"])

        # Merge forward while under the floor AND same section
        while (
            tok < MIN_SEGMENT_TOKENS
            and i + 1 < len(segments)
            and segments[i + 1]["section"] == seg["section"]
        ):
            i += 1
            next_seg = segments[i]
            next_label = f"[{next_seg['speaker'].upper()}]: {next_seg['text']}"
            seg["text"] = seg["text"].rstrip() + "\n\n" + next_label
            tok = _count_tokens(seg["text"])

        merged.append(seg)
        i += 1

    return merged


# ---------------------------------------------------------------------------
# Step 2 — Chunk a single segment
# ---------------------------------------------------------------------------

def _chunk_segment(segment: dict, silver: dict) -> list[Chunk]:
    """
    Split segment text into CHUNK_SIZE-token windows with CHUNK_OVERLAP
    token overlap. Each chunk inherits full segment metadata.

    Overlap ensures that sentences spanning a chunk boundary appear in
    both chunks, preventing retrieval misses on boundary-crossing content.
    """
    words  = segment["text"].split()
    chunks = []
    start  = 0
    chunk_index = 0

    while start < len(words):
        end        = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])
        token_count = _count_tokens(chunk_text)

        chunk_id = (
            f"{silver['transcript_id']}"
            f"_seg{segment['segment_index']}"
            f"_c{chunk_index}"
        )

        chunks.append(Chunk(
            chunk_id      = chunk_id,
            transcript_id = silver["transcript_id"],
            ticker        = silver["ticker"],
            quarter       = silver["quarter"],
            fiscal_year   = silver["fiscal_year"],
            call_date     = silver["call_date"],
            speaker       = segment["speaker"],
            role          = segment["role"],
            firm          = segment.get("firm", ""),
            section       = segment["section"],
            segment_index = segment["segment_index"],
            chunk_index   = chunk_index,
            text          = chunk_text,
            token_count   = token_count,
        ))

        chunk_index += 1

        # Advance by (CHUNK_SIZE - CHUNK_OVERLAP) to create the sliding window
        step = CHUNK_SIZE - CHUNK_OVERLAP
        start += step

        # If the remaining words would form a chunk smaller than the overlap,
        # fold them into the current chunk rather than creating a tiny tail
        remaining = len(words) - start
        if 0 < remaining < CHUNK_OVERLAP:
            break

    return chunks


# ---------------------------------------------------------------------------
# Step 3 — Embed chunks via OpenAI
# ---------------------------------------------------------------------------

def _embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """
    Calls OpenAI's embeddings API in batches.

    text-embedding-3-small produces 1536-dimensional vectors at ~$0.00002
    per 1K tokens — an 85-segment AAPL transcript costs well under $0.01.

    Includes basic retry logic with exponential backoff for rate limit errors.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    client    = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    texts     = [c.text for c in chunks]
    all_embeddings = []

    # OpenAI recommends batches of ≤2048 inputs
    batch_size = 256
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        success = False
        for attempt in range(4):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                all_embeddings.extend([e.embedding for e in response.data])
                success = True
                break
            except Exception as e:
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                print(f"  OpenAI error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)

        if not success:
            raise RuntimeError(f"Embedding failed after 4 attempts on batch {i}")

    for chunk, embedding in zip(chunks, all_embeddings):
        chunk.embedding = embedding

    return chunks


# ---------------------------------------------------------------------------
# Step 4 — Upsert to Pinecone
# ---------------------------------------------------------------------------

def _upsert_to_pinecone(chunks: list[Chunk]) -> None:
    """
    Upserts all chunks to Pinecone.

    Each vector includes:
    - id     : chunk_id (globally unique)
    - values : embedding floats
    - metadata : all scalar fields for filtered retrieval

    Metadata enables queries like:
        filter={"ticker": "AAPL", "section": "qa", "role": "CEO"}
    which pre-filters the ANN search before semantic similarity is computed.
    This is the key architectural advantage over unfiltered vector search.
    """
    try:
        from pinecone import Pinecone
    except ImportError:
        raise ImportError("Run: pip install pinecone-client")

    pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(PINECONE_INDEX)

    vectors = []
    for chunk in chunks:
        vectors.append({
            "id":     chunk.chunk_id,
            "values": chunk.embedding,
            "metadata": {
                "transcript_id": chunk.transcript_id or "",
                "ticker":        chunk.ticker or "",
                "quarter":       chunk.quarter or "",
                "fiscal_year":   chunk.fiscal_year or 0,
                "call_date":     chunk.call_date or "",
                "speaker":       chunk.speaker or "",
                "role":          chunk.role or "",
                "firm":          chunk.firm or "",
                "section":       chunk.section or "",
                "segment_index": chunk.segment_index,
                "chunk_index":   chunk.chunk_index,
                "text":          chunk.text,       # stored for retrieval display
                "token_count":   chunk.token_count,
            }
        })

    # Upsert in batches — Pinecone recommends ≤100 vectors per call
    total = 0
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)
        total += len(batch)
        print(f"  Upserted {total}/{len(vectors)} vectors...")

    print(f"Done. {total} vectors in index '{PINECONE_INDEX}'.")


# ---------------------------------------------------------------------------
# Token counting — lightweight, no tiktoken dependency required
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """
    Fast approximate token count (4 chars ≈ 1 token for English prose).
    Accurate enough for chunk sizing — we're not billing by token here.

    If you want exact counts, swap this for:
        import tiktoken
        enc = tiktoken.encoding_for_model("text-embedding-3-small")
        return len(enc.encode(text))
    """
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Dry-run summary
# ---------------------------------------------------------------------------

def _print_dry_run_summary(chunks: list[Chunk]) -> None:
    """Print a human-readable summary without making any API calls."""
    from collections import Counter

    total_tokens = sum(c.token_count for c in chunks)
    role_counts  = Counter(c.role for c in chunks)
    section_counts = Counter(c.section for c in chunks)

    print(f"\n{'='*60}")
    print(f"DRY RUN — no API calls made")
    print(f"{'='*60}")
    print(f"  Transcript    : {chunks[0].transcript_id if chunks else 'N/A'}")
    print(f"  Total chunks  : {len(chunks)}")
    print(f"  Total tokens  : {total_tokens:,}  (~${total_tokens/1000*0.00002:.4f} to embed)")
    print(f"  Chunks/section: {dict(section_counts)}")
    print(f"  Chunks/role   : {dict(role_counts)}")
    print(f"\n--- Sample chunks ---")
    for chunk in chunks[:4]:
        print(f"\n  [{chunk.chunk_id}]")
        print(f"  speaker={chunk.speaker} | role={chunk.role} | "
              f"section={chunk.section} | tokens={chunk.token_count}")
        print(f"  text preview: {chunk.text[:120].replace(chr(10), ' ')}...")


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point.

    Expects event to contain the S3 bucket + key of the Silver JSON file.
    Triggered by S3 ObjectCreated event on the silver/ prefix.

    event = {
        "Records": [{
            "s3": {
                "bucket": {"name": "my-bucket"},
                "object": {"key": "silver/AAPL_Q1_2026.json"}
            }
        }]
    }
    """
    import boto3

    s3 = boto3.client("s3")

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key    = record["s3"]["object"]["key"]

        print(f"Processing: s3://{bucket}/{key}")

        obj    = s3.get_object(Bucket=bucket, Key=key)
        silver = json.loads(obj["Body"].read())

        chunks = embed_transcript(silver, dry_run=False)
        print(f"Embedded and upserted {len(chunks)} chunks for {silver['transcript_id']}")

    return {"statusCode": 200, "chunksProcessed": len(chunks)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed a Silver transcript JSON")
    parser.add_argument("silver_file", help="Path to Silver JSON (output of normalize.py)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print chunk summary without calling OpenAI or Pinecone")
    args = parser.parse_args()

    with open(args.silver_file) as f:
        silver = json.load(f)

    chunks = embed_transcript(silver, dry_run=args.dry_run)
    print(f"\nTotal chunks produced: {len(chunks)}")
