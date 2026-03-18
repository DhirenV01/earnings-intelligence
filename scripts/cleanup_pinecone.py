"""
cleanup_pinecone.py

Delete all Pinecone vectors matching a given transcript_id.

Usage:
    python scripts/cleanup_pinecone.py MSFT_UNKNOWN
    python scripts/cleanup_pinecone.py MSFT_UNKNOWN --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

# Repo root on sys.path so .env is found relative to project
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env", override=False)
    except ImportError:
        pass

    missing = [k for k in ("PINECONE_API_KEY",) if not os.getenv(k)]
    if missing:
        print(f"ERROR: missing env vars: {', '.join(missing)}")
        sys.exit(1)


def delete_by_transcript_id(transcript_id: str, dry_run: bool = False) -> int:
    from pinecone import Pinecone

    index_name = os.getenv("PINECONE_INDEX", "earnings-intelligence")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)

    # Pinecone serverless doesn't support delete-by-filter directly.
    # Strategy: query with a zero vector + filter to collect all matching IDs,
    # paginating in batches of 1000 until exhausted, then delete by IDs.
    dummy = [0.0] * 1536
    pinecone_filter = {"transcript_id": {"$eq": transcript_id}}

    all_ids = []
    while True:
        res = index.query(
            vector=dummy,
            filter=pinecone_filter,
            top_k=1000,
            include_metadata=False,
        )
        batch_ids = [m["id"] for m in res.get("matches", [])]
        if not batch_ids:
            break
        all_ids.extend(batch_ids)
        # If fewer than 1000 returned, we've exhausted the matches
        if len(batch_ids) < 1000:
            break

    # Deduplicate in case of overlap across pages
    all_ids = list(dict.fromkeys(all_ids))

    if not all_ids:
        print(f"No vectors found for transcript_id='{transcript_id}'.")
        return 0

    print(f"Found {len(all_ids)} vector(s) for transcript_id='{transcript_id}'.")

    if dry_run:
        print("Dry run — no deletions performed.")
        for vid in all_ids:
            print(f"  would delete: {vid}")
        return len(all_ids)

    # Delete in batches of 1000 (Pinecone limit per delete call)
    batch_size = 1000
    deleted = 0
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i : i + batch_size]
        index.delete(ids=batch)
        deleted += len(batch)
        print(f"Deleted {deleted}/{len(all_ids)} vectors...")

    print(f"Done. {deleted} vector(s) removed from index '{index_name}'.")
    return deleted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete Pinecone vectors by transcript_id")
    parser.add_argument("transcript_id", help="transcript_id to delete, e.g. MSFT_UNKNOWN")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    args = parser.parse_args()

    load_env()
    delete_by_transcript_id(args.transcript_id, dry_run=args.dry_run)
