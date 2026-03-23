"""
tests/test_ingest.py

Tests for POST /ingest — multipart file upload endpoint.

Run:
    pytest tests/test_ingest.py -v

Mocking strategy:
    - ingestion.ingest_upload.ingest_transcript — patched at the module
      attribute; main.py does a lazy import inside the endpoint body, so
      patching the module attribute is authoritative.
"""

import io
import os
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Env vars must be set before api.main is imported.
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY",   "test-openai-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("PINECONE_INDEX",   "test-index")

from api.main import app, _rate_limit_tracker  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INGEST_RESULT = {
    "transcript_id":    "AAPL_Q1_2026",
    "chunks_upserted":  51,
    "tokens_estimated": 24576,
    "elapsed_seconds":  3.14,
    "skipped":          False,
}

_INGEST_SKIPPED = {
    "transcript_id":    "AAPL_Q1_2026",
    "chunks_upserted":  0,
    "tokens_estimated": 0,
    "elapsed_seconds":  0.05,
    "skipped":          True,
}

_DUMMY_TXT = b"Full Conference Call Transcript\nTim Cook: Revenue grew."


def _txt_file(content: bytes = _DUMMY_TXT, filename: str = "aapl.txt"):
    return ("file", (filename, io.BytesIO(content), "text/plain"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    _rate_limit_tracker.clear()
    yield
    _rate_limit_tracker.clear()


# ===========================================================================
# POST /ingest — happy path
# ===========================================================================

class TestIngestHappyPath:

    async def test_valid_upload_returns_200(self, client):
        with patch("ingestion.ingest_upload.ingest_transcript", return_value=_INGEST_RESULT):
            response = await client.post(
                "/ingest",
                files=[_txt_file()],
                data={"ticker": "AAPL"},
            )
        assert response.status_code == 200

    async def test_response_contains_all_required_fields(self, client):
        with patch("ingestion.ingest_upload.ingest_transcript", return_value=_INGEST_RESULT):
            response = await client.post(
                "/ingest",
                files=[_txt_file()],
                data={"ticker": "AAPL"},
            )
        data = response.json()
        for key in ("transcript_id", "chunks_upserted", "tokens_estimated", "elapsed_seconds", "skipped"):
            assert key in data, f"Missing key: {key}"

    async def test_transcript_id_returned(self, client):
        with patch("ingestion.ingest_upload.ingest_transcript", return_value=_INGEST_RESULT):
            response = await client.post(
                "/ingest",
                files=[_txt_file()],
                data={"ticker": "AAPL"},
            )
        assert response.json()["transcript_id"] == "AAPL_Q1_2026"

    async def test_chunks_upserted_returned(self, client):
        with patch("ingestion.ingest_upload.ingest_transcript", return_value=_INGEST_RESULT):
            response = await client.post(
                "/ingest",
                files=[_txt_file()],
                data={"ticker": "AAPL"},
            )
        assert response.json()["chunks_upserted"] == 51

    async def test_skipped_false_on_fresh_ingest(self, client):
        with patch("ingestion.ingest_upload.ingest_transcript", return_value=_INGEST_RESULT):
            response = await client.post(
                "/ingest",
                files=[_txt_file()],
                data={"ticker": "AAPL"},
            )
        assert response.json()["skipped"] is False

    async def test_skipped_true_on_duplicate(self, client):
        with patch("ingestion.ingest_upload.ingest_transcript", return_value=_INGEST_SKIPPED):
            response = await client.post(
                "/ingest",
                files=[_txt_file()],
                data={"ticker": "AAPL"},
            )
        data = response.json()
        assert data["skipped"] is True
        assert data["chunks_upserted"] == 0

    async def test_force_flag_forwarded(self, client):
        with patch("ingestion.ingest_upload.ingest_transcript", return_value=_INGEST_RESULT) as mock_ingest:
            await client.post(
                "/ingest",
                files=[_txt_file()],
                data={"ticker": "AAPL", "force": "true"},
            )
        _, kwargs = mock_ingest.call_args
        assert kwargs.get("force") is True

    async def test_raw_text_passed_to_pipeline(self, client):
        """The endpoint decodes the file bytes and passes raw_text to ingest_transcript."""
        with patch("ingestion.ingest_upload.ingest_transcript", return_value=_INGEST_RESULT) as mock_ingest:
            await client.post(
                "/ingest",
                files=[_txt_file(content=b"Full Conference Call Transcript\nTest content.")],
                data={"ticker": "AAPL"},
            )
        _, kwargs = mock_ingest.call_args
        assert "raw_text" in kwargs
        assert "Full Conference Call Transcript" in kwargs["raw_text"]


# ===========================================================================
# POST /ingest — validation errors (422)
# ===========================================================================

class TestIngestValidation:

    async def test_non_txt_file_returns_422(self, client):
        response = await client.post(
            "/ingest",
            files=[_txt_file(filename="transcript.pdf")],
            data={"ticker": "AAPL"},
        )
        assert response.status_code == 422

    async def test_422_contains_detail(self, client):
        response = await client.post(
            "/ingest",
            files=[_txt_file(filename="transcript.csv")],
            data={"ticker": "AAPL"},
        )
        assert "detail" in response.json()

    async def test_missing_ticker_returns_422(self, client):
        """ticker is a required Form field; omitting it must fail validation."""
        response = await client.post(
            "/ingest",
            files=[_txt_file()],
        )
        assert response.status_code == 422

    async def test_non_utf8_file_returns_422(self, client):
        bad_bytes = b"\xff\xfe invalid utf-8"
        response = await client.post(
            "/ingest",
            files=[_txt_file(content=bad_bytes)],
            data={"ticker": "AAPL"},
        )
        assert response.status_code == 422

    async def test_pipeline_value_error_returns_422(self, client):
        with patch("ingestion.ingest_upload.ingest_transcript",
                   side_effect=ValueError("ticker must be uppercase")):
            response = await client.post(
                "/ingest",
                files=[_txt_file()],
                data={"ticker": "aapl"},
            )
        assert response.status_code == 422


# ===========================================================================
# POST /ingest — pipeline errors (500)
# ===========================================================================

class TestIngestErrors:

    async def test_pipeline_runtime_error_returns_500(self, client):
        with patch("ingestion.ingest_upload.ingest_transcript",
                   side_effect=RuntimeError("Pinecone upsert failed")):
            response = await client.post(
                "/ingest",
                files=[_txt_file()],
                data={"ticker": "AAPL"},
            )
        assert response.status_code == 500

    async def test_500_contains_detail(self, client):
        with patch("ingestion.ingest_upload.ingest_transcript",
                   side_effect=Exception("OpenAI quota exceeded")):
            response = await client.post(
                "/ingest",
                files=[_txt_file()],
                data={"ticker": "AAPL"},
            )
        assert "detail" in response.json()
