"""
tests/test_query.py

Comprehensive tests for the FastAPI endpoints:
    GET  /health
    POST /query

Run:
    pytest tests/test_query.py -v

Mocking strategy:
    - query.query.query_transcripts  — patched at the module attribute; the
      endpoint does a lazy ``from query.query import query_transcripts`` on
      every request, so patching the module attribute is authoritative.
    - api.main._log_query            — suppressed by default; individual
      logging tests re-patch it inside the test body to inspect call args.
    - asyncio.wait_for               — patched to raise TimeoutError for the
      504 test; safe here because all other tests complete synchronously.
"""

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Env vars must be set before api.main is imported so the lifespan check
# sees all three keys and sets app.state.ready = True.
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY",   "test-openai-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("PINECONE_INDEX",   "test-index")

from api.main import app, _rate_limit_tracker          # noqa: E402
from query.query import QueryFilter, QueryResponse, SourceChunk  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — build realistic fixture objects without touching any real service
# ---------------------------------------------------------------------------

def _make_source(
    chunk_id: str  = "AAPL_Q1_2026_seg3_c0",
    speaker:  str  = "Timothy D. Cook",
    role:     str  = "CEO",
    firm:     str  = "Apple Inc.",
    section:  str  = "prepared_remarks",
    quarter:  str  = "Q1 2026",
    ticker:   str  = "AAPL",
    text:     str  = "Revenue grew 16% year over year.",
    score:    float = 0.9123,
) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id, speaker=speaker, role=role, firm=firm,
        section=section, quarter=quarter, ticker=ticker, text=text, score=score,
    )


def _make_response(
    question:    str   = "What did Tim Cook say about revenue?",
    answer:      str   = "Tim Cook noted strong revenue growth [SOURCE 1].",
    sources:     list  = None,
    found:       bool  = True,
    confidence:  str   = "high",
    model:       str   = "gpt-4o",
    chunks_used: int   = 1,
) -> QueryResponse:
    return QueryResponse(
        question=question,
        answer=answer,
        sources=sources if sources is not None else [_make_source()],
        found=found,
        confidence=confidence,
        model=model,
        chunks_used=chunks_used,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client():
    """AsyncClient wired directly to the ASGI app — no real network I/O."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """
    Clear the in-memory rate-limit tracker before and after every test so
    that request counts don't bleed between tests.
    """
    _rate_limit_tracker.clear()
    yield
    _rate_limit_tracker.clear()


# ===========================================================================
# GET /health
# ===========================================================================

class TestHealth:

    async def test_returns_200(self, client):
        response = await client.get("/health")
        assert response.status_code == 200

    async def test_body_is_status_ok(self, client):
        response = await client.get("/health")
        assert response.json() == {"status": "ok"}

    async def test_returns_200_with_no_headers(self, client):
        """Liveness check must not require any auth or content-type header."""
        response = await client.get("/health", headers={})
        assert response.status_code == 200

    async def test_exempt_from_rate_limiter(self, client):
        """
        The rate limiter skips /health (see skip_paths in middleware).
        31 consecutive requests must all succeed.
        """
        for _ in range(31):
            response = await client.get("/health")
        assert response.status_code == 200

    async def test_x_response_time_header_present(self, client):
        """Latency middleware must attach X-Response-Time-Ms to every response."""
        response = await client.get("/health")
        assert "x-response-time-ms" in response.headers


# ===========================================================================
# POST /query — happy path
# ===========================================================================

class TestQueryHappyPath:

    async def test_valid_question_returns_200(self, client):
        with patch("query.query.query_transcripts", return_value=_make_response()), \
             patch("api.main._log_query"):
            response = await client.post("/query", json={"question": "What is Apple's gross margin?"})
        assert response.status_code == 200

    async def test_response_contains_all_required_fields(self, client):
        with patch("query.query.query_transcripts", return_value=_make_response()), \
             patch("api.main._log_query"):
            response = await client.post("/query", json={"question": "Revenue?"})
        data = response.json()
        for key in ("question", "answer", "found", "confidence", "model", "chunks_used", "sources"):
            assert key in data, f"Missing top-level key: {key}"

    async def test_question_echoed_verbatim(self, client):
        q = "What did Tim Cook say about the Services segment?"
        with patch("query.query.query_transcripts", return_value=_make_response(question=q)), \
             patch("api.main._log_query"):
            response = await client.post("/query", json={"question": q})
        assert response.json()["question"] == q

    async def test_answer_returned(self, client):
        with patch("query.query.query_transcripts",
                   return_value=_make_response(answer="Services revenue hit $26.3B [SOURCE 1].")), \
             patch("api.main._log_query"):
            response = await client.post("/query", json={"question": "Services?"})
        assert response.json()["answer"] == "Services revenue hit $26.3B [SOURCE 1]."

    async def test_found_true_in_response(self, client):
        with patch("query.query.query_transcripts", return_value=_make_response(found=True)), \
             patch("api.main._log_query"):
            response = await client.post("/query", json={"question": "Revenue?"})
        assert response.json()["found"] is True

    async def test_confidence_high_in_response(self, client):
        with patch("query.query.query_transcripts", return_value=_make_response(confidence="high")), \
             patch("api.main._log_query"):
            response = await client.post("/query", json={"question": "Revenue?"})
        assert response.json()["confidence"] == "high"

    async def test_model_field_populated(self, client):
        with patch("query.query.query_transcripts", return_value=_make_response(model="gpt-4o")), \
             patch("api.main._log_query"):
            response = await client.post("/query", json={"question": "Revenue?"})
        assert response.json()["model"] == "gpt-4o"

    async def test_chunks_used_matches_sources_length(self, client):
        sources = [_make_source(chunk_id=f"AAPL_Q1_2026_seg{i}_c0") for i in range(3)]
        with patch("query.query.query_transcripts",
                   return_value=_make_response(sources=sources, chunks_used=3)), \
             patch("api.main._log_query"):
            response = await client.post("/query", json={"question": "Revenue?"})
        data = response.json()
        assert data["chunks_used"] == 3
        assert len(data["sources"]) == 3

    async def test_source_fields_serialized_correctly(self, client):
        source = _make_source(
            chunk_id="AAPL_Q1_2026_seg3_c0",
            speaker="Timothy D. Cook",
            role="CEO",
            firm="Apple Inc.",
            section="prepared_remarks",
            quarter="Q1 2026",
            ticker="AAPL",
            text="Revenue grew 16% year over year.",
            score=0.9123,
        )
        with patch("query.query.query_transcripts",
                   return_value=_make_response(sources=[source])), \
             patch("api.main._log_query"):
            response = await client.post("/query", json={"question": "Revenue?"})
        s = response.json()["sources"][0]
        assert s["chunk_id"] == "AAPL_Q1_2026_seg3_c0"
        assert s["speaker"]  == "Timothy D. Cook"
        assert s["role"]     == "CEO"
        assert s["firm"]     == "Apple Inc."
        assert s["section"]  == "prepared_remarks"
        assert s["quarter"]  == "Q1 2026"
        assert s["ticker"]   == "AAPL"
        assert s["text"]     == "Revenue grew 16% year over year."
        assert s["score"]    == 0.9123

    async def test_empty_sources_list_serializes_cleanly(self, client):
        with patch("query.query.query_transcripts",
                   return_value=_make_response(sources=[], chunks_used=0, found=False, confidence="low")), \
             patch("api.main._log_query"):
            response = await client.post("/query", json={"question": "Revenue?"})
        data = response.json()
        assert data["sources"] == []
        assert data["chunks_used"] == 0


# ===========================================================================
# POST /query — filters passed through to query_transcripts
# ===========================================================================

class TestQueryFilters:

    async def test_ticker_filter_forwarded(self, client):
        with patch("query.query.query_transcripts") as mock_qt, \
             patch("api.main._log_query"):
            mock_qt.return_value = _make_response()
            await client.post("/query", json={"question": "Revenue?", "ticker": "AAPL"})
        filters: QueryFilter = mock_qt.call_args[0][1]
        assert filters.ticker == "AAPL"

    async def test_quarter_filter_forwarded(self, client):
        with patch("query.query.query_transcripts") as mock_qt, \
             patch("api.main._log_query"):
            mock_qt.return_value = _make_response()
            await client.post("/query", json={"question": "Revenue?", "quarter": "Q1 2026"})
        filters: QueryFilter = mock_qt.call_args[0][1]
        assert filters.quarter == "Q1 2026"

    async def test_role_filter_forwarded(self, client):
        with patch("query.query.query_transcripts") as mock_qt, \
             patch("api.main._log_query"):
            mock_qt.return_value = _make_response()
            await client.post("/query", json={"question": "Revenue?", "role": "CEO"})
        filters: QueryFilter = mock_qt.call_args[0][1]
        assert filters.role == "CEO"

    async def test_section_filter_forwarded(self, client):
        with patch("query.query.query_transcripts") as mock_qt, \
             patch("api.main._log_query"):
            mock_qt.return_value = _make_response()
            await client.post("/query", json={"question": "Revenue?", "section": "qa"})
        filters: QueryFilter = mock_qt.call_args[0][1]
        assert filters.section == "qa"

    async def test_all_four_filters_forwarded_together(self, client):
        with patch("query.query.query_transcripts") as mock_qt, \
             patch("api.main._log_query"):
            mock_qt.return_value = _make_response()
            await client.post("/query", json={
                "question": "Revenue?",
                "ticker":   "MSFT",
                "quarter":  "Q2 2026",
                "role":     "CFO",
                "section":  "prepared_remarks",
            })
        filters: QueryFilter = mock_qt.call_args[0][1]
        assert filters.ticker  == "MSFT"
        assert filters.quarter == "Q2 2026"
        assert filters.role    == "CFO"
        assert filters.section == "prepared_remarks"

    async def test_omitted_filters_are_none(self, client):
        """Filters not present in the request body must arrive as None, not empty string."""
        with patch("query.query.query_transcripts") as mock_qt, \
             patch("api.main._log_query"):
            mock_qt.return_value = _make_response()
            await client.post("/query", json={"question": "Revenue?"})
        filters: QueryFilter = mock_qt.call_args[0][1]
        assert filters.ticker  is None
        assert filters.quarter is None
        assert filters.role    is None
        assert filters.section is None

    async def test_ticker_uppercased_in_pinecone_filter(self, client):
        """
        QueryFilter.ticker arrives lowercase from the request; the Pinecone
        filter builder in query.py uppercases it before the index query.
        The filter object passed to query_transcripts carries the original
        value — uppercasing happens inside query.py.  This test verifies
        the API layer does NOT silently mutate the caller's casing.
        """
        with patch("query.query.query_transcripts") as mock_qt, \
             patch("api.main._log_query"):
            mock_qt.return_value = _make_response()
            await client.post("/query", json={"question": "Revenue?", "ticker": "aapl"})
        filters: QueryFilter = mock_qt.call_args[0][1]
        assert filters.ticker == "aapl"


# ===========================================================================
# POST /query — not-found / low-confidence results are still HTTP 200
# ===========================================================================

class TestQueryNotFound:

    async def test_found_false_is_200_not_404(self, client):
        """found=False is a business-logic result, not an HTTP error."""
        not_found = _make_response(
            found=False, confidence="low",
            answer="No relevant content found for this question.",
            sources=[], chunks_used=0,
        )
        with patch("query.query.query_transcripts", return_value=not_found), \
             patch("api.main._log_query"):
            response = await client.post("/query", json={"question": "Unknowable question?"})
        assert response.status_code == 200
        data = response.json()
        assert data["found"]      is False
        assert data["confidence"] == "low"
        assert data["sources"]    == []

    async def test_medium_confidence_returned_correctly(self, client):
        with patch("query.query.query_transcripts",
                   return_value=_make_response(confidence="medium")), \
             patch("api.main._log_query"):
            response = await client.post("/query", json={"question": "Revenue?"})
        assert response.json()["confidence"] == "medium"


# ===========================================================================
# POST /query — request validation (422)
# ===========================================================================

class TestQueryValidation:

    async def test_empty_string_question_returns_422(self, client):
        response = await client.post("/query", json={"question": ""})
        assert response.status_code == 422

    async def test_whitespace_only_question_returns_422(self, client):
        response = await client.post("/query", json={"question": "   "})
        assert response.status_code == 422

    async def test_missing_question_field_returns_422(self, client):
        """Pydantic must reject a body that omits the required 'question' field."""
        response = await client.post("/query", json={})
        assert response.status_code == 422

    async def test_non_string_question_returns_422(self, client):
        response = await client.post("/query", json={"question": 42})
        assert response.status_code == 422

    async def test_null_question_returns_422(self, client):
        response = await client.post("/query", json={"question": None})
        assert response.status_code == 422

    async def test_422_contains_detail_key(self, client):
        response = await client.post("/query", json={"question": ""})
        assert "detail" in response.json()


# ===========================================================================
# POST /query — pipeline and timeout errors
# ===========================================================================

class TestQueryErrors:

    async def test_pipeline_runtime_error_returns_500(self, client):
        with patch("query.query.query_transcripts",
                   side_effect=RuntimeError("Pinecone connection refused")):
            response = await client.post("/query", json={"question": "Revenue?"})
        assert response.status_code == 500

    async def test_generic_exception_returns_500(self, client):
        with patch("query.query.query_transcripts",
                   side_effect=Exception("OpenAI API quota exceeded")):
            response = await client.post("/query", json={"question": "Revenue?"})
        assert response.status_code == 500

    async def test_500_response_has_detail(self, client):
        with patch("query.query.query_transcripts", side_effect=RuntimeError("boom")):
            response = await client.post("/query", json={"question": "Revenue?"})
        assert "detail" in response.json()

    async def test_timeout_returns_504(self, client):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            response = await client.post("/query", json={"question": "Revenue?"})
        assert response.status_code == 504

    async def test_504_detail_mentions_timeout_duration(self, client):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            response = await client.post("/query", json={"question": "Revenue?"})
        detail = response.json()["detail"]
        assert "30" in detail  # _QUERY_TIMEOUT_SECONDS = 30.0

    async def test_log_query_not_called_on_pipeline_error(self, client):
        """_log_query must only be called after a successful pipeline run."""
        with patch("query.query.query_transcripts", side_effect=RuntimeError("boom")), \
             patch("api.main._log_query") as mock_log:
            await client.post("/query", json={"question": "Revenue?"})
        mock_log.assert_not_called()

    async def test_log_query_not_called_on_timeout(self, client):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()), \
             patch("api.main._log_query") as mock_log:
            await client.post("/query", json={"question": "Revenue?"})
        mock_log.assert_not_called()


# ===========================================================================
# POST /query — query logging side-effects
# ===========================================================================

class TestQueryLogging:

    async def test_log_query_called_once_on_success(self, client):
        with patch("query.query.query_transcripts", return_value=_make_response()), \
             patch("api.main._log_query") as mock_log:
            await client.post("/query", json={"question": "Revenue?"})
        mock_log.assert_called_once()

    async def test_log_entry_contains_question(self, client):
        with patch("query.query.query_transcripts",
                   return_value=_make_response(question="What is gross margin?")), \
             patch("api.main._log_query") as mock_log:
            await client.post("/query", json={"question": "What is gross margin?"})
        entry = mock_log.call_args[0][0]
        assert entry["question"] == "What is gross margin?"

    async def test_log_entry_unset_filters_use_none_string(self, client):
        """
        The log entry uses the string "none" (not Python None) for unset filters
        — this matches the DynamoDB-safe convention established in api/main.py.
        """
        with patch("query.query.query_transcripts", return_value=_make_response()), \
             patch("api.main._log_query") as mock_log:
            await client.post("/query", json={"question": "Revenue?"})
        entry = mock_log.call_args[0][0]
        assert entry["ticker_filter"]  == "none"
        assert entry["quarter_filter"] == "none"
        assert entry["role_filter"]    == "none"
        assert entry["section_filter"] == "none"

    async def test_log_entry_set_filters_are_recorded(self, client):
        with patch("query.query.query_transcripts", return_value=_make_response()), \
             patch("api.main._log_query") as mock_log:
            await client.post("/query", json={
                "question": "Revenue?",
                "ticker":   "AAPL",
                "quarter":  "Q1 2026",
                "role":     "CEO",
                "section":  "prepared_remarks",
            })
        entry = mock_log.call_args[0][0]
        assert entry["ticker_filter"]  == "AAPL"
        assert entry["quarter_filter"] == "Q1 2026"
        assert entry["role_filter"]    == "CEO"
        assert entry["section_filter"] == "prepared_remarks"

    async def test_log_entry_contains_confidence_and_found(self, client):
        with patch("query.query.query_transcripts",
                   return_value=_make_response(found=True, confidence="high")), \
             patch("api.main._log_query") as mock_log:
            await client.post("/query", json={"question": "Revenue?"})
        entry = mock_log.call_args[0][0]
        assert entry["found"]      is True
        assert entry["confidence"] == "high"

    async def test_log_not_called_on_empty_question(self, client):
        with patch("api.main._log_query") as mock_log:
            await client.post("/query", json={"question": ""})
        mock_log.assert_not_called()


# ===========================================================================
# Rate limiting
# ===========================================================================

class TestRateLimiting:

    async def test_exactly_at_limit_succeeds(self, client):
        """The 30th request within the window must still get a 200."""
        with patch("query.query.query_transcripts", return_value=_make_response()), \
             patch("api.main._log_query"):
            for _ in range(29):
                await client.post("/query", json={"question": "Revenue?"})
            response = await client.post("/query", json={"question": "Revenue?"})
        assert response.status_code == 200

    async def test_one_over_limit_returns_429(self, client):
        """The 31st request within the window must be rejected."""
        with patch("query.query.query_transcripts", return_value=_make_response()), \
             patch("api.main._log_query"):
            for _ in range(30):
                await client.post("/query", json={"question": "Revenue?"})
            response = await client.post("/query", json={"question": "Revenue?"})
        assert response.status_code == 429

    async def test_429_has_retry_after_header(self, client):
        with patch("query.query.query_transcripts", return_value=_make_response()), \
             patch("api.main._log_query"):
            for _ in range(30):
                await client.post("/query", json={"question": "Revenue?"})
            response = await client.post("/query", json={"question": "Revenue?"})
        assert "retry-after" in response.headers

    async def test_429_has_detail_message(self, client):
        with patch("query.query.query_transcripts", return_value=_make_response()), \
             patch("api.main._log_query"):
            for _ in range(30):
                await client.post("/query", json={"question": "Revenue?"})
            response = await client.post("/query", json={"question": "Revenue?"})
        assert "detail" in response.json()

    async def test_health_not_counted_toward_rate_limit(self, client):
        """
        /health is in the rate-limiter's skip_paths set.  Even after 30
        health requests the next /query (count=1) must succeed.
        """
        for _ in range(30):
            await client.get("/health")
        with patch("query.query.query_transcripts", return_value=_make_response()), \
             patch("api.main._log_query"):
            response = await client.post("/query", json={"question": "Revenue?"})
        assert response.status_code == 200


# ===========================================================================
# GET /ready
# ===========================================================================

class TestReady:

    async def test_ready_returns_200_when_env_vars_set(self, client):
        """All three required env vars are set in the module-level os.environ.setdefault calls."""
        response = await client.get("/ready")
        assert response.status_code == 200

    async def test_ready_body_has_status_ready(self, client):
        response = await client.get("/ready")
        assert response.json()["status"] == "ready"

    async def test_ready_body_lists_env_vars(self, client):
        response = await client.get("/ready")
        data = response.json()
        assert "env_vars_set" in data
        assert isinstance(data["env_vars_set"], list)

    async def test_ready_lists_all_three_env_vars(self, client):
        response = await client.get("/ready")
        env_set = response.json()["env_vars_set"]
        for var in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"):
            assert var in env_set

    async def test_ready_returns_503_when_env_var_missing(self, client):
        import os
        original = os.environ.pop("OPENAI_API_KEY", None)
        try:
            response = await client.get("/ready")
            assert response.status_code == 503
        finally:
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original

    async def test_503_detail_names_missing_var(self, client):
        import os
        original = os.environ.pop("OPENAI_API_KEY", None)
        try:
            response = await client.get("/ready")
            assert "OPENAI_API_KEY" in response.json()["detail"]
        finally:
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original


# ===========================================================================
# GET /metrics
# ===========================================================================

_SAMPLE_LOG_ENTRIES = [
    {
        "query_id":       "abc-1",
        "timestamp":      "2026-03-20T10:00:00+00:00",
        "question":       "What is gross margin?",
        "ticker_filter":  "AAPL",
        "quarter_filter": "Q1 2026",
        "role_filter":    "none",
        "section_filter": "none",
        "found":          True,
        "confidence":     "high",
        "chunks_used":    5,
        "model":          "gpt-4o",
        "elapsed_seconds": 2.1,
    },
    {
        "query_id":       "abc-2",
        "timestamp":      "2026-03-21T11:00:00+00:00",
        "question":       "What did the CFO say?",
        "ticker_filter":  "MSFT",
        "quarter_filter": "none",
        "role_filter":    "CFO",
        "section_filter": "none",
        "found":          False,
        "confidence":     "low",
        "chunks_used":    0,
        "model":          "gpt-4o",
        "elapsed_seconds": 1.5,
    },
]


class TestMetrics:

    async def test_metrics_returns_200(self, client):
        with patch("api.main._get_log_entries", return_value=_SAMPLE_LOG_ENTRIES):
            response = await client.get("/metrics")
        assert response.status_code == 200

    async def test_metrics_contains_all_required_fields(self, client):
        with patch("api.main._get_log_entries", return_value=_SAMPLE_LOG_ENTRIES):
            response = await client.get("/metrics")
        data = response.json()
        for key in (
            "total_queries", "queries_by_confidence", "queries_by_ticker",
            "average_chunks_used", "found_rate", "average_latency_seconds",
            "last_10_queries",
        ):
            assert key in data, f"Missing key: {key}"

    async def test_total_queries_matches_log_count(self, client):
        with patch("api.main._get_log_entries", return_value=_SAMPLE_LOG_ENTRIES):
            response = await client.get("/metrics")
        assert response.json()["total_queries"] == 2

    async def test_queries_by_confidence_counts(self, client):
        with patch("api.main._get_log_entries", return_value=_SAMPLE_LOG_ENTRIES):
            response = await client.get("/metrics")
        conf = response.json()["queries_by_confidence"]
        assert conf["high"] == 1
        assert conf["low"]  == 1
        assert conf["medium"] == 0

    async def test_queries_by_ticker_counts(self, client):
        with patch("api.main._get_log_entries", return_value=_SAMPLE_LOG_ENTRIES):
            response = await client.get("/metrics")
        tickers = response.json()["queries_by_ticker"]
        assert tickers.get("AAPL") == 1
        assert tickers.get("MSFT") == 1

    async def test_found_rate_calculation(self, client):
        """1 of 2 queries found → 50.0%"""
        with patch("api.main._get_log_entries", return_value=_SAMPLE_LOG_ENTRIES):
            response = await client.get("/metrics")
        assert response.json()["found_rate"] == 50.0

    async def test_average_chunks_used(self, client):
        """(5 + 0) / 2 = 2.5"""
        with patch("api.main._get_log_entries", return_value=_SAMPLE_LOG_ENTRIES):
            response = await client.get("/metrics")
        assert response.json()["average_chunks_used"] == 2.5

    async def test_average_latency(self, client):
        """(2.1 + 1.5) / 2 = 1.8"""
        with patch("api.main._get_log_entries", return_value=_SAMPLE_LOG_ENTRIES):
            response = await client.get("/metrics")
        assert response.json()["average_latency_seconds"] == 1.8

    async def test_last_10_queries_bounded(self, client):
        """last_10_queries must contain at most 10 entries."""
        many_entries = _SAMPLE_LOG_ENTRIES * 10  # 20 entries
        with patch("api.main._get_log_entries", return_value=many_entries):
            response = await client.get("/metrics")
        assert len(response.json()["last_10_queries"]) <= 10

    async def test_empty_log_returns_zero_metrics(self, client):
        with patch("api.main._get_log_entries", return_value=[]):
            response = await client.get("/metrics")
        data = response.json()
        assert data["total_queries"]   == 0
        assert data["found_rate"]      == 0.0
        assert data["last_10_queries"] == []
