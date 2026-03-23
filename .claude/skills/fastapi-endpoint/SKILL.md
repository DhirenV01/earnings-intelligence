---
name: fastapi-endpoint
description: Production FastAPI endpoint with async patterns, validation, error handling, and tests
---

# FastAPI Endpoint Protocol

## Before Writing Code
1. Read `api/` directory to understand existing router structure and patterns
2. Check `api/dependencies.py` or equivalent for shared dependencies
3. Identify which pipeline/query modules the endpoint needs to call

## Endpoint Structure
1. Define Pydantic request model with field validators
2. Define Pydantic response model (never return raw dicts)
3. Create async endpoint function with typed parameters
4. Add to router with explicit status codes and response_model
5. Write pytest async test in `tests/`

## Patterns to Follow
- Use lifespan context manager for startup/shutdown (DB pools, client sessions)
- All external calls (Pinecone, OpenAI, S3) must be async with timeout
- Rate limiting via SlowAPI or custom middleware
- Health endpoint: GET /health returns {"status": "ok", "version": str}
- Readiness endpoint: GET /ready checks downstream dependencies
- Structured logging with request_id correlation

## Error Handling
- HTTPException with specific status codes, never bare 500s
- Async timeout: wrap external calls in asyncio.wait_for(coro, timeout=30)
- Retry with exponential backoff for transient failures (S3, Pinecone)
- Return problem detail format: {"detail": str, "type": str}

## Anti-Patterns (NEVER do these)
- No synchronous blocking calls in async endpoints
- No global mutable state outside lifespan context
- No bare except clauses
- No returning raw dicts, always use response_model
- No hardcoded API keys or config values

## Test Requirements
- Use pytest-asyncio with httpx.AsyncClient
- Test happy path, validation errors, and downstream failures
- Mock external services (Pinecone, OpenAI) with pytest-mock
- Assert status codes AND response body structure
