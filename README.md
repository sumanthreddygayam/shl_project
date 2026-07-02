# Conversational SHL Assessment Recommender

Production-oriented, stateless RAG + LangGraph FastAPI service for recommending
only catalogued SHL Individual Test Solutions.

## Public API submission links

- **Public API:** [https://shl-recommender-api-3nb0.onrender.com](https://shl-recommender-api-3nb0.onrender.com)
- **Health:** [GET /health](https://shl-recommender-api-3nb0.onrender.com/health)
- **Chat:** [POST /chat](https://shl-recommender-api-3nb0.onrender.com/chat)
- **Interactive API docs:** [Swagger UI](https://shl-recommender-api-3nb0.onrender.com/docs)

`/health` and `/chat` were verified live with HTTP 200 responses.

Submission document: [Approach (two pages maximum)](APPROACH.md)

## What is implemented

- SHL catalog ingestion and normalization (`scraper/catalog.json`)
- SentenceTransformer embeddings and FAISS vectorstore (`vectorstore/`)
- Stateless conversation parsing from `messages[]`
- Safety guardrails for prompt injection and out-of-scope/non-SHL requests
- Clarification, recommendation, refinement, and comparison flows
- Hybrid retrieval: semantic similarity, metadata matching, and keyword/family boosts
- Weighted ranker using the required fit weights
- FastAPI `/health` and `/chat`
- Recruiter-friendly browser UI at `/`
- Graph-based RAG workflow with no server-side memory
- Public trace replay evaluator and unit tests
- Dockerfile and deployment commands

Dataset URLs are intentionally not hardcoded. Configure them via environment
variables.

## Environment

Copy `.env.example` and fill in the dataset locations:

```bash
cp .env.example .env
```

Required:

```text
CATALOG_URL=
CONVERSATION_TRACES_URL=
```

Optional:

```text
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The deterministic catalog-grounded path works without Gemini credentials.

## Install

```bash
python -m pip install -r requirements.txt
```

## Build artifacts

```bash
python -m scraper.scrape_catalog
python -m embeddings.build_embeddings
python -m embeddings.build_embeddings --validate-only
```

The scraper writes normalized Individual Test Solutions only. The embedding
builder writes `vectorstore/index.faiss` and `vectorstore/manifest.json`.

## Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive OpenAPI documentation is available at:

```text
http://localhost:8000/docs
```

On Windows/PowerShell you can also run:

```powershell
.\run_ui.ps1
```

Endpoints:

- `GET /` recruiter chat UI
- `GET /health`
- `POST /chat`

`POST /chat` is stateless: callers send the complete conversation history in
`messages[]` on every request. The response contains the next `reply`, zero to
ten structured `recommendations`, and `end_of_conversation`.

Example:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring Java developer with 4 years experience"}]}'
```

## Required behavior coverage

The implementation is intentionally catalog-grounded:

- Clarification: `app.agent.ClarificationPolicy` asks before recommending when the request is vague.
- Recommendation: `app.retrieval.HybridAssessmentRetriever` and `app.ranking.HybridAssessmentRanker` return 1-10 catalog assessments.
- Refinement: `app.parser.RuleBasedConversationParser` reconstructs requirements from the stateless `messages[]` payload and preserves context for requests like “Actually, add personality tests”.
- Comparison: `app.comparison.CatalogComparator` compares only fields from `scraper/catalog.json`.
- Safety: `app.safety.RuleBasedSafetyGuard` refuses prompt injection, legal/hiring-advice, and non-SHL requests.
- URL grounding: tests assert every returned recommendation URL exists in the scraped catalog.

The graph workflow is in `app/graph.py`:

```text
safety -> parse -> route -> clarify | compare | retrieve -> rank -> respond
```

The browser UI at `/` is only a thin client. It sends the full conversation
history to `POST /chat`, so the server remains stateless.

## Evaluate

```bash
python -m evaluation.replay \
  --traces-dir sample_conversations \
  --report-path evaluation/evaluation_report.json
```

The report includes Recall@10, clarification accuracy, recommendation accuracy,
refinement accuracy, comparison accuracy, hallucination rate, and schema
compliance.

## GitHub hygiene

The `.gitignore` hides local-only and sensitive files, including:

- `ARCHITECTURE.md`
- `sample_conversations/`
- `.env`
- `.env.*`
- `*.log`
- `.agents/`
- `.codex/`

If any of those files were already committed before `.gitignore` existed, remove
them from Git tracking without deleting your local copies:

```bash
git rm --cached ARCHITECTURE.md
git rm --cached -r sample_conversations
git rm --cached .env
git rm --cached uvicorn.err.log uvicorn.out.log
git commit -m "Hide local-only files and secrets"
```

Then push:

```bash
git add .
git commit -m "Update FastAPI service"
git push origin main
```

## Test

```bash
python -m unittest discover -s tests -v
```

## Docker

```bash
docker build -t shl-recommender .
docker run --rm -p 8000:8000 --env-file .env shl-recommender
```

## Deploy the FastAPI service

The repository includes `render.yaml` for a Docker-based Render web service.
It binds Uvicorn to the host-provided `PORT` and uses `GET /health` as the
readiness check.

[Deploy the FastAPI service on Render](https://render.com/deploy?repo=https://github.com/sumanthreddygayam/shl_project)

After deployment, the public service exposes:

```text
GET  https://shl-recommender-api-3nb0.onrender.com/health
POST https://shl-recommender-api-3nb0.onrender.com/chat
GET  https://shl-recommender-api-3nb0.onrender.com/docs
```

For a fresh environment, run the catalog and embedding build commands before
serving, or bake generated artifacts into the image.
