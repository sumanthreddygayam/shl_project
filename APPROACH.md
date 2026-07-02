# Approach: Conversational SHL Assessment Recommender

## Design choices

The system is a stateless FastAPI service built around a typed LangGraph workflow:
`safety → parse → route → clarify/compare/retrieve → rank → respond`. `POST /chat`
receives the complete `messages[]` history on every request; no server-side chat
memory is required. This makes requests reproducible, horizontally scalable, and
easy to replay during evaluation. `GET /health` provides a lightweight readiness
check, while Pydantic models enforce strict request and response schemas.

I separated orchestration from business logic. LangGraph controls transitions,
while deterministic modules handle requirement extraction, safety, retrieval,
ranking, comparison, and response construction. The agent asks one clarification
at a time when role or seniority information is insufficient. Safety checks run
before retrieval and reject prompt injection, non-SHL requests, and hiring/legal
advice. Every returned product is resolved from the normalized SHL catalog, which
prevents invented names or URLs.

## Retrieval and ranking

The offline pipeline filters the source data to SHL Individual Test Solutions,
normalizes catalog fields, and writes `scraper/catalog.json`. Assessment name,
description, categories, skills, and job levels form the retrieval corpus. The
full setup combines three signals:

1. semantic similarity using `all-MiniLM-L6-v2` embeddings and a normalized FAISS
   index;
2. lexical overlap for exact technologies, role terms, and product names; and
3. metadata matches for seniority, technical skills, personality, leadership,
   communication, stakeholder, remote, and adaptive requirements.

Candidate sets are merged before a deterministic top-10 ranker. Fit weights are
technical 40%, cognitive 20%, personality 15%, leadership 10%, communication
10%, and behavioral 5%, with small retrieval-quality tie-breakers. Explicit
role-family rules protect high-confidence cases such as frontend engineering
from generic matches containing only “engineering.” On Render's memory-limited
free instance, semantic query encoding is disabled through configuration; the
same catalog-grounded lexical and metadata retrieval remains active. The FAISS
artifacts and semantic path remain available in environments with sufficient
memory.

## Prompt design

The prompt surface is intentionally small. The core system instruction limits
the assistant to SHL recommendations, requires catalog-only facts, forbids
invented products, and rejects prompt injection and out-of-scope advice. Product
selection is not delegated to free-form generation: deterministic parsing,
retrieval, ranking, and schema validation produce the shortlist. This reduces
hallucination risk and keeps behavior stable when no model credential is
available. Gemini configuration is optional; the deterministic path is the
production fallback and the deployed service does not require an API key.

## Evaluation and iteration

I replayed 10 supplied conversation traces (38 turns) through the same public
agent contract used by the API. The report measures Recall@10, clarification,
recommendation overlap, refinement, comparison, hallucination rate, and schema
compliance. The current replay result is:

| Metric | Result |
|---|---:|
| Recall@10 | 0.719 |
| Clarification accuracy | 0.632 |
| Recommendation overlap accuracy | 0.194 |
| Refinement accuracy | 0.901 |
| Comparison accuracy | 0.974 |
| Hallucination rate | 0.000 |
| Schema compliance | 1.000 |

The relatively low overlap score is useful evidence rather than something I
hide: the ranker often returns valid catalog alternatives beyond the small
expected set. Recall@10, zero hallucinations, and schema compliance better
reflect the primary constraints, while overlap identifies ranking precision as
the main future improvement area. Unit and API regression coverage currently
contains 31 passing tests, including stateless refinement, comparison grounding,
prompt-injection refusal, catalog URL validation, and OpenAPI availability.

Two failures directly shaped the design. First, the misspelling “fronend
engineer” initially matched generic engineering products. I added typo
normalization, frontend role-family filtering, and a regression test; the same
query changed from unrelated industrial results to seven frontend assessments,
all relevant by the test's role-family criterion. Second, loading the sentence
transformer on a small Render instance caused `/chat` to return a proxy HTML
error. Disabling semantic encoding only for that deployment reduced memory use;
live probes then verified both `/health` and `/chat` returned HTTP 200 JSON.

## AI-tool usage

I used OpenAI Codex for agentic coding support: repository inspection,
implementation suggestions, targeted edits, regression-test generation,
debugging the deployed API, and Render configuration. I reviewed the resulting
architecture and behavior through source inspection, automated tests, trace
replay metrics, and live endpoint probes. No no-code builder was used.
