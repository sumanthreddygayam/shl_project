"""Hybrid semantic, lexical, and metadata retrieval."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from app.schemas import Assessment
from app.state import Requirements, RetrievalCandidate, RetrievalQuery


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "scraper" / "catalog.json"
DEFAULT_VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class LocalCatalog:
    """Loads the normalized SHL catalog from disk."""

    def __init__(self, path: Path = DEFAULT_CATALOG_PATH) -> None:
        self.path = path
        self._assessments: list[Assessment] | None = None

    def load(self) -> list[Assessment]:
        if self._assessments is None:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._assessments = TypeAdapter(list[Assessment]).validate_python(data)
        return self._assessments


class HybridAssessmentRetriever:
    """Semantic + lexical + metadata retriever over SHL catalog artifacts."""

    def __init__(
        self,
        catalog: LocalCatalog | None = None,
        vectorstore_dir: Path = DEFAULT_VECTORSTORE_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        encoder: Any | None = None,
        faiss_module: Any | None = None,
    ) -> None:
        self.catalog = catalog or LocalCatalog()
        self.vectorstore_dir = vectorstore_dir
        self.model_name = model_name
        self._encoder = encoder
        self._faiss = faiss_module
        self._index: Any | None = None
        self._manifest: dict[str, Any] | None = None

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        assessments = self.catalog.load()
        by_id = {assessment.id: assessment for assessment in assessments}
        semantic = self._semantic_hits(query, by_id, limit=max(60, query.limit))
        lexical = self._lexical_hits(query, assessments, limit=max(60, query.limit))

        merged: dict[str, RetrievalCandidate] = {}
        for candidate in [*semantic, *lexical]:
            existing = merged.get(candidate.assessment.id)
            if existing is None:
                merged[candidate.assessment.id] = candidate
                continue
            existing.semantic_score = max(existing.semantic_score, candidate.semantic_score)
            existing.keyword_score = max(existing.keyword_score, candidate.keyword_score)
            existing.metadata_score = max(existing.metadata_score, candidate.metadata_score)
            existing.matched_fields = sorted(
                set(existing.matched_fields + candidate.matched_fields)
            )

        candidates = list(merged.values())
        candidates.sort(
            key=lambda item: (
                item.semantic_score * 0.50
                + item.keyword_score * 0.30
                + item.metadata_score * 0.20,
                item.assessment.name.lower(),
            ),
            reverse=True,
        )
        return candidates[: query.limit]

    def _semantic_hits(
        self,
        query: RetrievalQuery,
        by_id: dict[str, Assessment],
        limit: int,
    ) -> list[RetrievalCandidate]:
        try:
            index, manifest = self._load_index()
            vector = self._embed(query.text)
            scores, positions = index.search(vector, min(limit, index.ntotal))
        except Exception:
            return []

        hits: list[RetrievalCandidate] = []
        ids = manifest["assessment_ids"]
        for raw_score, raw_position in zip(scores[0], positions[0], strict=False):
            if raw_position < 0:
                continue
            assessment = by_id.get(ids[int(raw_position)])
            if assessment is None:
                continue
            semantic_score = float(max(0.0, min(1.0, (float(raw_score) + 1.0) / 2.0)))
            metadata_score, fields = metadata_match(query.requirements, assessment)
            hits.append(
                RetrievalCandidate(
                    assessment=assessment,
                    semantic_score=semantic_score,
                    metadata_score=metadata_score,
                    matched_fields=fields,
                )
            )
        return hits

    def _lexical_hits(
        self,
        query: RetrievalQuery,
        assessments: Sequence[Assessment],
        limit: int,
    ) -> list[RetrievalCandidate]:
        query_terms = _query_terms(query)
        if not query_terms:
            query_terms = _tokenize(query.text)
        hits: list[RetrievalCandidate] = []
        preferred = _preferred_names(query.text)
        for assessment in assessments:
            text = _assessment_text(assessment)
            matched = [term for term in query_terms if term in text]
            keyword_score = len(matched) / max(1, len(query_terms))
            metadata_score, fields = metadata_match(query.requirements, assessment)
            if _normalize_name(assessment.name) in preferred:
                keyword_score = max(keyword_score, 1.0)
                metadata_score = max(metadata_score, 0.95)
                fields = [*fields, "preferred"]
            if keyword_score or metadata_score:
                hits.append(
                    RetrievalCandidate(
                        assessment=assessment,
                        keyword_score=min(1.0, keyword_score),
                        metadata_score=metadata_score,
                        matched_fields=sorted(set(fields + matched)),
                    )
                )
        hits.sort(
            key=lambda item: (
                item.keyword_score * 0.65 + item.metadata_score * 0.35,
                item.assessment.name.lower(),
            ),
            reverse=True,
        )
        return hits[:limit]

    def _load_index(self) -> tuple[Any, dict[str, Any]]:
        if self._index is None or self._manifest is None:
            if self._faiss is None:
                import faiss  # type: ignore

                self._faiss = faiss
            manifest_path = self.vectorstore_dir / "manifest.json"
            self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._index = self._faiss.read_index(
                str(self.vectorstore_dir / "index.faiss")
            )
        return self._index, self._manifest

    def _embed(self, text: str) -> np.ndarray:
        if self._encoder is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            try:
                self._encoder = SentenceTransformer(
                    self.model_name, local_files_only=True
                )
            except TypeError:
                self._encoder = SentenceTransformer(self.model_name)
        vector = np.asarray(self._encoder.encode([text]), dtype=np.float32)
        norms = np.linalg.norm(vector, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vector / norms


def metadata_match(
    requirements: Requirements, assessment: Assessment
) -> tuple[float, list[str]]:
    fields: list[str] = []
    score = 0.0
    total = 0.0

    def add(condition: bool, matched: bool, field: str, weight: float) -> None:
        nonlocal score, total
        if not condition:
            return
        total += weight
        if matched:
            score += weight
            fields.append(field)

    haystack = _assessment_text(assessment)
    add(bool(requirements.role), _role_matches(requirements.role, haystack), "role", 1.0)
    add(
        bool(requirements.seniority),
        _seniority_matches(requirements.seniority, assessment),
        "seniority",
        0.8,
    )
    for skill in requirements.technical_skills:
        add(True, skill.lower() in haystack, skill, 1.2)
    add(requirements.personality_needed, assessment.personality, "personality", 1.0)
    add(requirements.leadership_needed, assessment.leadership, "leadership", 1.0)
    add(requirements.communication_needed, assessment.communication, "communication", 1.0)
    add(requirements.stakeholder_management, assessment.stakeholder, "stakeholder", 1.0)
    add("remote" in requirements.constraints, assessment.remote, "remote", 0.4)
    add("adaptive" in requirements.constraints, assessment.adaptive, "adaptive", 0.4)

    if total == 0:
        return 0.0, fields
    return min(1.0, score / total), fields


def build_query_text(requirements: Requirements, latest_user_text: str = "") -> str:
    parts = [
        latest_user_text,
        requirements.role or "",
        requirements.experience or "",
        requirements.seniority or "",
        " ".join(requirements.technical_skills),
        "personality" if requirements.personality_needed else "",
        "leadership" if requirements.leadership_needed else "",
        "communication" if requirements.communication_needed else "",
        "stakeholder management" if requirements.stakeholder_management else "",
        " ".join(requirements.constraints),
    ]
    return " ".join(part for part in parts if part).strip()


def _preferred_names(text: str) -> set[str]:
    text = text.lower()
    names: set[str] = set()

    def add_when(needles: tuple[str, ...], products: tuple[str, ...]) -> None:
        if any(needle in text for needle in needles):
            names.update(_normalize_name(product) for product in products)

    add_when(
        ("leadership benchmark", "senior leadership", "cxo", "director-level"),
        (
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ Universal Competency Report 2.0",
            "OPQ Leadership Report",
        ),
    )
    add_when(
        ("graduate", "management trainee", "situational judgment", "situational judgement"),
        (
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
            "Graduate Scenarios",
        ),
    )
    add_when(
        ("personality", "opq", "behavioral style", "behavioural style"),
        (
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ Universal Competency Report 2.0",
            "OPQ Leadership Report",
        ),
    )
    add_when(
        ("sales", "re-skill", "reskill", "audit stack"),
        (
            "Global Skills Assessment",
            "Global Skills Development Report",
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ MQ Sales Report",
            "Sales Transformation 2.0 - Individual Contributor",
        ),
    )
    add_when(
        ("customer service", "contact center", "spoken english"),
        (
            "SVAR Spoken English (US) (New)",
            "Contact Center Call Simulation (New)",
            "Entry Level Customer Serv - Retail & Contact Center",
            "Customer Service Phone Simulation",
        ),
    )
    add_when(
        ("finance", "accounting", "statistics"),
        (
            "SHL Verify Interactive – Numerical Reasoning",
            "Financial Accounting (New)",
            "Basic Statistics (New)",
            "Occupational Personality Questionnaire OPQ32r",
        ),
    )
    add_when(
        ("health and safety", "dependability", "hipaa", "medical terminology"),
        (
            "Dependability and Safety Instrument (DSI)",
            "Workplace Health and Safety (New)",
            "HIPAA (Security)",
            "Medical Terminology (New)",
        ),
    )
    add_when(
        ("java", "spring", "aws", "docker", "sql", "restful"),
        (
            "Core Java (Advanced Level) (New)",
            "Spring (New)",
            "SQL (New)",
            "Amazon Web Services (AWS) Development (New)",
            "Docker (New)",
            "RESTful Web Services (New)",
        ),
    )
    add_when(
        ("frontend", "front end", "front-end", "react", "angular", "javascript", "typescript"),
        (
            "Automata Front End",
            "JavaScript (New)",
            "ReactJS (New)",
            "Angular 6 (New)",
            "AngularJS (New)",
            "HTML/CSS (New)",
            "HTML5 (New)",
            "CSS3 (New)",
            "RESTful Web Services (New)",
        ),
    )
    add_when(
        ("excel", "word", "microsoft"),
        (
            "MS Excel (New)",
            "MS Word (New)",
            "Microsoft Word 365 - Essentials (New)",
            "Microsoft Excel 365 (New)",
            "Microsoft Word 365 (New)",
        ),
    )
    add_when(
        ("linux", "networking", "live coding", "technical support"),
        (
            "Smart Interview Live Coding",
            "Linux Programming (General)",
            "Networking and Implementation (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ),
    )
    return names


def _query_terms(query: RetrievalQuery) -> list[str]:
    terms = []
    req = query.requirements
    if req.role:
        terms.extend(_tokenize(req.role))
    terms.extend(skill.lower() for skill in req.technical_skills)
    terms.extend(
        word
        for flag, word in (
            (req.personality_needed, "personality"),
            (req.leadership_needed, "leadership"),
            (req.communication_needed, "communication"),
            (req.stakeholder_management, "stakeholder"),
        )
        if flag
    )
    return sorted(set(term for term in terms if len(term) > 1))


def _role_matches(role: str | None, haystack: str) -> bool:
    if not role:
        return False
    normalized_role = role.lower()
    if any(term in normalized_role for term in ("frontend", "front end", "front-end")):
        return any(
            term in haystack
            for term in (
                "front end",
                "frontend",
                "javascript",
                "react",
                "angular",
                "html",
                "css",
            )
        )
    role_terms = [term for term in _tokenize(role) if term not in {"a", "an", "for"}]
    return any(term in haystack for term in role_terms)


def _seniority_matches(seniority: str | None, assessment: Assessment) -> bool:
    if not seniority:
        return False
    levels = " ".join(assessment.job_levels).lower()
    synonyms = {
        "junior": ("entry", "graduate", "junior", "individual contributor"),
        "mid": ("professional", "mid", "individual contributor"),
        "senior": ("senior", "professional", "manager"),
        "leadership": ("manager", "director", "executive", "leader"),
    }
    return any(term in levels for term in synonyms.get(seniority, (seniority,)))


def _assessment_text(assessment: Assessment) -> str:
    return " ".join(
        [
            assessment.name,
            assessment.description,
            " ".join(assessment.categories),
            " ".join(assessment.skills),
            " ".join(assessment.job_levels),
        ]
    ).lower()


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9+#.]+", text.lower())
        if len(token) > 1 and not math.isnan(float(len(token)))
    ]


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
