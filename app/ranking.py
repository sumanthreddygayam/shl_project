"""Deterministic weighted assessment ranking."""

from __future__ import annotations

from collections.abc import Sequence

from app.schemas import Assessment
from app.state import RankedAssessment, Requirements, RetrievalCandidate, ScoreBreakdown


WEIGHTS = {
    "technical_fit": 0.40,
    "cognitive_fit": 0.20,
    "personality_fit": 0.15,
    "leadership_fit": 0.10,
    "communication_fit": 0.10,
    "behavioral_fit": 0.05,
}


class HybridAssessmentRanker:
    """Weighted scoring engine aligned to the project specification."""

    def rank(
        self,
        requirements: Requirements,
        candidates: Sequence[RetrievalCandidate],
        limit: int = 10,
    ) -> list[RankedAssessment]:
        ranked = [
            RankedAssessment(
                assessment=candidate.assessment,
                retrieval=candidate,
                score=_score_assessment(requirements, candidate),
            )
            for candidate in candidates
        ]
        ranked.sort(
            key=lambda item: (
                item.score.total,
                item.retrieval.semantic_score,
                item.retrieval.keyword_score,
                item.assessment.name.lower(),
            ),
            reverse=True,
        )
        return ranked[:limit]


def _score_assessment(
    requirements: Requirements, candidate: RetrievalCandidate
) -> ScoreBreakdown:
    assessment = candidate.assessment
    technical = _technical_fit(requirements, assessment)
    cognitive = 1.0 if assessment.cognitive else _soft_match(candidate, 0.35)
    personality = _flag_fit(requirements.personality_needed, assessment.personality)
    leadership = _flag_fit(requirements.leadership_needed, assessment.leadership)
    communication = _flag_fit(
        requirements.communication_needed, assessment.communication
    )
    behavioral = 1.0 if assessment.behavioral else _soft_match(candidate, 0.25)

    total = (
        technical * WEIGHTS["technical_fit"]
        + cognitive * WEIGHTS["cognitive_fit"]
        + personality * WEIGHTS["personality_fit"]
        + leadership * WEIGHTS["leadership_fit"]
        + communication * WEIGHTS["communication_fit"]
        + behavioral * WEIGHTS["behavioral_fit"]
    )
    # Retrieval quality is a small tie-shaping boost without changing the specified
    # component weights.
    retrieval_boost = (
        candidate.semantic_score * 0.05
        + candidate.keyword_score * 0.03
        + candidate.metadata_score * 0.02
    )
    preferred_boost = 0.20 if "preferred" in candidate.matched_fields else 0.0
    return ScoreBreakdown(
        technical_fit=technical,
        cognitive_fit=cognitive,
        personality_fit=personality,
        leadership_fit=leadership,
        communication_fit=communication,
        behavioral_fit=behavioral,
        total=min(1.0, total + retrieval_boost + preferred_boost),
    )


def _technical_fit(requirements: Requirements, assessment: Assessment) -> float:
    if requirements.technical_skills:
        haystack = _assessment_text(assessment)
        matches = sum(skill.lower() in haystack for skill in requirements.technical_skills)
        if matches:
            return min(1.0, 0.55 + matches / max(1, len(requirements.technical_skills)) * 0.45)
        return 0.45 if assessment.technical else 0.0
    if requirements.role and any(
        word in requirements.role.lower()
        for word in ("developer", "engineer", "programmer", "technical", "data")
    ):
        return 0.85 if assessment.technical else 0.25
    return 0.35 if assessment.technical else 0.55


def _flag_fit(needed: bool, present: bool) -> float:
    if needed:
        return 1.0 if present else 0.05
    return 0.35 if present else 0.6


def _soft_match(candidate: RetrievalCandidate, fallback: float) -> float:
    return max(fallback, candidate.metadata_score * 0.75, candidate.keyword_score * 0.6)


def _assessment_text(assessment: Assessment) -> str:
    return " ".join(
        [
            assessment.name,
            assessment.description,
            " ".join(assessment.skills),
            " ".join(assessment.categories),
            " ".join(assessment.job_levels),
        ]
    ).lower()
