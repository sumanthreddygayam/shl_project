"""Internal state reconstructed afresh for every API request."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, TypedDict

from pydantic import Field

from app.schemas import Assessment, ChatMessage, Recommendation, StrictModel


class Intent(StrEnum):
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"
    REFUSE = "refuse"


class SafetyReason(StrEnum):
    ALLOWED = "allowed"
    PROMPT_INJECTION = "prompt_injection"
    OUT_OF_SCOPE = "out_of_scope"
    NON_SHL = "non_shl"


class Requirements(StrictModel):
    role: str | None = None
    experience: str | None = None
    seniority: str | None = None
    technical_skills: list[str] = Field(default_factory=list)
    personality_needed: bool = False
    leadership_needed: bool = False
    communication_needed: bool = False
    stakeholder_management: bool = False
    constraints: list[str] = Field(default_factory=list)


class SafetyDecision(StrictModel):
    allowed: bool
    reason: SafetyReason = SafetyReason.ALLOWED
    matched_pattern: str | None = None


class RetrievalQuery(StrictModel):
    text: str
    requirements: Requirements
    limit: int = Field(default=20, ge=1, le=20)


class RetrievalCandidate(StrictModel):
    assessment: Assessment
    semantic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    keyword_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata_score: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_fields: list[str] = Field(default_factory=list)


class ScoreBreakdown(StrictModel):
    technical_fit: float = Field(default=0.0, ge=0.0, le=1.0)
    cognitive_fit: float = Field(default=0.0, ge=0.0, le=1.0)
    personality_fit: float = Field(default=0.0, ge=0.0, le=1.0)
    leadership_fit: float = Field(default=0.0, ge=0.0, le=1.0)
    communication_fit: float = Field(default=0.0, ge=0.0, le=1.0)
    behavioral_fit: float = Field(default=0.0, ge=0.0, le=1.0)
    total: float = Field(default=0.0, ge=0.0, le=1.0)


class RankedAssessment(StrictModel):
    assessment: Assessment
    score: ScoreBreakdown
    retrieval: RetrievalCandidate


class ComparisonResult(StrictModel):
    assessment_ids: list[str] = Field(min_length=2)
    markdown_table: str = Field(min_length=1)


def _replace(_left: object, right: object) -> object:
    """LangGraph reducer: nodes replace values instead of retaining memory."""
    return right


class AgentState(TypedDict, total=False):
    messages: Annotated[list[ChatMessage], _replace]
    requirements: Annotated[Requirements, _replace]
    intent: Annotated[Intent, _replace]
    safety: Annotated[SafetyDecision, _replace]
    clarification_count: Annotated[int, _replace]
    turn_count: Annotated[int, _replace]
    comparison_names: Annotated[list[str], _replace]
    candidates: Annotated[list[RetrievalCandidate], _replace]
    ranked: Annotated[list[RankedAssessment], _replace]
    comparison: Annotated[ComparisonResult | None, _replace]
    reply: Annotated[str, _replace]
    recommendations: Annotated[list[Recommendation], _replace]
    end_of_conversation: Annotated[bool, _replace]
