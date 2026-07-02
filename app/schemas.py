"""Validated public API and normalized catalog contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class StrictModel(BaseModel):
    """Base contract that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(StrictModel):
    role: MessageRole
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(StrictModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=16)

    @field_validator("messages")
    @classmethod
    def must_include_user_message(
        cls, messages: list[ChatMessage]
    ) -> list[ChatMessage]:
        if not any(message.role == MessageRole.USER for message in messages):
            raise ValueError("messages must contain at least one user message")
        return messages


class TestType(StrEnum):
    TECHNICAL = "technical"
    COGNITIVE = "cognitive"
    PERSONALITY = "personality"
    LEADERSHIP = "leadership"
    COMMUNICATION = "communication"
    BEHAVIORAL = "behavioral"
    STAKEHOLDER = "stakeholder"
    GENERAL = "general"


class Recommendation(StrictModel):
    name: str = Field(min_length=1)
    url: HttpUrl
    test_type: TestType


class ChatResponse(StrictModel):
    reply: str = Field(min_length=1)
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False


class HealthResponse(StrictModel):
    status: str = Field(default="ok", pattern=r"^ok$")


class Assessment(StrictModel):
    """Canonical representation of one SHL Individual Test Solution."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str
    url: HttpUrl
    job_levels: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    technical: bool = False
    cognitive: bool = False
    behavioral: bool = False
    personality: bool = False
    leadership: bool = False
    communication: bool = False
    stakeholder: bool = False
    remote: bool = False
    adaptive: bool = False

    def embedding_text(self) -> str:
        """Return the exact fields allowed in the embedding corpus."""
        sections = (
            self.name,
            self.description,
            ", ".join(self.categories),
            ", ".join(self.skills),
            ", ".join(self.job_levels),
        )
        return "\n".join(section for section in sections if section)


class Catalog(StrictModel):
    assessments: list[Assessment]

    @field_validator("assessments")
    @classmethod
    def assessment_ids_are_unique(
        cls, assessments: list[Assessment]
    ) -> list[Assessment]:
        ids = [assessment.id for assessment in assessments]
        if len(ids) != len(set(ids)):
            raise ValueError("catalog assessment ids must be unique")
        return assessments
