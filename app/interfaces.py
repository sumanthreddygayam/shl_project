"""Ports implemented by runtime and offline pipeline adapters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from app.schemas import Assessment, ChatMessage, ChatResponse
from app.state import (
    ComparisonResult,
    Intent,
    RankedAssessment,
    Requirements,
    RetrievalCandidate,
    RetrievalQuery,
    SafetyDecision,
)


class CatalogSource(Protocol):
    def download(self, source_url: str) -> Any: ...
    def normalize(self, payload: Any) -> list[Assessment]: ...
    def write(self, assessments: Sequence[Assessment], destination: Path) -> None: ...


class EmbeddingIndexBuilder(Protocol):
    def build(self, assessments: Sequence[Assessment], destination: Path) -> None: ...


class ConversationParser(Protocol):
    async def parse(self, messages: Sequence[ChatMessage]) -> Requirements: ...


class SafetyGuard(Protocol):
    async def inspect(self, messages: Sequence[ChatMessage]) -> SafetyDecision: ...


class IntentRouter(Protocol):
    async def route(
        self, messages: Sequence[ChatMessage], requirements: Requirements
    ) -> Intent: ...


class ClarificationPolicy(Protocol):
    def next_question(
        self, requirements: Requirements, clarification_count: int, turn_count: int
    ) -> str | None: ...


class AssessmentRetriever(Protocol):
    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalCandidate]: ...


class AssessmentRanker(Protocol):
    def rank(
        self,
        requirements: Requirements,
        candidates: Sequence[RetrievalCandidate],
        limit: int = 10,
    ) -> list[RankedAssessment]: ...


class AssessmentComparator(Protocol):
    def compare(
        self, names: Sequence[str], catalog: Sequence[Assessment]
    ) -> ComparisonResult: ...


class ResponseGenerator(Protocol):
    async def generate(self, state: dict[str, Any]) -> ChatResponse: ...


class RecommenderAgent(Protocol):
    async def chat(self, messages: Sequence[ChatMessage]) -> ChatResponse: ...
