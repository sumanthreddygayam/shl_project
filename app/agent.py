"""Application-level conversational recommender orchestration."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from app.comparison import CatalogComparator, extract_comparison_names
from app.parser import RuleBasedConversationParser
from app.ranking import HybridAssessmentRanker
from app.retrieval import HybridAssessmentRetriever, build_query_text
from app.safety import REFUSAL_REPLY, RuleBasedSafetyGuard
from app.schemas import ChatMessage, ChatResponse, MessageRole, Recommendation, TestType
from app.state import Intent, Requirements, RetrievalQuery


class ClarificationPolicy:
    """Asks at most two concise questions, one at a time."""

    def next_question(
        self, requirements: Requirements, clarification_count: int, turn_count: int
    ) -> str | None:
        if turn_count >= 8 or clarification_count >= 2:
            return None
        if not requirements.role:
            return "What role are you hiring for?"
        has_strong_signal = (
            len(requirements.technical_skills) >= 2
            or requirements.personality_needed
            or requirements.leadership_needed
            or requirements.communication_needed
            or requirements.stakeholder_management
            or len(requirements.constraints) >= 2
        )
        if not has_strong_signal and not requirements.seniority and not requirements.experience:
            return "What seniority level is this role?"
        return None


class IntentRouter:
    async def route(
        self, messages: Sequence[ChatMessage], requirements: Requirements
    ) -> Intent:
        latest = latest_user_text(messages).lower()
        if extract_comparison_names(latest):
            return Intent.COMPARE
        if _is_refinement(latest):
            return Intent.REFINE
        if not requirements.role:
            return Intent.CLARIFY
        if requirements.experience or requirements.seniority:
            return Intent.RECOMMEND
        return Intent.CLARIFY


class ConversationalSHLAgent:
    """Production runtime agent: stateless in, stateless out."""

    def __init__(
        self,
        parser: RuleBasedConversationParser | None = None,
        safety: RuleBasedSafetyGuard | None = None,
        router: IntentRouter | None = None,
        clarifier: ClarificationPolicy | None = None,
        retriever: HybridAssessmentRetriever | None = None,
        ranker: HybridAssessmentRanker | None = None,
        comparator: CatalogComparator | None = None,
    ) -> None:
        self.parser = parser or RuleBasedConversationParser()
        self.safety = safety or RuleBasedSafetyGuard()
        self.router = router or IntentRouter()
        self.clarifier = clarifier or ClarificationPolicy()
        self.retriever = retriever or HybridAssessmentRetriever()
        self.ranker = ranker or HybridAssessmentRanker()
        self.comparator = comparator or CatalogComparator()

    async def chat(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        safety = await self.safety.inspect(messages)
        if not safety.allowed:
            return ChatResponse(reply=REFUSAL_REPLY, recommendations=[])

        requirements = await self.parser.parse(messages)
        intent = await self.router.route(messages, requirements)
        turn_count = sum(1 for message in messages if message.role == MessageRole.USER)
        clarification_count = _assistant_question_count(messages)

        if intent == Intent.COMPARE:
            return self._compare(messages)

        question = self.clarifier.next_question(
            requirements, clarification_count, turn_count
        )
        if question and intent != Intent.REFINE:
            return ChatResponse(reply=question, recommendations=[])

        return await self._recommend(messages, requirements)

    def _compare(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        names = extract_comparison_names(latest_user_text(messages))
        try:
            comparison = self.comparator.compare(names, self.retriever.catalog.load())
        except ValueError as exc:
            return ChatResponse(
                reply=f"{exc}. Please provide two SHL assessment names to compare.",
                recommendations=[],
            )
        return ChatResponse(reply=comparison.markdown_table, recommendations=[])

    async def _recommend(
        self, messages: Sequence[ChatMessage], requirements: Requirements
    ) -> ChatResponse:
        latest = latest_user_text(messages)
        candidates = await self.retriever.retrieve(
            RetrievalQuery(
                text=build_query_text(requirements, latest),
                requirements=requirements,
                limit=20,
            )
        )
        ranked = self.ranker.rank(requirements, candidates, limit=10)
        ranked = _apply_role_focus(requirements, ranked)
        recommendations = [
            Recommendation(
                name=item.assessment.name,
                url=item.assessment.url,
                test_type=_test_type(item.assessment),
            )
            for item in ranked
        ]
        if not recommendations:
            return ChatResponse(
                reply=(
                    "I could not find a strong SHL catalog match for that request. "
                    "Please share the role, seniority, and key skills."
                ),
                recommendations=[],
            )
        role_phrase = f" for {requirements.role}" if requirements.role else ""
        return ChatResponse(
            reply=(
                f"Here are the strongest SHL assessment matches{role_phrase}. "
                "I used the catalog fields plus your stated skills, seniority, and trait requirements."
            ),
            recommendations=recommendations,
        )


def latest_user_text(messages: Sequence[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == MessageRole.USER:
            return message.content
    return ""


def _assistant_question_count(messages: Sequence[ChatMessage]) -> int:
    return sum(
        1
        for message in messages
        if message.role == MessageRole.ASSISTANT and "?" in message.content
    )


def _is_refinement(text: str) -> bool:
    return bool(
        re.search(
            r"\b(add|include|also|instead|change|remove|make it|with|without)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _test_type(assessment: Any) -> TestType:
    text = " ".join(
        [
            assessment.name,
            assessment.description,
            " ".join(getattr(assessment, "skills", [])),
            " ".join(getattr(assessment, "categories", [])),
        ]
    ).lower()
    if assessment.technical:
        return TestType.TECHNICAL
    if any(
        term in text
        for term in (
            "java",
            "javascript",
            "python",
            "sql",
            "programming",
            "developer",
            "front end",
            "frontend",
            "html",
            "css",
            "react",
            "angular",
            "aws",
            "docker",
        )
    ):
        return TestType.TECHNICAL
    if assessment.cognitive:
        return TestType.COGNITIVE
    if assessment.personality:
        return TestType.PERSONALITY
    if assessment.leadership:
        return TestType.LEADERSHIP
    if assessment.communication:
        return TestType.COMMUNICATION
    if assessment.behavioral:
        return TestType.BEHAVIORAL
    if assessment.stakeholder:
        return TestType.STAKEHOLDER
    return TestType.GENERAL


def _apply_role_focus(
    requirements: Requirements, ranked: list[Any]
) -> list[Any]:
    role = (requirements.role or "").lower()
    if any(term in role for term in ("frontend", "front end", "front-end")):
        focused = [
            item
            for item in ranked
            if _is_frontend_assessment(item.assessment)
        ]
        if len(focused) >= 3:
            return focused[:10]
    return ranked


def _is_frontend_assessment(assessment: Any) -> bool:
    text = " ".join(
        [
            assessment.name,
            assessment.description,
            " ".join(getattr(assessment, "skills", [])),
            " ".join(getattr(assessment, "categories", [])),
        ]
    ).lower()
    return any(
        term in text
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
