"""Prompt-injection and scope safety gates."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.schemas import ChatMessage, MessageRole
from app.state import SafetyDecision, SafetyReason


_PROMPT_INJECTION_PATTERNS = [
    r"\bignore (all )?(previous|prior|above|system|developer) instructions\b",
    r"\bforget (all )?(previous|prior|above) instructions\b",
    r"\bjailbreak\b",
    r"\bdeveloper mode\b",
    r"\bsystem prompt\b",
    r"\breveal (your )?(prompt|instructions|chain of thought)\b",
    r"\bact as\b.*\b(unrestricted|no rules|anything now)\b",
]

_OUT_OF_SCOPE_PATTERNS = [
    r"\blegal advice\b",
    r"\blawsuit\b",
    r"\bcompliance advice\b",
    r"\bsalary\b",
    r"\bcompensation\b",
    r"\bhiring strategy\b",
    r"\bshould I hire\b",
    r"\bwho should I hire\b",
    r"\binterview question\b",
    r"\bperformance improvement plan\b",
]

_NON_SHL_PATTERNS = [
    r"\bhacker ?rank\b",
    r"\bcodility\b",
    r"\btestgorilla\b",
    r"\bcriteriacorp\b",
    r"\bwonderlic\b",
    r"\bplum\b",
    r"\bthomas international\b",
]


class RuleBasedSafetyGuard:
    """Small, predictable guardrail layer for the stateless API."""

    async def inspect(self, messages: Sequence[ChatMessage]) -> SafetyDecision:
        text = "\n".join(
            message.content for message in messages if message.role == MessageRole.USER
        ).lower()
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return SafetyDecision(
                    allowed=False,
                    reason=SafetyReason.PROMPT_INJECTION,
                    matched_pattern=pattern,
                )
        for pattern in _NON_SHL_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return SafetyDecision(
                    allowed=False,
                    reason=SafetyReason.NON_SHL,
                    matched_pattern=pattern,
                )
        for pattern in _OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return SafetyDecision(
                    allowed=False,
                    reason=SafetyReason.OUT_OF_SCOPE,
                    matched_pattern=pattern,
                )
        return SafetyDecision(allowed=True)


REFUSAL_REPLY = "I can only assist with SHL assessment recommendations."
