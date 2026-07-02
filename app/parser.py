"""Stateless conversation parsing and requirement extraction."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.schemas import ChatMessage, MessageRole
from app.state import Requirements


_SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "java": ("java", "spring", "j2ee"),
    "javascript": (
        "javascript",
        "node",
        "react",
        "angular",
        "typescript",
        "frontend",
        "front end",
        "front-end",
        "fronend",
        "fronted",
    ),
    "python": ("python", "django", "flask", "fastapi"),
    "sql": ("sql", "database", "mysql", "postgres", "oracle"),
    "c#": ("c#", ".net", "dotnet"),
    "aws": ("aws", "cloud"),
    "devops": ("devops", "docker", "kubernetes", "ci/cd"),
    "docker": ("docker",),
    "sales": ("sales", "account executive", "business development"),
    "customer service": ("customer service", "call center", "contact center"),
    "data science": ("data science", "machine learning", "analytics", "ai"),
    "excel": ("excel", "spreadsheet"),
    "word": ("word", "ms word", "microsoft word"),
    "accounting": ("accounting", "financial accounting", "finance"),
    "statistics": ("statistics", "stats"),
    "linux": ("linux",),
    "networking": ("networking", "network"),
}

_ROLE_PATTERNS = [
    r"\b(?:i am |i'm |we are |we're )?hiring (?:a |an |for )?(?P<role>[a-z0-9+#.\-/ ]{3,80})(?: with| who|,|\.|$)",
    r"\b(?:role|position|job)\s*(?:is|:|-)\s*(?P<role>[a-z0-9+#.\-/ ]{3,80})(?: with|,|\.|$)",
    r"\b(?:need|looking for)\s+(?:a |an )?(?P<role>[a-z0-9+#.\-/ ]{3,80})(?: with| who|,|\.|$)",
]

_SENIORITY_PATTERNS = [
    (r"\b(?:intern|graduate|entry[- ]level|junior)\b", "junior"),
    (r"\b(?:mid[- ]level|intermediate)\b", "mid"),
    (r"\b(?:senior|lead|principal|staff)\b", "senior"),
    (r"\b(?:manager|director|head of|executive|vp)\b", "leadership"),
]


class RuleBasedConversationParser:
    """Reconstruct requirements from the full `messages[]` payload."""

    async def parse(self, messages: Sequence[ChatMessage]) -> Requirements:
        user_texts = [
            message.content
            for message in messages
            if message.role == MessageRole.USER
        ]
        text = "\n".join(user_texts).lower()
        latest = user_texts[-1].lower() if user_texts else ""
        # Refinement language only has meaning after an earlier user turn. A first
        # request such as "frontend engineer with 4 years experience" is a complete
        # role description, not a refinement.
        latest_is_refinement = len(user_texts) > 1 and _is_refinement(latest)
        latest_role = _extract_role(latest, allow_standalone=True)

        requirements = Requirements()
        requirements.role = latest_role if latest_role and not latest_is_refinement else _extract_role(text)
        requirements.experience = _extract_experience(latest) or _extract_experience(text)
        requirements.seniority = _extract_seniority(
            latest if latest_role and not latest_is_refinement else text,
            requirements.experience,
        )
        requirements.technical_skills = (
            _extract_skills(latest)
            if latest_role and not latest_is_refinement
            else _extract_skills(text)
        )
        requirements.personality_needed = _has_any(
            text, ("personality", "behavioral style", "behavioural style", "opq")
        )
        requirements.leadership_needed = _has_any(
            text, ("leadership", "leader", "management", "manager", "supervisor")
        )
        requirements.communication_needed = _has_any(
            text, ("communication", "communicate", "presentation", "verbal", "writing")
        )
        requirements.stakeholder_management = _has_any(
            text, ("stakeholder", "client-facing", "client facing", "influence")
        )
        requirements.constraints = _extract_constraints(
            latest if latest_role and not latest_is_refinement else text,
            latest,
        )
        return requirements


def _extract_role(text: str, allow_standalone: bool = False) -> str | None:
    for pattern in _ROLE_PATTERNS:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            role = _clean_role(matches[-1].group("role"))
            if role and role not in {"assessment", "test", "tests"}:
                return role
    for role_hint in (
        "graduate management trainee",
        "senior leadership",
        "sales",
        "customer service",
        "contact center",
        "finance",
        "accounting",
        "healthcare",
        "plant operators",
        "admin assistants",
        "administrative assistants",
        "frontend developer",
        "front end developer",
        "front-end developer",
        "java developer",
        "software developer",
        "developer",
        "manager",
        "leader",
        "analyst",
    ):
        if role_hint in text:
            return _clean_role(role_hint)
    if allow_standalone:
        standalone = re.search(
            r"\b(?P<role>[a-z+#. -]{2,50}?\b(?:developer|engineer|analyst|assistant|operator|manager|trainee|sales|support))\b",
            text,
            flags=re.IGNORECASE,
        )
        if standalone:
            return _clean_role(standalone.group("role"))
    return None


def _extract_experience(text: str) -> str | None:
    match = re.search(
        r"\b(?P<years>\d{1,2})\s*\+?\s*(?:years|yrs|year)(?:\s+of\s+exper(?:ience|ince))?\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return f"{match.group('years')} years"
    return None


def _extract_seniority(text: str, experience: str | None) -> str | None:
    for pattern, value in _SENIORITY_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return value
    if experience:
        years = int(re.search(r"\d+", experience).group(0))  # type: ignore[union-attr]
        if years <= 2:
            return "junior"
        if years <= 6:
            return "mid"
        return "senior"
    return None


def _extract_skills(text: str) -> list[str]:
    skills: list[str] = []
    for canonical, aliases in _SKILL_ALIASES.items():
        if any(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) for alias in aliases):
            skills.append(canonical)
    return skills


def _extract_constraints(text: str, latest: str) -> list[str]:
    constraints: list[str] = []
    for phrase in ("remote", "adaptive", "short", "quick", "entry level", "graduate"):
        if phrase in text:
            constraints.append(phrase)
    for phrase in (
        "cognitive",
        "situational judgement",
        "situational judgment",
        "leadership benchmark",
        "sales organization",
        "re-skill",
        "reskill",
        "audit stack",
        "contact center",
        "spoken english",
        "finance",
        "accounting",
        "health and safety",
        "dependability",
        "hipaa",
        "medical terminology",
    ):
        if phrase in text:
            constraints.append(phrase)
    if latest.startswith("add "):
        constraints.append(latest)
    return sorted(set(constraints))


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _clean_role(role: str) -> str:
    role = re.sub(
        r"\b(?:assessment|test|tests|candidate|candidates|with|who)\b", "", role,
        flags=re.IGNORECASE,
    )
    role = re.sub(
        r"\b(?:entry[- ]level|junior|mid[- ]level|senior|lead|principal|staff|graduate)\b",
        "",
        role,
        flags=re.IGNORECASE,
    )
    role = re.sub(r"\b(?:front[ -]?end|fronend|fronted)\b", "frontend", role)
    return " ".join(role.strip(" .,-").split())


def _is_refinement(text: str) -> bool:
    return bool(
        re.search(
            r"\b(add|include|also|instead|change|remove|drop|replace|make it|with|without)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
