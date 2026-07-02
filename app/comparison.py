"""Catalog-grounded assessment comparison."""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

from app.schemas import Assessment
from app.state import ComparisonResult


class CatalogComparator:
    """Compares named assessments using only normalized catalog fields."""

    def compare(
        self, names: Sequence[str], catalog: Sequence[Assessment]
    ) -> ComparisonResult:
        if len(names) < 2:
            raise ValueError("comparison requires at least two assessment names")
        selected = [_find_assessment(name, catalog) for name in names[:2]]
        missing = [name for name, assessment in zip(names, selected, strict=False) if assessment is None]
        if missing:
            raise ValueError(f"Could not find SHL assessment(s): {', '.join(missing)}")
        a, b = selected[0], selected[1]
        assert a is not None and b is not None
        return ComparisonResult(
            assessment_ids=[a.id, b.id],
            markdown_table=_comparison_table(a, b),
        )


def extract_comparison_names(text: str) -> list[str]:
    normalized = text.strip()
    match = re.search(
        r"\bcompare\s+(?P<a>.+?)\s+(?:and|vs\.?|versus)\s+(?P<b>.+?)(?:\?|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if match:
        return [_clean_name(match.group("a")), _clean_name(match.group("b"))]
    match = re.search(
        r"\bdifference between\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)(?:\?|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if match:
        return [_clean_name(match.group("a")), _clean_name(match.group("b"))]
    return []


def _find_assessment(name: str, catalog: Sequence[Assessment]) -> Assessment | None:
    needle = _normalize(name)
    aliases = _aliases(needle)
    for assessment in catalog:
        normalized = _normalize(assessment.name)
        if normalized in aliases or needle == normalized:
            return assessment
    best: tuple[float, Assessment] | None = None
    for assessment in catalog:
        normalized = _normalize(assessment.name)
        score = max(SequenceMatcher(None, alias, normalized).ratio() for alias in aliases)
        if needle and needle in normalized:
            score = max(score, 0.88)
        if best is None or score > best[0]:
            best = (score, assessment)
    if best and best[0] >= 0.58:
        return best[1]
    return None


def _comparison_table(a: Assessment, b: Assessment) -> str:
    rows = [
        ("Name", a.name, b.name),
        ("Description", a.description, b.description),
        ("Categories", _join(a.categories), _join(b.categories)),
        ("Skills", _join(a.skills), _join(b.skills)),
        ("Job levels", _join(a.job_levels), _join(b.job_levels)),
        ("Technical", _yes(a.technical), _yes(b.technical)),
        ("Cognitive", _yes(a.cognitive), _yes(b.cognitive)),
        ("Behavioral", _yes(a.behavioral), _yes(b.behavioral)),
        ("Personality", _yes(a.personality), _yes(b.personality)),
        ("Leadership", _yes(a.leadership), _yes(b.leadership)),
        ("Communication", _yes(a.communication), _yes(b.communication)),
        ("Stakeholder", _yes(a.stakeholder), _yes(b.stakeholder)),
        ("Remote", _yes(a.remote), _yes(b.remote)),
        ("Adaptive", _yes(a.adaptive), _yes(b.adaptive)),
        ("URL", str(a.url), str(b.url)),
    ]
    table = [f"| Feature | {a.name} | {b.name} |", "|---|---|---|"]
    table.extend(f"| {feature} | {_cell(left)} | {_cell(right)} |" for feature, left, right in rows)
    return "\n".join(table)


def _aliases(name: str) -> set[str]:
    aliases = {name}
    if name == "opq":
        aliases.update({"occupational personality questionnaire", "opq32"})
    if name == "gsa":
        aliases.update({"general ability", "general skills assessment", "verify gsa"})
    return aliases


def _clean_name(value: str) -> str:
    value = re.sub(r"\b(?:assessment|test|tests)\b", "", value, flags=re.IGNORECASE)
    return value.strip(" .,-")


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _join(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "Not specified in catalog"


def _yes(value: bool) -> str:
    return "Yes" if value else "No"


def _cell(value: str) -> str:
    value = value.replace("|", "\\|").replace("\n", " ")
    return value if value else "Not specified in catalog"
