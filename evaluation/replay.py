"""Conversation trace loader and replay harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.agent import ConversationalSHLAgent
from app.retrieval import LocalCatalog
from app.schemas import ChatMessage, MessageRole
from evaluation.metrics import (
    hallucination_rate,
    overlap_accuracy,
    recall_at_k,
    safe_mean,
)


DEFAULT_TRACES_DIR = Path("sample_conversations")
DEFAULT_REPORT_PATH = Path("evaluation/evaluation_report.json")


@dataclass(frozen=True)
class TraceTurn:
    user: str
    expected_reply: str = ""
    expected_recommendations: list[str] = field(default_factory=list)
    expected_end: bool = False


@dataclass(frozen=True)
class ConversationTrace:
    trace_id: str
    turns: list[TraceTurn]


def load_traces(root: Path = DEFAULT_TRACES_DIR) -> list[ConversationTrace]:
    paths = sorted(root.rglob("*.md"))
    traces = []
    for path in paths:
        turns = _parse_markdown_trace(path.read_text(encoding="utf-8", errors="replace"))
        if turns:
            traces.append(ConversationTrace(trace_id=path.stem, turns=turns))
    return traces


async def replay_traces(
    traces: list[ConversationTrace] | None = None,
    agent: ConversationalSHLAgent | None = None,
) -> dict[str, object]:
    traces = traces or load_traces()
    agent = agent or ConversationalSHLAgent()
    catalog = LocalCatalog().load()
    catalog_names = {assessment.name for assessment in catalog}

    turn_reports: list[dict[str, object]] = []
    recall_scores: list[float] = []
    clarification_scores: list[float] = []
    recommendation_scores: list[float] = []
    refinement_scores: list[float] = []
    comparison_scores: list[float] = []
    hallucination_scores: list[float] = []
    schema_scores: list[float] = []

    for trace in traces:
        messages: list[ChatMessage] = []
        previous_expected: list[str] = []
        for index, turn in enumerate(trace.turns, start=1):
            messages.append(ChatMessage(role=MessageRole.USER, content=turn.user))
            response = await agent.chat(messages)
            actual_names = [recommendation.name for recommendation in response.recommendations]
            expected = turn.expected_recommendations

            recall = recall_at_k(expected, actual_names, 10)
            accuracy = overlap_accuracy(expected, actual_names)
            hallucination = hallucination_rate(actual_names, catalog_names)
            is_schema_valid = 1.0
            is_expected_clarify = not expected
            is_actual_clarify = not response.recommendations and "?" in response.reply
            clarification = 1.0 if is_expected_clarify == is_actual_clarify else 0.0
            is_refinement = bool(previous_expected and expected and expected != previous_expected)
            refinement = accuracy if is_refinement else 1.0
            is_comparison = "compare" in turn.user.lower() or "difference" in turn.user.lower()
            comparison = 1.0 if not is_comparison else float("| Feature |" in response.reply)

            recall_scores.append(recall)
            clarification_scores.append(clarification)
            recommendation_scores.append(accuracy)
            refinement_scores.append(refinement)
            comparison_scores.append(comparison)
            hallucination_scores.append(hallucination)
            schema_scores.append(is_schema_valid)

            turn_reports.append(
                {
                    "trace_id": trace.trace_id,
                    "turn": index,
                    "user": turn.user,
                    "expected_recommendations": expected,
                    "actual_recommendations": actual_names,
                    "recall_at_10": recall,
                    "recommendation_accuracy": accuracy,
                    "clarification_accuracy": clarification,
                    "refinement_accuracy": refinement,
                    "comparison_accuracy": comparison,
                    "hallucination_rate": hallucination,
                    "schema_compliance": is_schema_valid,
                }
            )
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=response.reply)
            )
            if expected:
                previous_expected = expected

    return {
        "trace_count": len(traces),
        "turn_count": len(turn_reports),
        "metrics": {
            "Recall@10": safe_mean(recall_scores),
            "clarification_accuracy": safe_mean(clarification_scores),
            "recommendation_accuracy": safe_mean(recommendation_scores),
            "refinement_accuracy": safe_mean(refinement_scores),
            "comparison_accuracy": safe_mean(comparison_scores),
            "hallucination_rate": safe_mean(hallucination_scores),
            "schema_compliance": safe_mean(schema_scores),
        },
        "turns": turn_reports,
    }


def write_report(report: dict[str, object], path: Path = DEFAULT_REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_markdown_trace(text: str) -> list[TraceTurn]:
    chunks = re.split(r"^### Turn \d+\s*$", text, flags=re.MULTILINE)[1:]
    turns = []
    for chunk in chunks:
        user = _extract_blockquote_after_heading(chunk, "User")
        agent = _extract_agent_section(chunk)
        if not user:
            continue
        recommendations = _extract_recommendation_names(agent)
        end = "end_of_conversation`: **true**" in chunk.lower()
        turns.append(
            TraceTurn(
                user=user,
                expected_reply=agent.strip(),
                expected_recommendations=recommendations,
                expected_end=end,
            )
        )
    return turns


def _extract_blockquote_after_heading(chunk: str, heading: str) -> str:
    pattern = rf"\*\*{re.escape(heading)}\*\*\s*(?P<body>.*?)(?:\n\s*\*\*|$)"
    match = re.search(pattern, chunk, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    lines = []
    for line in match.group("body").splitlines():
        line = line.strip()
        if line.startswith(">"):
            lines.append(line.lstrip("> ").strip())
        elif lines and line:
            break
    return "\n".join(lines).strip()


def _extract_agent_section(chunk: str) -> str:
    match = re.search(r"\*\*Agent\*\*\s*(?P<body>.*)$", chunk, flags=re.DOTALL)
    return match.group("body") if match else ""


def _extract_recommendation_names(agent_text: str) -> list[str]:
    names = []
    for line in agent_text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] in {"#", "---"} or cells[1].lower() == "name":
            continue
        if cells[0].isdigit() and cells[1]:
            names.append(_strip_markdown(cells[1]))
    return names


def _strip_markdown(value: str) -> str:
    value = re.sub(r"<([^>]+)>", r"\1", value)
    value = re.sub(r"[_*`]", "", value)
    return value.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay public SHL traces.")
    parser.add_argument("--traces-dir", type=Path, default=DEFAULT_TRACES_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    report = asyncio.run(replay_traces(load_traces(args.traces_dir)))
    write_report(report, args.report_path)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
