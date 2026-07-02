"""LangGraph workflow construction."""

from __future__ import annotations

from typing import Any

from app.agent import (
    ConversationalSHLAgent,
    latest_user_text,
    _apply_role_focus,
    _test_type,
)
from app.retrieval import build_query_text
from app.safety import REFUSAL_REPLY
from app.schemas import ChatMessage, ChatResponse, Recommendation
from app.state import AgentState, Intent, RetrievalQuery


class GraphBackedSHLAgent:
    """Recommender adapter that exposes `chat()` through the graph workflow."""

    def __init__(self, agent: ConversationalSHLAgent | None = None) -> None:
        self.agent = agent or ConversationalSHLAgent()
        self.graph = build_graph(self.agent)

    async def chat(self, messages: list[ChatMessage] | tuple[ChatMessage, ...]) -> ChatResponse:
        state = await self.graph.ainvoke({"messages": list(messages)})
        return ChatResponse(
            reply=state["reply"],
            recommendations=state.get("recommendations", []),
            end_of_conversation=state.get("end_of_conversation", False),
        )


def build_graph(agent: ConversationalSHLAgent | None = None) -> Any:
    """Build the stateless graph-based RAG workflow."""

    runtime_agent = agent or ConversationalSHLAgent()
    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        return _FallbackGraph(runtime_agent)

    async def safety_node(state: AgentState) -> AgentState:
        safety = await runtime_agent.safety.inspect(state["messages"])
        return {**state, "safety": safety}

    async def parse_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        requirements = await runtime_agent.parser.parse(messages)
        return {
            **state,
            "requirements": requirements,
            "turn_count": sum(1 for message in messages if message.role == "user"),
            "clarification_count": sum(
                1 for message in messages if message.role == "assistant" and "?" in message.content
            ),
        }

    async def route_node(state: AgentState) -> AgentState:
        if not state["safety"].allowed:
            return {**state, "intent": Intent.REFUSE}
        intent = await runtime_agent.router.route(state["messages"], state["requirements"])
        return {**state, "intent": intent}

    async def refuse_node(state: AgentState) -> AgentState:
        return {
            **state,
            "reply": REFUSAL_REPLY,
            "recommendations": [],
            "end_of_conversation": False,
        }

    async def compare_node(state: AgentState) -> AgentState:
        response = runtime_agent._compare(state["messages"])
        return {
            **state,
            "reply": response.reply,
            "recommendations": response.recommendations,
            "end_of_conversation": response.end_of_conversation,
        }

    async def clarify_node(state: AgentState) -> AgentState:
        question = runtime_agent.clarifier.next_question(
            state["requirements"],
            state.get("clarification_count", 0),
            state.get("turn_count", 0),
        )
        if question and state["intent"] != Intent.REFINE:
            return {
                **state,
                "intent": Intent.CLARIFY,
                "reply": question,
                "recommendations": [],
                "end_of_conversation": False,
            }
        return state

    async def retrieve_node(state: AgentState) -> AgentState:
        requirements = state["requirements"]
        candidates = await runtime_agent.retriever.retrieve(
            RetrievalQuery(
                text=build_query_text(requirements, latest_user_text(state["messages"])),
                requirements=requirements,
                limit=20,
            )
        )
        return {**state, "candidates": candidates}

    async def rank_node(state: AgentState) -> AgentState:
        ranked = runtime_agent.ranker.rank(
            state["requirements"], state.get("candidates", []), limit=10
        )
        ranked = _apply_role_focus(state["requirements"], ranked)
        return {**state, "ranked": ranked}

    async def respond_node(state: AgentState) -> AgentState:
        ranked = state.get("ranked", [])
        recommendations = [
            Recommendation(
                name=item.assessment.name,
                url=item.assessment.url,
                test_type=_test_type(item.assessment),
            )
            for item in ranked
        ]
        if not recommendations:
            return {
                **state,
                "reply": (
                    "I could not find a strong SHL catalog match for that request. "
                    "Please share the role, seniority, and key skills."
                ),
                "recommendations": [],
                "end_of_conversation": False,
            }
        role = state["requirements"].role
        role_phrase = f" for {role}" if role else ""
        return {
            **state,
            "reply": (
                f"Here are the strongest SHL assessment matches{role_phrase}. "
                "I used the catalog fields plus your stated skills, seniority, and trait requirements."
            ),
            "recommendations": recommendations,
            "end_of_conversation": False,
        }

    def after_route(state: AgentState) -> str:
        if state["intent"] == Intent.REFUSE:
            return "refuse"
        if state["intent"] == Intent.COMPARE:
            return "compare"
        return "clarify"

    def after_clarify(state: AgentState) -> str:
        if state.get("reply") and state["intent"] == Intent.CLARIFY:
            return "end"
        return "retrieve"

    graph = StateGraph(AgentState)
    graph.add_node("safety", safety_node)
    graph.add_node("parse", parse_node)
    graph.add_node("route", route_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("compare", compare_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rank", rank_node)
    graph.add_node("respond", respond_node)
    graph.set_entry_point("safety")
    graph.add_edge("safety", "parse")
    graph.add_edge("parse", "route")
    graph.add_conditional_edges(
        "route",
        after_route,
        {"refuse": "refuse", "compare": "compare", "clarify": "clarify"},
    )
    graph.add_conditional_edges(
        "clarify",
        after_clarify,
        {"end": END, "retrieve": "retrieve"},
    )
    graph.add_edge("retrieve", "rank")
    graph.add_edge("rank", "respond")
    graph.add_edge("respond", END)
    graph.add_edge("refuse", END)
    graph.add_edge("compare", END)
    return graph.compile()


class _FallbackGraph:
    def __init__(self, agent: ConversationalSHLAgent) -> None:
        self.agent = agent

    async def ainvoke(self, state: AgentState) -> AgentState:
        response = await self.agent.chat(state["messages"])
        return {
            **state,
            "reply": response.reply,
            "recommendations": response.recommendations,
            "end_of_conversation": response.end_of_conversation,
        }
