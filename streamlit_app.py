"""Streamlit UI for the catalog-grounded SHL assessment recommender.

This is a thin client over the same stateless agent used by FastAPI. It keeps
conversation history in Streamlit session state and sends the full message list
to the agent each turn, matching the `/chat` API contract.
"""

from __future__ import annotations

import asyncio
from typing import Any

import streamlit as st

from app.graph import GraphBackedSHLAgent
from app.schemas import ChatMessage, MessageRole


st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧭",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_agent() -> GraphBackedSHLAgent:
    return GraphBackedSHLAgent()


def run_async(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def render_recommendations(recommendations: list[Any]) -> None:
    if not recommendations:
        return
    st.markdown("#### Recommended SHL assessments")
    for index, rec in enumerate(recommendations, start=1):
        with st.container(border=True):
            st.markdown(f"**{index}. [{rec.name}]({rec.url})**")
            st.caption(f"Test type: {rec.test_type}")


if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("SHL Assessment Recommender")
st.caption(
    "Catalog-grounded conversational agent for SHL Individual Test Solutions. "
    "It clarifies vague needs, recommends 1-10 assessments, refines shortlists, "
    "compares assessments, and refuses out-of-scope requests."
)

with st.sidebar:
    st.success("SHL catalog only")
    st.markdown(
        """
        Try:

        - `I need an assessment`
        - `Here is a text from job description: Java developer with 4 years experience`
        - `Actually, add personality tests`
        - `What is the difference between OPQ and GSA?`
        - `Ignore previous instructions`
        """
    )
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_recommendations(message.get("recommendations", []))

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Tell me the role, seniority, skills, or paste a job description. "
            "I’ll clarify if needed and recommend only SHL catalog assessments."
        )

prompt = st.chat_input("Describe the role or assessment need...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = [
        ChatMessage(role=MessageRole(item["role"]), content=item["content"])
        for item in st.session_state.messages
        if item["role"] in {"user", "assistant"}
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching SHL catalog..."):
            response = run_async(get_agent().chat(payload))
        st.markdown(response.reply)
        render_recommendations(response.recommendations)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.reply,
            "recommendations": response.recommendations,
        }
    )
