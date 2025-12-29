"""Pytest suite for the LangGraph reflection workflow.

Tests include:
- Graph structure (nodes and edges)
- Conditional routing via `should_continue`
- `writer_node` behavior with a stubbed LLM and stream writer
- `reviewer_node` behavior (normal approval and max-revisions branch)
"""

from types import SimpleNamespace

import pytest

from graph_simple import create_reflection_graph, should_continue
import nodes as nodes_module
from state import NodeName, Decision
from config import MAX_REVISIONS, MAX_REVISIONS_MESSAGE


class DummyLLM:
    """A lightweight stub of ChatOpenAI used for testing."""

    def __init__(self, model=None, temperature=None):
        self.model = model
        self.temperature = temperature

    def invoke(self, payload):
        # Return an object with a .content attribute like the real LLM response
        return SimpleNamespace(content="dummy response content")

    def with_structured_output(self, schema):
        # Return a simple object whose invoke(...) returns a dict-like decision
        class Structured:
            def invoke(self_inner, prompt):
                return {"decision": Decision.APPROVE.value}

        return Structured()


def test_should_continue_true_and_false():
    assert should_continue({"continue_revision": True}) == NodeName.WRITER.value
    assert should_continue({"continue_revision": False}) == NodeName.PUBLISHER.value
    # default when not provided
    assert should_continue({}) == NodeName.PUBLISHER.value


def test_create_reflection_graph_nodes_and_edges():
    g = create_reflection_graph()

    assert NodeName.WRITER.value in g.nodes
    assert NodeName.REVIEWER.value in g.nodes
    assert NodeName.PUBLISHER.value in g.nodes

    # Ensure there is an entry from START into the writer node
    from langgraph.graph import START

    assert START in g.edges
    # edges[START] could be a list or mapping depending on implementation; be permissive
    assert any(
        (NodeName.WRITER.value == e or (isinstance(e, (list, tuple)) and NodeName.WRITER.value in e))
        for e in g.edges[START]
    )


def test_writer_node_invokes_llm_and_updates_state(monkeypatch):
    # Capture custom events emitted by the stream writer
    emitted = []

    def fake_stream_writer(data):
        emitted.append(data)

    monkeypatch.setattr(nodes_module, "get_stream_writer", lambda: fake_stream_writer)
    monkeypatch.setattr(nodes_module, "ChatOpenAI", DummyLLM)

    initial_state = {
        "messages": [],
        "original_customer_message": "Hello!",
        "revision_count": 0,
        "latest_feedback_for_writer": "",
        "latest_message_response_by_writer": "",
        "latest_reviewer_decision": Decision.REVISE.value,
    }

    result = nodes_module.writer_node(initial_state, SimpleNamespace())

    assert result["revision_count"] == 1
    assert result["latest_message_response_by_writer"] == "dummy response content"
    # messages should contain an AIMessage with the writer name
    msg = result["messages"][0]
    assert getattr(msg, "content", None) == "dummy response content"
    assert getattr(msg, "name", None) == NodeName.WRITER.value
    # ensure stream writer was called
    assert any(isinstance(x, dict) for x in emitted)


def test_reviewer_node_approves(monkeypatch):
    # LLM returns feedback and structured output says APPROVE
    monkeypatch.setattr(nodes_module, "ChatOpenAI", DummyLLM)

    state = {
        "messages": [],
        "original_customer_message": "Original",
        "revision_count": 1,
        "latest_message_response_by_writer": "Some reply",
    }

    result = nodes_module.reviewer_node(state)

    assert result["latest_reviewer_decision"] == Decision.APPROVE.value
    assert result["continue_revision"] is False
    assert "latest_feedback_for_writer" in result


def test_reviewer_node_max_revisions_triggers_approval():
    # When revision_count >= MAX_REVISIONS, reviewer auto-approves with a max-revision message
    state = {
        "messages": [],
        "original_customer_message": "Original",
        "revision_count": MAX_REVISIONS,
        "latest_message_response_by_writer": "Some reply",
    }

    result = nodes_module.reviewer_node(state)

    assert result["latest_reviewer_decision"] == Decision.APPROVE.value
    assert result["continue_revision"] is False
    # Check the MAX_REVISIONS_MESSAGE appears in the reviewer messages
    assert any(MAX_REVISIONS_MESSAGE in getattr(m, "content", "") for m in result["messages"]) 
