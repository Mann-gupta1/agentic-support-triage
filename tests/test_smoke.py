"""Offline smoke tests. No API key, no GPU, no fine-tuned adapter required.

These exist to catch the failures that silently produce a believable but wrong
report: a gate that never fires, a grader that passes ungrounded text, a
retriever that returns nothing.
"""
from __future__ import annotations

from src.grader import lexical_grade
from src.rag import retrieve


def test_retrieval_returns_grounded_chunks():
    hits = retrieve("my card has not arrived yet", k=3)
    assert hits, "retriever returned nothing"
    assert all(h["text"] and h["source"] for h in hits)


def test_lexical_grader_rewards_grounded_text():
    context = retrieve("how long does a card take to arrive", k=2)
    grounded = context[0]["text"]
    invented = "Your card will arrive in exactly 42 minutes by helicopter courier."
    grounded_score, _ = lexical_grade(grounded, context)
    invented_score, _ = lexical_grade(invented, context)
    assert grounded_score > invented_score, "grader cannot tell invented from grounded"


def test_lexical_grader_floors_empty_reply():
    score, reason = lexical_grade("", retrieve("card", k=1))
    assert score == 0.0 and "empty" in reason


def test_confidence_gate_escalates_on_low_confidence():
    from src.graph import build_graph
    from src.router import Router

    class AlwaysUnsure(Router):
        name = "stub-unsure"

        def predict(self, text: str) -> tuple[int, float]:
            return 0, 0.01

    out = build_graph(AlwaysUnsure()).invoke({"message": "where is my card"})
    assert out["outcome"] == "escalated"
    assert "confidence" in out["escalation_reason"]


def test_confident_router_reaches_a_reply():
    from src.graph import build_graph
    from src.router import Router

    class AlwaysSure(Router):
        name = "stub-sure"

        def predict(self, text: str) -> tuple[int, float]:
            return 0, 0.99

    out = build_graph(AlwaysSure()).invoke({"message": "where is my card"})
    assert out["outcome"] in {"answered", "escalated"}
    assert out["draft"], "no reply was drafted"
    assert out["trace"][0].startswith("classify")
