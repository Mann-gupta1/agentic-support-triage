"""LangGraph triage agent.

    classify -> [gate] -> retrieve -> draft -> grade -> [gate] -> respond | escalate

Both gates are the point of the project. A router that is 92% accurate is also
8% confidently wrong, and the only thing standing between that and a customer is
whether the graph knows when to stop. Every escalation records why.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, StateGraph

from .config import GRADER_SCORE_THRESHOLD, OPENAI, ROUTER_CONFIDENCE_THRESHOLD
from .data import label_to_text
from .grader import grade
from .rag import retrieve
from .router import Router


class TriageState(TypedDict, total=False):
    message: str
    intent: str
    intent_id: int
    confidence: float
    context: list[dict[str, str]]
    draft: str
    score: float
    reason: str
    outcome: Literal["answered", "escalated"]
    escalation_reason: str
    trace: Annotated[list[str], lambda a, b: a + b]


def build_graph(router: Router):
    """Compile the graph around a given router, so the same graph can be run
    over each eval arm without changing a line of orchestration."""

    def classify(state: TriageState) -> TriageState:
        intent_id, confidence = router.predict(state["message"])
        return {
            "intent_id": intent_id,
            "intent": label_to_text(intent_id),
            "confidence": confidence,
            "trace": [f"classify: {label_to_text(intent_id)} @ {confidence:.2f}"],
        }

    def gate_confidence(state: TriageState) -> Literal["retrieve", "escalate"]:
        if state["confidence"] < ROUTER_CONFIDENCE_THRESHOLD:
            return "escalate"
        return "retrieve"

    def retrieve_policy(state: TriageState) -> TriageState:
        hits = retrieve(f"{state['intent']} {state['message']}", k=3)
        sources = ", ".join(sorted({h["source"] for h in hits})) or "none"
        return {"context": hits, "trace": [f"retrieve: {len(hits)} chunks from {sources}"]}

    def draft(state: TriageState) -> TriageState:
        text = _draft_reply(state["message"], state["intent"], state["context"])
        return {"draft": text, "trace": [f"draft: {len(text)} chars"]}

    def grade_draft(state: TriageState) -> TriageState:
        score, reason = grade(state["draft"], state["context"])
        return {"score": score, "reason": reason, "trace": [f"grade: {score:.2f}"]}

    def gate_grade(state: TriageState) -> Literal["respond", "escalate"]:
        if state["score"] < GRADER_SCORE_THRESHOLD:
            return "escalate"
        return "respond"

    def respond(state: TriageState) -> TriageState:
        return {"outcome": "answered", "trace": ["respond: sent to customer"]}

    def escalate(state: TriageState) -> TriageState:
        if state.get("confidence", 1.0) < ROUTER_CONFIDENCE_THRESHOLD:
            why = (
                f"router confidence {state.get('confidence', 0):.2f} below "
                f"{ROUTER_CONFIDENCE_THRESHOLD}"
            )
        else:
            why = (
                f"grounding score {state.get('score', 0):.2f} below "
                f"{GRADER_SCORE_THRESHOLD}: {state.get('reason', '')}"
            )
        return {
            "outcome": "escalated",
            "escalation_reason": why,
            "trace": [f"escalate: {why}"],
        }

    g = StateGraph(TriageState)
    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve_policy)
    g.add_node("draft", draft)
    g.add_node("grade", grade_draft)
    g.add_node("respond", respond)
    g.add_node("escalate", escalate)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", gate_confidence,
                            {"retrieve": "retrieve", "escalate": "escalate"})
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "grade")
    g.add_conditional_edges("grade", gate_grade,
                            {"respond": "respond", "escalate": "escalate"})
    g.add_edge("respond", END)
    g.add_edge("escalate", END)
    return g.compile()


def _draft_reply(message: str, intent: str, context: list[dict[str, str]]) -> str:
    """LLM when configured; otherwise quote the top policy chunk verbatim.

    The offline path quotes rather than paraphrases on purpose - a template that
    invents wording would score well on the lexical grader for the wrong reason
    and make the whole harness lie to you.
    """
    if not context:
        return "I could not find a policy covering this. Passing you to a colleague."
    if not OPENAI.is_configured:
        top = context[0]
        return (
            f"On {intent}: {top['text']} "
            f"(source: {top['source']} policy)"
        )
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI.api_key, base_url=OPENAI.base_url)
    policy = "\n\n".join(f"[{c['source']}] {c['text']}" for c in context)
    resp = client.chat.completions.create(
        model=OPENAI.model,
        messages=[{
            "role": "user",
            "content": (
                "Answer the customer using only the policy below. If the policy "
                "does not cover it, say so instead of guessing. Two sentences.\n\n"
                f"POLICY:\n{policy}\n\nCUSTOMER ({intent}): {message}"
            ),
        }],
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()
