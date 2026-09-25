"""Scores a drafted reply for groundedness in the retrieved policy.

Two graders on purpose. The LLM grader is the one you would ship; the lexical
grader is what lets the whole graph run offline and in CI, and it keeps the
LLM grader honest - if they disagree wildly on a case, one of them is wrong and
that is worth knowing before production is.
"""
from __future__ import annotations

import json
import re

from .config import OPENAI

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def lexical_grade(reply: str, context: list[dict[str, str]]) -> tuple[float, str]:
    """Fraction of the reply's content words that appear in the retrieved policy.

    Crude by design: it catches a reply that invented a number or a timeframe,
    which is the failure that actually costs money here.
    """
    reply_tokens = {t for t in _tokens(reply) if len(t) > 3}
    if not reply_tokens:
        return 0.0, "empty reply"
    context_tokens = set()
    for chunk in context:
        context_tokens |= _tokens(chunk["text"])
    overlap = len(reply_tokens & context_tokens) / len(reply_tokens)
    return overlap, f"lexical overlap with policy: {overlap:.2f}"


def llm_grade(reply: str, context: list[dict[str, str]]) -> tuple[float, str]:
    """LLM judge for groundedness. Raises if no API key is configured."""
    if not OPENAI.is_configured:
        raise RuntimeError("OPENAI_API_KEY is not set")
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI.api_key, base_url=OPENAI.base_url)
    policy = "\n\n".join(f"[{c['source']}] {c['text']}" for c in context)
    prompt = (
        "You are grading a support reply for groundedness, not for tone.\n"
        "Score 1.0 only if every factual claim in the reply is supported by the "
        "policy below. Score 0.0 if the reply states a fact the policy does not.\n\n"
        f"POLICY:\n{policy}\n\nREPLY:\n{reply}\n\n"
        'Reply with JSON only: {"score": <float 0-1>, "reason": "<one sentence>"}'
    )
    resp = client.chat.completions.create(
        model=OPENAI.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(resp.choices[0].message.content or "{}")
        return float(parsed["score"]), str(parsed.get("reason", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        # An unparseable grade is not a passing grade.
        return 0.0, f"grader returned malformed output: {exc}"


def grade(reply: str, context: list[dict[str, str]]) -> tuple[float, str]:
    """LLM grader when a key is configured, lexical grader otherwise."""
    if OPENAI.is_configured:
        return llm_grade(reply, context)
    return lexical_grade(reply, context)
