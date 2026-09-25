"""Three interchangeable intent routers, so the eval compares like with like.

Every router returns (label_id, confidence). Confidence is what the graph's gate
reads, so a router that cannot express uncertainty is useless here no matter how
accurate it is - that is the point the evaluation is built to show.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .config import ARTIFACTS, BASE_MODEL, MAX_SEQ_LEN, OPENAI
from .data import label_names, label_to_text, texts_and_labels


class Router(ABC):
    name: str

    @abstractmethod
    def predict(self, text: str) -> tuple[int, float]:
        """Return (label_id, confidence in [0, 1])."""

    def predict_batch(self, texts: list[str]) -> list[tuple[int, float]]:
        return [self.predict(t) for t in texts]


class TfidfRouter(Router):
    """Classical-ML floor: TF-IDF + logistic regression. Trains in seconds on CPU.

    If an expensive LLM cannot beat this, the LLM is not earning its cost - which
    is exactly the kind of call this project exists to make with a number.
    """

    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        train_x, train_y = texts_and_labels("train")
        self._pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=4.0),
        )
        self._pipe.fit(train_x, train_y)

    def predict(self, text: str) -> tuple[int, float]:
        probs = self._pipe.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        return idx, float(probs[idx])


class LlmZeroShotRouter(Router):
    """Hosted-LLM arm. Needs OPENAI_API_KEY; skipped by the eval when absent.

    77 labels do not fit comfortably in a prompt, so this shortlists with TF-IDF
    and asks the model to choose among the top-k. That is the honest version of
    "zero-shot" at this label count, and the shortlist is reported alongside the
    accuracy so the comparison is not silently doing the model's work for it.
    """

    name = "llm-zeroshot"

    def __init__(self, shortlist_k: int = 10) -> None:
        if not OPENAI.is_configured:
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI

        self._client = OpenAI(api_key=OPENAI.api_key, base_url=OPENAI.base_url)
        self._k = shortlist_k
        self._shortlister = TfidfRouter()
        self._names = label_names()

    def _shortlist(self, text: str) -> list[int]:
        probs = self._shortlister._pipe.predict_proba([text])[0]
        return list(np.argsort(probs)[::-1][: self._k])

    def predict(self, text: str) -> tuple[int, float]:
        candidates = self._shortlist(text)
        menu = "\n".join(f"{i}. {label_to_text(c)}" for i, c in enumerate(candidates))
        prompt = (
            "Classify the customer message into exactly one intent.\n"
            f"Message: {text!r}\n\nIntents:\n{menu}\n\n"
            'Reply with JSON only: {"index": <int>, "confidence": <float 0-1>}'
        )
        resp = self._client.chat.completions.create(
            model=OPENAI.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
            choice = int(parsed["index"])
            conf = float(parsed.get("confidence", 0.5))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            # A malformed reply is a real failure mode, not an exception to hide.
            # Fall back to the shortlist's own top pick at low confidence.
            return candidates[0], 0.0
        if not 0 <= choice < len(candidates):
            return candidates[0], 0.0
        return int(candidates[choice]), max(0.0, min(1.0, conf))


class LoraRouter(Router):
    """LoRA-tuned classification head over a small causal LM. Requires finetune.py
    to have been run first - it deliberately does not silently fall back."""

    name = "lora"

    def __init__(self, adapter_dir: Path | None = None) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._dir = adapter_dir or (ARTIFACTS / "lora-router")
        if not (self._dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No LoRA adapter at {self._dir}. Run: python -m src.finetune"
            )
        self._torch = torch
        self._device = _pick_device(torch)
        self._tok = AutoTokenizer.from_pretrained(self._dir)
        base = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=len(label_names())
        )
        base.config.pad_token_id = self._tok.pad_token_id
        self._model = PeftModel.from_pretrained(base, self._dir).to(self._device).eval()

    def predict(self, text: str) -> tuple[int, float]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[tuple[int, float]]:
        torch = self._torch
        out: list[tuple[int, float]] = []
        for start in range(0, len(texts), 32):
            chunk = texts[start : start + 32]
            enc = self._tok(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding=True,
            ).to(self._device)
            with torch.no_grad():
                probs = torch.softmax(self._model(**enc).logits, dim=-1)
            conf, idx = torch.max(probs, dim=-1)
            out.extend(zip(idx.tolist(), conf.tolist()))
        return out


def _pick_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


ROUTERS: dict[str, type[Router]] = {
    "tfidf": TfidfRouter,
    "llm-zeroshot": LlmZeroShotRouter,
    "lora": LoraRouter,
}
