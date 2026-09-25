"""Single source of truth for paths, model ids, and thresholds."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
POLICY_DIR = ROOT / "data" / "policies"
CHROMA_DIR = ROOT / ".chroma"

# banking77: 77 intent labels over short customer-support utterances.
# The mteb mirror is parquet-native. The original PolyAI/banking77 ships a
# loading script, which datasets>=3 refuses to execute.
DATASET_ID = "mteb/banking77"
NUM_LABELS = 77

# Small enough to LoRA-tune on a free Colab T4 or an M-series Mac via MPS.
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B")

# Confidence gate: below this, the graph escalates to a human instead of replying.
# Tune it from the reliability curve that evaluate.py prints - do not guess.
ROUTER_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.65"))
GRADER_SCORE_THRESHOLD = float(os.getenv("GRADER_SCORE_THRESHOLD", "0.6"))

MAX_SEQ_LEN = 64
SEED = 42


@dataclass(frozen=True)
class LoraSettings:
    r: int = int(os.getenv("LORA_R", "16"))
    alpha: int = int(os.getenv("LORA_ALPHA", "32"))
    dropout: float = 0.05
    learning_rate: float = float(os.getenv("LORA_LR", "2e-4"))
    epochs: float = float(os.getenv("LORA_EPOCHS", "3"))
    batch_size: int = int(os.getenv("LORA_BATCH", "32"))


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


LORA = LoraSettings()
OPENAI = OpenAISettings()

for _d in (ARTIFACTS, REPORTS, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)
