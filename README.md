# Agentic support triage — a confidence-gated LangGraph agent with a LoRA-tuned router

A customer-support triage agent that classifies an incoming message, retrieves the
governing policy, drafts a grounded reply, and **escalates to a human whenever it is
not sure enough** — at either of two gates.

The interesting question here is not "how accurate is the classifier". It is
**"how accurate is it on the traffic it chooses to answer, and how much traffic does
that leave for a human"**. A router that is 90% accurate is also 10% confidently
wrong, and the only thing between that and a customer is whether the system knows
when to stop. Everything in this repo is built to put a number on that trade.

```
message
   │
   ▼
classify ──(confidence < threshold)──────────────┐
   │                                             │
   │ (confident)                                 │
   ▼                                             │
retrieve policy ──▶ draft reply ──▶ grade        │
                                     │           │
                    (ungrounded) ────┼───────────┤
                                     │           ▼
                             (grounded)      escalate
                                     │      (with reason)
                                     ▼
                                  respond
```

## Results

Real numbers from `reports/`, produced by `python -m src.evaluate`. Nothing below
is hand-written — regenerate it and the file in `reports/` is the source of truth.

**Classical-ML baseline** (TF-IDF + logistic regression, 500 sampled test items,
all 77 intents present):

| Router | Accuracy | Macro-F1 | p50 ms | p95 ms | Mean conf | Mean conf when wrong |
|---|---|---|---|---|---|---|
| `tfidf` | 0.906 | 0.896 | 1.2 | 5.2 | 0.65 | 0.29 |

The gap between mean confidence overall (0.65) and mean confidence when wrong
(0.29) is what makes the gate work at all — the router's uncertainty is
informative, not noise. That shows up directly as selective accuracy:

| Threshold | Coverage (auto-answered) | Accuracy on accepted |
|---|---|---|
| 0.00 | 100.0% | 0.906 |
| 0.30 | 86.2% | 0.954 |
| 0.50 | 70.6% | 0.986 |
| 0.65 | 56.2% | 0.989 |
| 0.80 | 38.6% | 1.000 |

Read the 0.50 row: **70.6% of traffic answered without a human, at 98.6% accuracy
on what it answered.** Raising the gate to 0.80 buys perfect accuracy and costs
more than half the automation. That is the actual product decision, and it is a
number, not a preference.

**LoRA-tuned arm:** not yet run. Open `notebooks/colab_finetune.ipynb` on a
Colab T4, then `python -m src.evaluate --arms tfidf,lora` fills it in. Until
then this table stays as it is — the repo does not print a tuned number it has
not measured.

**LLM zero-shot arm:** requires `OPENAI_API_KEY`. Skipped automatically when absent.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# offline: no API key, no GPU, no adapter needed
pytest -q
python -m src.evaluate --arms tfidf --limit 500

# fine-tune the router -> use Colab, see notebooks/colab_finetune.ipynb
# (a local MPS run is measured below and is not viable)
LORA_EPOCHS=1 python -m src.finetune     # smoke run
python -m src.finetune                   # full run
python -m src.evaluate --arms tfidf,lora --limit 500

# add the hosted-LLM arm and the LLM drafter/grader
cp .env.example .env    # set OPENAI_API_KEY
python -m src.evaluate --arms tfidf,llm-zeroshot,lora
```

Run the agent end to end:

```python
from src.graph import build_graph
from src.router import TfidfRouter

out = build_graph(TfidfRouter()).invoke({"message": "my card still has not arrived"})
print(out["outcome"], "|", out.get("escalation_reason", ""))
print("\n".join(out["trace"]))
```

## What is in here

| File | What it does |
|---|---|
| `src/router.py` | Three interchangeable routers behind one interface: TF-IDF, hosted-LLM few-shot, LoRA-tuned. Each returns `(label, confidence)`. |
| `src/finetune.py` | LoRA/PEFT fine-tune of a small causal LM (`Qwen2.5-0.5B`) as a 77-way classifier. |
| `src/graph.py` | The LangGraph state machine, both gates, and the escalation reasons. |
| `src/grader.py` | Groundedness grading. LLM judge when a key is set, lexical overlap otherwise so the graph runs offline and in CI. |
| `src/rag.py` | Chroma retrieval over `data/policies/*.md`. |
| `src/evaluate.py` | The harness. Accuracy, macro-F1, latency, and the selective-prediction sweep. Writes `reports/`. |
| `notebooks/colab_finetune.ipynb` | The training path that actually works: clone, install, smoke run, full run, evaluate, download the adapter. |
| `tests/test_smoke.py` | Offline tests: retrieval returns something, the grader can tell invented from grounded, and both gates actually fire. |

## Decisions worth defending

**Why a classical baseline at all.** TF-IDF plus logistic regression trains in
seconds on CPU and scores 0.906 here. Any LLM arm has to beat that on accuracy
*and* justify its latency and cost, or it is not earning its place. Shipping the
expensive option without measuring the cheap one first is how AI features turn
into bills.

**Why two gates, not one.** They catch different failures. The confidence gate
catches "I do not know what this is about". The grounding gate catches "I know
what it is about and I just invented the answer". A single gate on either signal
misses the other class entirely.

**Why the grader has an offline path.** A grader that only works with an API key
cannot run in CI, which means it stops running. The lexical grader is crude — it
measures token overlap with the retrieved policy — but it catches a reply that
invented a timeframe or a fee, which is the failure that costs money here. When
the two graders disagree sharply on a case, one of them is wrong, and finding
that out before production is the point.

**Why sampled, not head-sliced, evaluation.** The banking77 test split is grouped
by label. An early version of the harness took the first N items and reported
0.890 accuracy against a 0.319 macro-F1 — the sample only covered a fraction of
the 77 intents. `--limit` now draws a seeded random sample and the report prints
how many distinct intents it actually contains.

**Why the fine-tune runs on Colab and not locally.** Measured, not assumed: on
an M-series Mac at fp32 on MPS, a single training step took 74-166 seconds. At
846 steps that is 17 to 39 hours. MPS has no usable mixed precision for this
model, so there is no fp16 path to rescue it. On a Colab T4 with fp16 the same
run is roughly 10-15 minutes. `src/finetune.py` turns on fp16 and pinned memory
only when CUDA is present, so the same file is correct on both.

**Why `mteb/banking77`.** The original `PolyAI/banking77` ships a loading script,
which `datasets>=3` refuses to execute. The mteb mirror is parquet-native. It
stores `label` as a plain int rather than a `ClassLabel`, so label names are
derived from the paired `label_text` column.

## Known limits

- The policy corpus is four short documents. Retrieval quality is not what this
  project is measuring, and the corpus is deliberately small enough to read.
- The lexical grader rewards quoting. The offline drafter quotes policy verbatim
  rather than paraphrasing, precisely so the harness is not grading its own
  paraphrase and reporting a flattering number.
- Latency for the `lora` arm is measured single-item on whatever device is
  available. It is not a throughput benchmark.
- Chroma can emit a `recursive_mutex` message from its telemetry thread at
  interpreter shutdown. It is a teardown warning, after results are written.

## Dataset

[banking77](https://huggingface.co/datasets/mteb/banking77) — 13k customer-support
utterances over 77 banking intents. CC-BY-4.0.

## Run it on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mann-gupta1/agentic-support-triage/blob/main/notebooks/colab_finetune.ipynb)

Set the runtime to **T4 GPU** first.
