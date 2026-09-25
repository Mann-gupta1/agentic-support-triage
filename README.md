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

Real numbers from `reports/eval-2026-09-25T06-58-32Z.md`, produced by
`python -m src.evaluate`. Nothing below is hand-written — regenerate it and the
file in `reports/` is the source of truth. 500 sampled test items, all 77
intents present.

| Router | Accuracy | Macro-F1 | p50 ms | p95 ms | Mean conf | Mean conf when wrong |
|---|---|---|---|---|---|---|
| `tfidf` (TF-IDF + logistic regression) | 0.906 | 0.896 | 1.6 | 2.5 | 0.65 | 0.29 |
| `lora` (Qwen2.5-0.5B, LoRA, 3 epochs) | **0.924** | **0.920** | 53.9 | 75.7 | 0.95 | 0.70 |

On the headline the fine-tune looks barely worth it: +1.8pp accuracy for 34x the
latency. **That headline is misleading, and the selective-prediction sweep is why
the harness exists.**

| Router | Threshold | Coverage (auto-answered) | Accuracy on accepted |
|---|---|---|---|
| `tfidf` | 0.50 | 70.6% | 0.986 |
| `lora` | 0.90 | **85.6%** | 0.981 |

Held at the same quality bar — about 98% accuracy on whatever it chooses to
answer — **the tuned router automates 85.6% of traffic against the baseline's
70.6%.** That is 15 points more automation, and none of it is visible in the
accuracy column. The fine-tune's real contribution is not being right more often;
it is being right more often *on the cases it is confident about*, which is the
only thing a gated system can actually spend.

**The thresholds are not comparable across routers, and that is the catch.** The
tuned model is badly calibrated in absolute terms: mean confidence 0.95, and
still 0.70 when it is wrong, against 0.29 for TF-IDF. It is more accurate and
less honest about being wrong. So the gate has to be set per router from its own
curve — 0.50 for TF-IDF, 0.90 for LoRA — and a threshold copied from one to the
other silently destroys the gate. `ROUTER_CONFIDENCE_THRESHOLD` in
`src/config.py` is a single value on purpose, so swapping the router without
re-reading this table is a decision you have to make, not one you can drift into.

**Is 34x latency worth 15pp more automation?** At 53.9 ms p50 the router is still
far under what a support reply needs, and the drafting call dominates the request
anyway. So yes here — but that is a judgement about this workload at this scale,
not a general fact, and it flips the moment the router sits on a hot path.

**What the first run got wrong.** A one-epoch smoke run scored 0.848 — *worse*
than the TF-IDF baseline — and briefly looked like a clean "the cheap option
wins" finding. It was under-training, not a result. Three epochs moved it to
0.924. Checking that before writing it up is the difference between a finding and
a story.

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
LORA_EPOCHS=1 LORA_FP16=0 python -m src.finetune   # smoke run, ~6 min on a T4
LORA_FP16=0 python -m src.finetune                # full run, ~24 min on a T4
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

**Why the fine-tune runs on Colab and not locally.** Measured, not assumed. On
an M-series Mac at fp32 on MPS a single training step took 74-166 seconds; at 846
steps that is 17 to 39 hours. MPS has no usable mixed precision for this model,
so there is no fp16 path to rescue it. The same run on a Colab T4 took **23m43s**
at ~1.2 s/step — between 55x and 124x faster.

**Why the T4 run is fp32 too.** fp16 was the plan and it does not work on this
stack: on torch 2.11.0+cu128 with a T4, the first optimizer step dies inside the
GradScaler's unscale with `NotImplementedError` from
`_amp_foreach_non_finite_check_and_unscale_`. Rather than pin a torch version to
chase it, `LORA_FP16=0` falls back to fp32, which finishes in 24 minutes. A slow
run beats a broken one, and `src/finetune.py` prints the device, the torch
version and the fp16 decision on startup so this is visible in one line instead
of inferred from a stack trace.

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
