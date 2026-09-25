"""Base-vs-tuned evaluation harness.

    python -m src.evaluate --arms tfidf,lora --limit 500

Reports accuracy and macro-F1, latency, and - the part that matters for the
graph - selective accuracy: how accurate each router is on the traffic it is
confident enough to answer, and how much traffic that leaves for a human.
A router that is more accurate overall but cannot rank its own uncertainty is
worse here, and this harness is built to make that visible rather than hide it
behind a single headline number.

Every number in reports/ comes from a real run. Nothing here is hardcoded.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import NUM_LABELS, REPORTS, ROUTER_CONFIDENCE_THRESHOLD, SEED
from .data import texts_and_labels
from .router import ROUTERS, Router

THRESHOLD_SWEEP = [0.0, 0.3, 0.5, 0.65, 0.8, 0.9]


def evaluate_router(router: Router, texts: list[str], labels: list[int]) -> dict:
    start = time.perf_counter()
    per_item: list[float] = []
    preds: list[int] = []
    confs: list[float] = []
    for text in texts:
        t0 = time.perf_counter()
        pred, conf = router.predict(text)
        per_item.append((time.perf_counter() - t0) * 1000)
        preds.append(pred)
        confs.append(conf)
    wall = time.perf_counter() - start

    preds_a, labels_a, confs_a = np.array(preds), np.array(labels), np.array(confs)
    correct = preds_a == labels_a

    selective = []
    for thr in THRESHOLD_SWEEP:
        accepted = confs_a >= thr
        coverage = float(accepted.mean())
        acc = float(correct[accepted].mean()) if accepted.any() else float("nan")
        selective.append(
            {"threshold": thr, "coverage": coverage, "accuracy_on_accepted": acc}
        )

    return {
        "router": router.name,
        "n": len(texts),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "latency_ms_p50": float(np.percentile(per_item, 50)),
        "latency_ms_p95": float(np.percentile(per_item, 95)),
        "wall_seconds": round(wall, 2),
        "mean_confidence": float(confs_a.mean()),
        "mean_confidence_when_wrong": (
            float(confs_a[~correct].mean()) if (~correct).any() else float("nan")
        ),
        "selective": selective,
    }


def _table(results: list[dict]) -> str:
    head = (
        "| Router | Accuracy | Macro-F1 | p50 ms | p95 ms | Mean conf | "
        "Mean conf when wrong |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = "".join(
        f"| `{r['router']}` | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | "
        f"{r['latency_ms_p50']:.1f} | {r['latency_ms_p95']:.1f} | "
        f"{r['mean_confidence']:.2f} | {r['mean_confidence_when_wrong']:.2f} |\n"
        for r in results
    )
    return head + rows


def _selective_table(r: dict) -> str:
    head = ("| Threshold | Coverage (auto-answered) | Accuracy on accepted |\n"
            "|---|---|---|\n")
    rows = "".join(
        f"| {s['threshold']:.2f} | {s['coverage']:.1%} | {s['accuracy_on_accepted']:.3f} |\n"
        for s in r["selective"]
    )
    return f"\n**`{r['router']}` confidence gate**\n\n" + head + rows


def write_report(results: list[dict], limit: int, n_labels: int) -> tuple[str, str]:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    md = REPORTS / f"eval-{stamp}.md"
    js = REPORTS / f"eval-{stamp}.json"

    body = [
        f"# Intent router evaluation\n",
        f"- Run (UTC): `{stamp}`",
        f"- Test items: `{limit}` sampled (seed {SEED}) from the banking77 held-out test split",
        f"- Distinct intents present in the sample: `{n_labels}` of {NUM_LABELS}",
        f"- Graph gate threshold in config: `{ROUTER_CONFIDENCE_THRESHOLD}`\n",
        "## Headline\n",
        _table(results),
        "\n## Selective prediction\n",
        "Coverage is the share of traffic the router answers without a human. "
        "Read the two together: the useful router is the one that keeps accuracy "
        "high on what it accepts while still covering enough traffic to be worth "
        "deploying.\n",
    ]
    body += [_selective_table(r) for r in results]
    md.write_text("\n".join(body))
    js.write_text(json.dumps(results, indent=2))
    return str(md), str(js)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="tfidf",
                    help=f"comma-separated, from: {','.join(ROUTERS)}")
    ap.add_argument("--limit", type=int, default=500,
                    help="test items to score (0 = full test split)")
    args = ap.parse_args()

    texts, labels = texts_and_labels("test")
    if args.limit and args.limit < len(texts):
        # Sample, never head-slice: the test split is grouped by label, so the
        # first N items cover only a handful of the 77 intents and macro-F1
        # becomes meaningless against a full-label denominator.
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(texts), size=args.limit, replace=False)
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]
    print(f"test items: {len(texts)} | distinct labels present: {len(set(labels))}")

    results = []
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if name not in ROUTERS:
            raise SystemExit(f"Unknown arm {name!r}. Choose from: {','.join(ROUTERS)}")
        print(f"building {name} ...", flush=True)
        try:
            router = ROUTERS[name]()
        except (RuntimeError, FileNotFoundError) as exc:
            # A missing key or a missing adapter is a skipped arm, not a crash
            # that throws away the arms that did run.
            print(f"  skipped {name}: {exc}")
            continue
        print(f"scoring {name} over {len(texts)} items ...", flush=True)
        results.append(evaluate_router(router, texts, labels))

    if not results:
        raise SystemExit("No arms ran. Check your API key or run src.finetune first.")

    print("\n" + _table(results))
    md, js = write_report(results, len(texts), len(set(labels)))
    print(f"report: {md}\njson:   {js}")


if __name__ == "__main__":
    main()
