"""banking77 loading and label helpers.

banking77 is 13k short customer-support utterances over 77 intents. It is small
enough to fine-tune cheaply and hard enough that a keyword baseline loses, which
is what makes the base-vs-tuned comparison worth reporting.
"""
from __future__ import annotations

from functools import lru_cache

from datasets import DatasetDict, Value, load_dataset

from .config import DATASET_ID, SEED


@lru_cache(maxsize=1)
def load_splits(val_fraction: float = 0.1) -> DatasetDict:
    """Return train/validation/test. banking77 ships train+test only, so the
    validation split is carved out of train with a fixed seed."""
    raw = load_dataset(DATASET_ID)
    raw = raw.cast_column("label", Value("int64"))
    split = raw["train"].train_test_split(test_size=val_fraction, seed=SEED)
    return DatasetDict(
        train=split["train"],
        validation=split["test"],
        test=raw["test"],
    )


@lru_cache(maxsize=1)
def label_names() -> list[str]:
    """Ordered label names, index == label id.

    The mteb mirror stores `label` as a plain int rather than a ClassLabel, so
    the names come from the paired `label_text` column instead of `.names`.
    """
    splits = load_splits()
    mapping: dict[int, str] = {}
    for split in ("train", "test"):
        ds = splits[split]
        for lid, text in zip(ds["label"], ds["label_text"]):
            mapping.setdefault(int(lid), str(text))
    missing = set(range(len(mapping))) - mapping.keys()
    if missing:
        raise RuntimeError(f"label ids absent from the dataset: {sorted(missing)}")
    return [mapping[i] for i in range(len(mapping))]


def label_to_text(label_id: int) -> str:
    """'card_arrival' -> 'card arrival'. Used in prompts, where underscores
    measurably hurt small models."""
    return label_names()[label_id].replace("_", " ")


def texts_and_labels(split: str) -> tuple[list[str], list[int]]:
    ds = load_splits()[split]
    return list(ds["text"]), list(ds["label"])
