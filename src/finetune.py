"""LoRA fine-tune of a small causal LM as a 77-way intent classifier.

Run:  python -m src.finetune
Writes the adapter to artifacts/lora-router/. Nothing else in the project
fabricates a tuned model, so until this has run the `lora` arm simply errors.
"""
from __future__ import annotations

import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .config import ARTIFACTS, BASE_MODEL, LORA, MAX_SEQ_LEN, SEED
from .data import label_names, load_splits


def _metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def main() -> None:
    out_dir = ARTIFACTS / "lora-router"
    names = label_names()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        # Qwen and most causal LMs ship without a pad token; classification
        # batching needs one, and reusing EOS is the standard choice.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(names),
        id2label={i: n for i, n in enumerate(names)},
        label2id={n: i for i, n in enumerate(names)},
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA.r,
        lora_alpha=LORA.alpha,
        lora_dropout=LORA.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length"
        )

    splits = load_splits().map(tokenize, batched=True)
    splits = splits.rename_column("label", "labels")
    keep = ["input_ids", "attention_mask", "labels"]
    splits.set_format("torch", columns=keep)

    # bf16/fp16 on CUDA is the difference between ~10 minutes on a Colab T4 and
    # a run that never finishes. Apple MPS has no usable mixed precision here, so
    # it stays fp32 - and at fp32 this model is too slow on MPS to be worth it.
    import torch as _torch

    on_cuda = _torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(ARTIFACTS / "trainer"),
        fp16=on_cuda,
        dataloader_pin_memory=on_cuda,
        learning_rate=LORA.learning_rate,
        per_device_train_batch_size=LORA.batch_size,
        per_device_eval_batch_size=LORA.batch_size,
        num_train_epochs=LORA.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        compute_metrics=_metrics,
    )
    trainer.train()

    print("\nValidation:", trainer.evaluate())
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Adapter written to {out_dir}")
    print("Now run: python -m src.evaluate --arms tfidf,lora")


if __name__ == "__main__":
    main()
