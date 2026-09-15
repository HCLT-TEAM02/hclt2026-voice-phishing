#!/usr/bin/env python3
"""
LoRA fine-tuning script for voice phishing detection.

Supports: Qwen3.5-2B-Base, Ministral-3-3B-Base-2512, EXAONE-4.0-1.2B, Midm-2.0-Mini-Instruct
Usage: python scripts/train_lora.py --config configs/<model>/lora_train_config.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import (
    load_base_model,
    load_tokenizer,
    prepare_training_example,
    resolve_lora_target_modules,
    validate_lora_placement,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for voice phishing detection.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training config JSON (e.g. configs/qwen/lora_train_config.json)",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_torch_dtype(torch: Any, dtype_name: str | None, fallback: Any) -> Any:
    if not dtype_name:
        return fallback
    dtype = getattr(torch, str(dtype_name), None)
    return dtype if dtype is not None else fallback


def uses_wandb(report_to: Any) -> bool:
    if isinstance(report_to, str):
        return report_to.lower() == "wandb"
    if isinstance(report_to, list):
        return any(str(v).lower() == "wandb" for v in report_to)
    return False


def tokenize_train_row(
    row: dict[str, Any], tokenizer: Any, config: dict[str, Any]
) -> dict[str, list[int]]:
    label = row.get("output")
    if label not in {"0", "1"}:
        raise ValueError(
            "학습 output은 문자열 '0' 또는 '1'이어야 합니다: "
            f"sample_id={row.get('sample_id')!r}, output={label!r}"
        )
    if row.get("target_label") != int(label):
        raise ValueError(
            "학습 output과 target_label이 일치하지 않습니다: "
            f"sample_id={row.get('sample_id')!r}, output={label!r}, "
            f"target_label={row.get('target_label')!r}"
        )
    prepared = prepare_training_example(
        tokenizer,
        instruction=row.get("instruction", ""),
        conversation=row.get("input", ""),
        label=label,
        config=config,
    )
    return {
        "input_ids": prepared.input_ids,
        "attention_mask": prepared.attention_mask,
        "labels": prepared.labels,
    }


def build_dataset_features():
    from datasets import Features, Value

    return Features(
        {
            "sample_id": Value("string"),
            "sample_type": Value("string"),
            "parent_conversation_id": Value("int64"),
            "conversation_label": Value("int64"),
            "target_label": Value("int64"),
            "prefix_end_idx": Value("int64"),
            "instruction": Value("string"),
            "input": Value("string"),
            "output": Value("string"),
        }
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "datasets/transformers/peft/torch가 설치되어 있지 않습니다. "
            "pip install -r requirements.txt 후 다시 실행하세요."
        ) from exc

    tokenizer = load_tokenizer(config, use_fast=True)

    dataset = load_dataset(
        "json",
        data_files={
            "train": config["train_file"],
            "valid": config["valid_file"],
        },
        features=build_dataset_features(),
    )
    use_4bit = bool(config.get("use_4bit", False))
    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        default_compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        compute_dtype = resolve_torch_dtype(
            torch,
            config.get("bnb_4bit_compute_dtype"),
            default_compute_dtype,
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", False)),
        )

    default_model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_dtype = resolve_torch_dtype(torch, config.get("torch_dtype"), default_model_dtype)
    model = load_base_model(
        config,
        dtype=model_dtype,
        quantization_config=bnb_config,
    )

    if use_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
    if bool(config.get("gradient_checkpointing", False)) and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=resolve_lora_target_modules(config),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    lora_structure = validate_lora_placement(model, config)
    os.makedirs(config["output_dir"], exist_ok=True)
    with open(
        os.path.join(config["output_dir"], "lora_structure.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(lora_structure, f, ensure_ascii=False, indent=2)
    print(json.dumps(lora_structure, ensure_ascii=False, indent=2))

    tokenized_train = dataset["train"].map(
        lambda row: tokenize_train_row(row, tokenizer, config),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing train split",
    )
    tokenized_valid = dataset["valid"].map(
        lambda row: tokenize_train_row(row, tokenizer, config),
        remove_columns=dataset["valid"].column_names,
        desc="Tokenizing valid split",
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    report_to = config.get("report_to", "none")
    wandb_project = config.get("wandb_project")
    if wandb_project and uses_wandb(report_to):
        os.environ.setdefault("WANDB_PROJECT", str(wandb_project))

    eval_steps = config.get("eval_steps")
    save_steps = config.get("save_steps")
    eval_strategy = "steps" if isinstance(eval_steps, int) and eval_steps > 0 else "epoch"
    save_strategy = "steps" if isinstance(save_steps, int) and save_steps > 0 else "epoch"

    training_arg_values: dict[str, Any] = {
        "output_dir": config["output_dir"],
        "num_train_epochs": float(config.get("epochs", 3)),
        "per_device_train_batch_size": int(config.get("batch_size", 2)),
        "per_device_eval_batch_size": int(config.get("batch_size", 2)),
        "gradient_accumulation_steps": int(config.get("grad_accum", 8)),
        "learning_rate": float(config.get("lr", 2e-4)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.0)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "logging_steps": int(config.get("logging_steps", 20)),
        "save_strategy": save_strategy,
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "bf16": bf16,
        "fp16": torch.cuda.is_available() and not bf16,
        "report_to": report_to,
        "run_name": config.get("run_name"),
        "load_best_model_at_end": bool(config.get("load_best_model_at_end", True)),
        "metric_for_best_model": config.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": bool(config.get("greater_is_better", False)),
        "remove_unused_columns": False,
    }
    if eval_strategy == "steps":
        training_arg_values["eval_steps"] = int(eval_steps)
    if save_strategy == "steps":
        training_arg_values["save_steps"] = int(save_steps)

    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in training_args_signature.parameters:
        training_arg_values["eval_strategy"] = eval_strategy
    else:
        training_arg_values["evaluation_strategy"] = eval_strategy

    training_args = TrainingArguments(**training_arg_values)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
    )
    trainer.train()

    final_dir = os.path.join(config["output_dir"], "adapter-final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    best_summary = {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "metric_for_best_model": training_args.metric_for_best_model,
        "greater_is_better": training_args.greater_is_better,
    }
    with open(os.path.join(final_dir, "best_checkpoint_summary.json"), "w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)

    print(f"Saved LoRA adapter to: {final_dir}")
    print(json.dumps(best_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
