#!/usr/bin/env python3
"""
Unified evaluation script for baseline and fine-tuned models.

Baseline:  set adapter_path to null (or omit) in config
Finetuned: set adapter_path to the LoRA adapter directory in config
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import calc_binary_metrics, get_row_label
from src.modeling import (
    get_candidate_label_token_ids,
    load_base_model,
    load_tokenizer,
    prepare_prompt,
    resolve_model_input_device,
)


VALID_LABELS = {"0", "1"}
CANDIDATE_LABELS = ["0", "1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a (fine-tuned) model on a JSONL test set.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to evaluation config JSON (e.g. configs/qwen/baseline_eval_config.json)",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if line_idx == len(lines) - 1:
                        break
                    raise
    return rows


def resolve_effective_max_length(tokenizer: Any, config: dict[str, Any]) -> int:
    config_max_length = config.get("max_length")
    if isinstance(config_max_length, int) and config_max_length > 0:
        return config_max_length
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(model_max_length, int) or model_max_length <= 0 or model_max_length > 100_000:
        return 4096
    return model_max_length


def score_candidate_labels(
    model: Any,
    tokenizer: Any,
    prompt_ids_by_row: list[list[int]],
    candidate_token_ids: dict[str, list[int]],
    model_input_device: Any,
) -> list[dict[str, float]]:
    import torch

    prompt_encoded = tokenizer.pad(
        [
            {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
            for input_ids in prompt_ids_by_row
        ],
        padding=True,
        return_tensors="pt",
    )
    prompt_lengths = prompt_encoded["attention_mask"].sum(dim=1).tolist()
    scores_by_row = [dict() for _ in prompt_ids_by_row]

    # Fast path: single-token labels ("0"/"1") need only one forward pass
    label_token_ids: dict[str, int] = {}
    for label, token_ids in candidate_token_ids.items():
        if len(token_ids) != 1:
            label_token_ids = {}
            break
        label_token_ids[label] = token_ids[0]

    if label_token_ids:
        encoded = {k: v.to(model_input_device) for k, v in prompt_encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits

        next_token_log_probs = torch.log_softmax(logits, dim=-1)
        for row_idx, prompt_length in enumerate(prompt_lengths):
            next_token_pos = max(int(prompt_length) - 1, 0)
            row_log_probs = next_token_log_probs[row_idx, next_token_pos]
            for label, token_id in label_token_ids.items():
                scores_by_row[row_idx][label] = float(row_log_probs[token_id].item())
        return scores_by_row

    for label, label_ids in candidate_token_ids.items():
        full_input_ids = [prompt_ids + label_ids for prompt_ids in prompt_ids_by_row]
        encoded = tokenizer.pad(
            [
                {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
                for input_ids in full_input_ids
            ],
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(model_input_device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits

        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        shifted_input_ids = encoded["input_ids"][:, 1:]
        for row_idx, prompt_length in enumerate(prompt_lengths):
            label_start = max(int(prompt_length) - 1, 0)
            label_positions = torch.arange(
                label_start,
                label_start + len(label_ids),
                device=log_probs.device,
            )
            token_ids = shifted_input_ids[row_idx, label_positions]
            expected_ids = torch.tensor(label_ids, device=token_ids.device)
            if not torch.equal(token_ids, expected_ids):
                raise RuntimeError("평가 candidate token 경계가 일치하지 않습니다.")
            token_log_probs = log_probs[row_idx, label_positions, token_ids]
            scores_by_row[row_idx][label] = float(token_log_probs.sum().item())

    return scores_by_row


def synchronize_model_device(model_input_device: Any) -> None:
    """Wait for queued CUDA work so wall-clock inference timing is accurate."""
    import torch

    if torch.cuda.is_available():
        device_type = getattr(model_input_device, "type", None)
        torch.cuda.synchronize(model_input_device if device_type == "cuda" else None)


def load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    try:
        import torch
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit(
            "transformers/torch/peft가 설치되어 있지 않습니다. "
            "pip install -r requirements.txt 후 다시 실행하세요."
        ) from exc

    tokenizer = load_tokenizer(config)

    torch_dtype_name = config.get("torch_dtype", "bfloat16")
    torch_dtype = getattr(__import__("torch"), torch_dtype_name)
    model = load_base_model(config, dtype=torch_dtype)

    adapter_path = config.get("adapter_path")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    model_input_device = resolve_model_input_device(model)
    return model, tokenizer, model_input_device


def restore_previous_run(
    predictions_path: Path,
    resume_enabled: bool,
) -> tuple[list[dict[str, Any]], set[str], list[str], list[str]]:
    prediction_rows: list[dict[str, Any]] = []
    processed_sample_ids: set[str] = set()
    y_true: list[str] = []
    y_pred: list[str] = []

    if not (resume_enabled and predictions_path.exists()):
        return prediction_rows, processed_sample_ids, y_true, y_pred

    prediction_rows = load_jsonl(predictions_path)
    for row in prediction_rows:
        sample_id = row.get("sample_id")
        if sample_id is not None:
            processed_sample_ids.add(str(sample_id))
        gold = row.get("gold")
        prediction = row.get("prediction")
        if gold in VALID_LABELS and prediction in VALID_LABELS:
            y_true.append(gold)
            y_pred.append(prediction)
    return prediction_rows, processed_sample_ids, y_true, y_pred


def compute_next_progress_report(
    processed_count_before_run: int,
    progress_every_rows: int,
    total_rows: int,
) -> int:
    if processed_count_before_run >= total_rows:
        return total_rows
    return ((processed_count_before_run // progress_every_rows) + 1) * progress_every_rows


def build_prediction_row(
    row: dict[str, Any],
    score_map: dict[str, float],
    prompt_text: str,
    was_truncated: bool,
    inference_seconds: float,
    batch_inference_seconds: float,
    inference_batch_size: int,
) -> dict[str, Any]:
    gold = get_row_label(row)
    prediction = max(CANDIDATE_LABELS, key=lambda label: score_map.get(label, float("-inf")))
    return {
        "sample_id": row["sample_id"],
        "sample_type": row.get("sample_type"),
        "parent_conversation_id": row.get("parent_conversation_id"),
        "gold": gold or None,
        "prediction": prediction,
        "raw_prediction_text": prediction,
        "label_scores": score_map,
        "raw_label_log_probabilities": score_map,
        "selected_label_log_probability": score_map.get(prediction),
        "inference_seconds": inference_seconds,
        "batch_inference_seconds": batch_inference_seconds,
        "inference_batch_size": inference_batch_size,
        "prompt_truncated": was_truncated,
        "prompt_char_length": len(prompt_text),
        "correct": prediction == gold if gold in VALID_LABELS else None,
    }


def maybe_record_metric_labels(
    prediction_row: dict[str, Any],
    y_true: list[str],
    y_pred: list[str],
) -> None:
    gold = prediction_row["gold"]
    prediction = prediction_row["prediction"]
    if gold in VALID_LABELS and prediction in VALID_LABELS:
        y_true.append(gold)
        y_pred.append(prediction)


def maybe_print_progress(
    processed_total: int,
    next_progress_report: int,
    progress_every_rows: int,
    total_rows: int,
    started_at: float,
    processed_count_before_run: int,
) -> int:
    if processed_total < next_progress_report:
        return next_progress_report

    elapsed_seconds = time.time() - started_at
    processed_in_run = max(1, processed_total - processed_count_before_run)
    rows_left = max(0, total_rows - processed_total)
    eta_seconds = int((elapsed_seconds / processed_in_run) * rows_left)
    print(
        f"[PROGRESS] {processed_total}/{total_rows} "
        f"(elapsed={int(elapsed_seconds)}s, eta~{eta_seconds}s)"
    )
    return next_progress_report + progress_every_rows


def flush_predictions_file(predictions_file: Any) -> None:
    predictions_file.flush()
    os.fsync(predictions_file.fileno())


def run_evaluation(
    *,
    dataset: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    model_input_device: Any,
    predictions_path: Path,
    resume_enabled: bool,
    flush_every_rows: int,
    progress_every_rows: int,
    batch_size: int,
    max_prompt_tokens: int,
    candidate_token_ids: dict[str, list[int]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str], int, int]:
    prediction_rows, processed_sample_ids, y_true, y_pred = restore_previous_run(
        predictions_path=predictions_path,
        resume_enabled=resume_enabled,
    )

    predictions_file_mode = "a" if resume_enabled and predictions_path.exists() else "w"
    processed_count_before_run = len(processed_sample_ids)
    total_rows = len(dataset)
    started_at = time.time()
    next_progress_report = compute_next_progress_report(
        processed_count_before_run=processed_count_before_run,
        progress_every_rows=progress_every_rows,
        total_rows=total_rows,
    )
    rows_since_flush = 0

    with predictions_path.open(predictions_file_mode, encoding="utf-8") as predictions_file:
        for start_idx in range(0, len(dataset), batch_size):
            candidate_batch_rows = dataset[start_idx : start_idx + batch_size]
            batch_rows = [
                row
                for row in candidate_batch_rows
                if str(row.get("sample_id")) not in processed_sample_ids
            ]
            if not batch_rows:
                continue

            prepared_prompts = [
                prepare_prompt(
                    tokenizer,
                    row.get("instruction", ""),
                    row.get("input", ""),
                    config,
                    max_prompt_tokens=max_prompt_tokens,
                )
                for row in batch_rows
            ]
            prompt_texts = [prompt_text for prompt_text, _, _ in prepared_prompts]
            prompt_ids_by_row = [prompt_ids for _, prompt_ids, _ in prepared_prompts]
            synchronize_model_device(model_input_device)
            inference_started_at = time.perf_counter()
            label_scores = score_candidate_labels(
                model=model,
                tokenizer=tokenizer,
                prompt_ids_by_row=prompt_ids_by_row,
                candidate_token_ids=candidate_token_ids,
                model_input_device=model_input_device,
            )
            synchronize_model_device(model_input_device)
            batch_inference_seconds = time.perf_counter() - inference_started_at
            inference_seconds = batch_inference_seconds / len(batch_rows)

            for row, score_map, (_, _, was_truncated), prompt_text in zip(
                batch_rows,
                label_scores,
                prepared_prompts,
                prompt_texts,
            ):
                prediction_row = build_prediction_row(
                    row=row,
                    score_map=score_map,
                    prompt_text=prompt_text,
                    was_truncated=was_truncated,
                    inference_seconds=inference_seconds,
                    batch_inference_seconds=batch_inference_seconds,
                    inference_batch_size=len(batch_rows),
                )
                maybe_record_metric_labels(prediction_row, y_true, y_pred)
                prediction_rows.append(prediction_row)
                processed_sample_ids.add(str(row.get("sample_id")))

                predictions_file.write(json.dumps(prediction_row, ensure_ascii=False) + "\n")

                processed_total = len(processed_sample_ids)
                next_progress_report = maybe_print_progress(
                    processed_total=processed_total,
                    next_progress_report=next_progress_report,
                    progress_every_rows=progress_every_rows,
                    total_rows=total_rows,
                    started_at=started_at,
                    processed_count_before_run=processed_count_before_run,
                )

                rows_since_flush += 1
                if rows_since_flush >= flush_every_rows:
                    flush_predictions_file(predictions_file)
                    rows_since_flush = 0

        if rows_since_flush > 0:
            flush_predictions_file(predictions_file)

    return prediction_rows, y_true, y_pred, processed_count_before_run, len(processed_sample_ids)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)

    dataset_path = Path(config["dataset_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    dataset = load_jsonl(dataset_path)
    max_samples = config.get("max_samples")
    if isinstance(max_samples, int):
        dataset = dataset[:max_samples]

    model, tokenizer, model_input_device = load_model_and_tokenizer(config)
    adapter_path = config.get("adapter_path")

    resume_enabled = bool(config.get("resume", True))
    flush_every_rows = max(1, int(config.get("flush_every_rows", 10)))
    progress_every_rows = max(1, int(config.get("progress_every_rows", 100)))
    batch_size = max(1, int(config.get("batch_size", 1)))
    max_length = resolve_effective_max_length(tokenizer, config)
    candidate_token_ids = get_candidate_label_token_ids(tokenizer)
    label_token_budget = max(len(token_ids) for token_ids in candidate_token_ids.values())
    if max_length <= label_token_budget:
        raise ValueError(
            f"max_length={max_length}가 평가 라벨 token 예약 "
            f"{label_token_budget}보다 커야 합니다."
        )
    max_prompt_tokens = max_length - label_token_budget

    (
        prediction_rows,
        y_true,
        y_pred,
        processed_count_before_run,
        processed_total_after_run,
    ) = run_evaluation(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        model_input_device=model_input_device,
        predictions_path=predictions_path,
        resume_enabled=resume_enabled,
        flush_every_rows=flush_every_rows,
        progress_every_rows=progress_every_rows,
        batch_size=batch_size,
        max_prompt_tokens=max_prompt_tokens,
        candidate_token_ids=candidate_token_ids,
        config=config,
    )

    metrics = calc_binary_metrics(y_true, y_pred)
    truncated_count = sum(1 for row in prediction_rows if row["prompt_truncated"])
    eval_mode = "finetuned" if adapter_path else "baseline"
    report = {
        "model_name": config["model_name_or_path"],
        "adapter_path": adapter_path,
        "eval_mode": eval_mode,
        "test_file": str(dataset_path),
        "prediction_mode": "label_scoring",
        "max_length": max_length,
        "label_token_budget": label_token_budget,
        "candidate_token_ids": candidate_token_ids,
        "truncated_prompt_count": truncated_count,
        "num_rows": len(prediction_rows),
        "labeled_rows": len(y_true),
        "unlabeled_rows": len(prediction_rows) - len(y_true),
        "metrics": metrics,
        "examples": prediction_rows[:20],
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"[{eval_mode.upper()}] resume={resume_enabled} "
        f"processed_before_run={processed_count_before_run} "
        f"processed_total={processed_total_after_run}"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
