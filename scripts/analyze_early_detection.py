#!/usr/bin/env python3
"""Compute Table 3 of the paper: early detection before and after fine-tuning.

For every phishing call in the test split the cumulative-context samples are ordered by
prefix_end_idx. The first sample whose gold label is 1 is the first risk sample. The metrics are:

  immediate        a positive prediction on the first risk sample
  within_two       a positive prediction anywhere from the first risk sample through two
                   intervals later (one interval is five utterances)
  pre_signal       a positive prediction on any sample before the first risk sample
  final_miss       no positive prediction on any sample from the first risk sample onward
  reversal         after the first positive prediction, a later sample predicted 0 again

Calls with no cumulative-context sample carrying a positive gold label are excluded, matching the
paper: 53 of the 59 phishing calls in the test split are analyzed.

Whole-call samples are not used. Predictions are joined to the test metadata by sample_id.

Usage:
  python scripts/analyze_early_detection.py \
      --test data/final/test.jsonl \
      --predictions outputs/finetuned_eval_qwen35_2b_base/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

INTERVAL_UTTERANCES = 5
WITHIN_INTERVALS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Early-detection metrics (paper Table 3).")
    parser.add_argument("--test", type=Path, required=True, help="test split JSONL")
    parser.add_argument(
        "--predictions", type=Path, required=True, help="predictions.jsonl from evaluate_model.py"
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()

    # Cumulative-context samples of phishing calls, grouped by call and ordered in time.
    by_call: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(args.test):
        if row.get("sample_type") != "prefix":
            continue
        if int(row.get("conversation_label", 0)) != 1:
            continue
        by_call[row["parent_conversation_id"]].append(row)
    for samples in by_call.values():
        samples.sort(key=lambda r: r["prefix_end_idx"])

    predicted = {
        str(row["sample_id"]): str(row.get("prediction"))
        for row in load_jsonl(args.predictions)
    }

    immediate = within_two = pre_signal = final_miss = reversal = 0
    analyzed = 0
    skipped_no_positive = 0
    missing_prediction = 0

    for samples in by_call.values():
        labels = [int(s["target_label"]) for s in samples]
        if 1 not in labels:
            skipped_no_positive += 1
            continue
        analyzed += 1
        t0 = labels.index(1)

        preds = []
        for sample in samples:
            value = predicted.get(str(sample["sample_id"]))
            if value is None:
                missing_prediction += 1
            preds.append(value)

        if any(p == "1" for p in preds[:t0]):
            pre_signal += 1
        if preds[t0] == "1":
            immediate += 1
        if any(p == "1" for p in preds[t0 : t0 + WITHIN_INTERVALS + 1]):
            within_two += 1

        after = preds[t0:]
        if "1" not in after:
            final_miss += 1
        else:
            first = after.index("1")
            if any(p == "0" for p in after[first + 1 :]):
                reversal += 1

    if missing_prediction:
        print(f"warning: {missing_prediction} samples had no matching prediction")
    print(f"phishing calls with cumulative-context samples : {len(by_call)}")
    print(f"excluded, no positive-label sample             : {skipped_no_positive}")
    print(f"analyzed                                       : {analyzed}")
    print()
    print(f"{'immediate detection':28s} {immediate}")
    print(f"{'within two intervals':28s} {within_two}")
    print(f"{'pre-signal positive':28s} {pre_signal}")
    print(f"{'final miss':28s} {final_miss}")
    print(f"{'reversal':28s} {reversal}")


if __name__ == "__main__":
    main()
