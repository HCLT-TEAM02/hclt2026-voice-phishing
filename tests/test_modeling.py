import json
import re
from pathlib import Path

from src.modeling import (
    MINISTRAL3_LORA_TARGET_REGEX,
    QWEN35_LORA_TARGET_REGEX,
    get_candidate_label_token_ids,
    load_tokenizer,
    prepare_training_example,
)
from scripts.train_lora import tokenize_train_row


ROOT = Path(__file__).resolve().parents[1]


def test_real_tokenizers_prepare_exact_label_boundary():
    for model_name in ("qwen", "ministral", "exaone", "midm"):
        model_dir = ROOT / "models" / model_name
        if not model_dir.exists():
            continue
        config = json.loads((ROOT / "configs" / model_name / "lora_train_config.json").read_text())
        tokenizer = load_tokenizer(config)
        candidate_ids = get_candidate_label_token_ids(tokenizer)

        for label in ("0", "1"):
            prepared = prepare_training_example(
                tokenizer,
                instruction="보이스피싱 여부를 분류하라.",
                conversation="A: 안전한 문의입니다.\nB: 감사합니다.",
                label=label,
                config=config,
            )
            label_ids = candidate_ids[label]
            assert prepared.input_ids[-len(label_ids) :] == label_ids
            assert prepared.labels[-len(label_ids) :] == label_ids
            assert all(value == -100 for value in prepared.labels[: -len(label_ids)])


def test_long_real_input_keeps_tail_and_exact_max_length():
    data_path = ROOT / "data" / "final" / "train.jsonl"
    if not data_path.exists():
        return
    longest = max(
        (json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line),
        key=lambda row: len(row["input"]),
    )
    for model_name in ("qwen", "ministral", "exaone", "midm"):
        model_dir = ROOT / "models" / model_name
        if not model_dir.exists():
            continue
        config = json.loads(
            (ROOT / "configs" / model_name / "lora_train_config.json").read_text()
        )
        tokenizer = load_tokenizer(config)
        prepared = prepare_training_example(
            tokenizer,
            instruction=longest["instruction"],
            conversation=longest["input"],
            label=longest["output"],
            config=config,
        )

        decoded = tokenizer.decode(prepared.input_ids, skip_special_tokens=True)
        assert prepared.prompt_truncated is True, model_name
        assert len(prepared.input_ids) <= config["max_length"], model_name
        assert longest["input"][-20:] in decoded, model_name
        assert longest["input"][:20] not in decoded, model_name
        assert "[중략]" not in decoded, model_name


def test_multimodal_lora_regexes_only_match_text_backbones():
    cases = (
        (
            ROOT / "models" / "qwen" / "model.safetensors.index.json",
            QWEN35_LORA_TARGET_REGEX,
            ("model.visual.", "mtp."),
        ),
        (
            ROOT / "models" / "ministral" / "model.safetensors.index.json",
            MINISTRAL3_LORA_TARGET_REGEX,
            ("vision_tower.", "multi_modal_projector."),
        ),
    )
    for index_path, target_regex, banned_prefixes in cases:
        if not index_path.exists():
            continue
        weight_names = json.loads(index_path.read_text())["weight_map"]
        matched_modules = {
            name.removesuffix(".weight")
            for name in weight_names
            if name.endswith(".weight") and re.fullmatch(target_regex, name.removesuffix(".weight"))
        }
        assert matched_modules
        assert not any(
            banned in module_name
            for module_name in matched_modules
            for banned in banned_prefixes
        )


def test_training_data_matches_expected_schema():
    data_path = ROOT / "data" / "final" / "train.jsonl"
    if not data_path.exists():
        return
    expected_keys = {
        "sample_id",
        "sample_type",
        "parent_conversation_id",
        "conversation_label",
        "target_label",
        "prefix_end_idx",
        "instruction",
        "input",
        "output",
    }
    with data_path.open(encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            assert set(row) == expected_keys, line_number
            assert row["output"] in {"0", "1"}, line_number
            assert row["target_label"] in {0, 1}, line_number
            assert row["output"] == str(row["target_label"]), line_number


def test_invalid_training_label_is_rejected_instead_of_coerced():
    invalid_rows = (
        {"sample_id": "bad-output", "output": "other", "target_label": 0},
        {"sample_id": "mismatch", "output": "1", "target_label": 0},
    )
    for row in invalid_rows:
        try:
            tokenize_train_row(row, tokenizer=None, config={})
        except ValueError:
            pass
        else:
            raise AssertionError(f"잘못된 학습 라벨이 통과했습니다: {row}")
