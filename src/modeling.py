from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.common import SYSTEM_MESSAGE, make_prompt, trim_prompt_text


VALID_MODEL_FAMILIES = {"qwen35", "ministral3", "exaone4", "midm2"}

EXPECTED_PROMPT_MODES = {
    "qwen35": "chat",
    "ministral3": "plain",
    "exaone4": "chat",
    "midm2": "chat",
}

QWEN35_LORA_TARGET_REGEX = (
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)

MINISTRAL3_LORA_TARGET_REGEX = (
    r"^(?:model\.language_model|language_model\.model)\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)

STANDARD_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(frozen=True)
class PreparedTrainingExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_truncated: bool


def get_model_family(config: dict[str, Any]) -> str:
    family = str(config.get("model_family", "")).strip().lower()
    if family not in VALID_MODEL_FAMILIES:
        raise ValueError(
            f"model_family은 {sorted(VALID_MODEL_FAMILIES)} 중 하나여야 합니다: {family!r}"
        )
    return family


def get_prompt_mode(config: dict[str, Any]) -> str:
    family = get_model_family(config)
    prompt_mode = str(config.get("prompt_mode", "")).strip().lower()
    expected = EXPECTED_PROMPT_MODES[family]
    if prompt_mode != expected:
        raise ValueError(
            f"{family}의 prompt_mode은 {expected!r}이어야 합니다: {prompt_mode!r}"
        )
    return prompt_mode


def prompt_add_special_tokens(config: dict[str, Any]) -> bool:
    # Chat template이 이미 BOS/role special token을 렌더링한다.
    return get_prompt_mode(config) == "plain"


def build_model_prompt(
    tokenizer: Any,
    prompt: str,
    config: dict[str, Any],
    *,
    system_message: str = SYSTEM_MESSAGE,
) -> str:
    prompt_mode = get_prompt_mode(config)
    if prompt_mode == "plain":
        return f"{system_message}\n\n{prompt}"

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"prompt_mode='chat'이지만 {get_model_family(config)} tokenizer에 "
            "chat template이 없습니다."
        )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if "enable_thinking" in config:
        kwargs["enable_thinking"] = bool(config["enable_thinking"])

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        if "enable_thinking" not in kwargs:
            raise
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_prompt(tokenizer: Any, prompt_text: str, config: dict[str, Any]) -> list[int]:
    return tokenizer.encode(
        prompt_text,
        add_special_tokens=prompt_add_special_tokens(config),
    )


def get_label_token_ids(tokenizer: Any, label: str) -> list[int]:
    if label not in {"0", "1"}:
        raise ValueError(f"학습/평가 라벨은 0 또는 1이어야 합니다: {label!r}")
    token_ids = tokenizer.encode(label, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"라벨 {label!r}이 token으로 인코딩되지 않았습니다.")
    return list(token_ids)


def get_candidate_label_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    return {label: get_label_token_ids(tokenizer, label) for label in ("0", "1")}


def prepare_prompt(
    tokenizer: Any,
    instruction: str,
    conversation: str,
    config: dict[str, Any],
    *,
    max_prompt_tokens: int,
) -> tuple[str, list[int], bool]:
    raw_prompt = make_prompt(instruction, conversation)
    return prepare_raw_prompt(
        tokenizer,
        raw_prompt,
        config,
        max_prompt_tokens=max_prompt_tokens,
    )


def prepare_raw_prompt(
    tokenizer: Any,
    raw_prompt: str,
    config: dict[str, Any],
    *,
    max_prompt_tokens: int,
    system_message: str = SYSTEM_MESSAGE,
) -> tuple[str, list[int], bool]:
    """Apply the configured model template and the standard truncation policy."""
    prompt_text = build_model_prompt(
        tokenizer,
        raw_prompt,
        config,
        system_message=system_message,
    )
    prompt_text, was_truncated = trim_prompt_text(
        tokenizer,
        prompt_text,
        max_prompt_tokens,
        add_special_tokens=prompt_add_special_tokens(config),
    )
    prompt_ids = encode_prompt(tokenizer, prompt_text, config)
    if len(prompt_ids) > max_prompt_tokens:
        raise ValueError(
            f"프롬프트 길이 제한 적용 실패: {len(prompt_ids)} > {max_prompt_tokens}"
        )
    return prompt_text, prompt_ids, was_truncated


def prepare_training_example(
    tokenizer: Any,
    *,
    instruction: str,
    conversation: str,
    label: str,
    config: dict[str, Any],
) -> PreparedTrainingExample:
    label_ids = get_label_token_ids(tokenizer, label)
    max_length = int(config.get("max_length", 1024))
    max_prompt_tokens = max_length - len(label_ids)
    if max_prompt_tokens < 1:
        raise ValueError("max_length가 정답 token을 포함하기에 너무 작습니다.")

    _, prompt_ids, was_truncated = prepare_prompt(
        tokenizer,
        instruction,
        conversation,
        config,
        max_prompt_tokens=max_prompt_tokens,
    )
    input_ids = prompt_ids + label_ids
    labels = [-100] * len(prompt_ids) + label_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > max_length:
        raise ValueError(f"학습 입력 길이 초과: {len(input_ids)} > {max_length}")
    if input_ids[-len(label_ids) :] != label_ids:
        raise AssertionError("input_ids의 마지막 token이 정답 token과 다릅니다.")
    if labels[-len(label_ids) :] != label_ids or not any(v != -100 for v in labels):
        raise AssertionError("정답 token이 loss 대상에 포함되지 않았습니다.")

    return PreparedTrainingExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_truncated=was_truncated,
    )


def load_tokenizer(config: dict[str, Any], *, use_fast: bool = True) -> Any:
    family = get_model_family(config)
    model_path = config["model_name_or_path"]
    common_kwargs = {
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }

    if family == "ministral3":
        from transformers import MistralCommonBackend

        tokenizer = MistralCommonBackend.from_pretrained(model_path, **common_kwargs)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=use_fast,
            **common_kwargs,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    get_prompt_mode(config)
    if get_prompt_mode(config) == "chat" and not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{family} tokenizer의 chat template을 찾을 수 없습니다.")
    get_candidate_label_token_ids(tokenizer)
    return tokenizer


def load_base_model(
    config: dict[str, Any],
    *,
    dtype: Any,
    quantization_config: Any = None,
) -> Any:
    family = get_model_family(config)
    if family == "qwen35":
        from transformers import Qwen3_5ForConditionalGeneration as ModelClass
    elif family == "ministral3":
        from transformers import Mistral3ForConditionalGeneration as ModelClass
    else:
        from transformers import AutoModelForCausalLM as ModelClass

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": config.get("device_map", "auto"),
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if "max_memory" in config:
        model_kwargs["max_memory"] = {
            int(k) if isinstance(k, str) and k.isdigit() else k: v
            for k, v in config["max_memory"].items()
        }
    return ModelClass.from_pretrained(config["model_name_or_path"], **model_kwargs)


def resolve_lora_target_modules(config: dict[str, Any]) -> str | list[str]:
    profile = str(config.get("lora_target_profile", "")).strip().lower()
    family = get_model_family(config)
    expected_profile = {
        "qwen35": "qwen35_text",
        "ministral3": "ministral3_text",
        "exaone4": "standard_text",
        "midm2": "standard_text",
    }[family]
    if profile != expected_profile:
        raise ValueError(
            f"{family}의 lora_target_profile은 {expected_profile!r}이어야 합니다: {profile!r}"
        )
    if profile == "qwen35_text":
        return QWEN35_LORA_TARGET_REGEX
    if profile == "ministral3_text":
        return MINISTRAL3_LORA_TARGET_REGEX
    return STANDARD_LORA_TARGET_MODULES


def validate_lora_placement(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    family = get_model_family(config)
    targeted_names = sorted(set(getattr(model, "targeted_module_names", []) or []))
    trainable_names = sorted(name for name, param in model.named_parameters() if param.requires_grad)
    if not targeted_names or not trainable_names:
        raise RuntimeError("LoRA가 어떤 모듈에도 적용되지 않았습니다.")

    banned_fragments = {
        "qwen35": (".visual.", "mtp."),
        "ministral3": ("vision_tower.", "multi_modal_projector."),
        "exaone4": (),
        "midm2": (),
    }[family]
    bad_targets = [
        name for name in targeted_names if any(fragment in name for fragment in banned_fragments)
    ]
    bad_trainable = [
        name for name in trainable_names if any(fragment in name for fragment in banned_fragments)
    ]
    if bad_targets or bad_trainable:
        raise RuntimeError(
            f"텍스트 backbone 밖에 LoRA가 적용됐습니다: "
            f"targets={bad_targets[:10]}, trainable={bad_trainable[:10]}"
        )

    target_spec = resolve_lora_target_modules(config)
    if isinstance(target_spec, str):
        unexpected = [name for name in targeted_names if re.fullmatch(target_spec, name) is None]
        if unexpected:
            raise RuntimeError(f"예상하지 않은 LoRA target: {unexpected[:10]}")

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    all_params = sum(param.numel() for param in model.parameters())
    return {
        "model_family": family,
        "lora_target_profile": config["lora_target_profile"],
        "targeted_module_count": len(targeted_names),
        "targeted_module_names": targeted_names,
        "trainable_parameter_tensor_count": len(trainable_names),
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_ratio": trainable_params / all_params if all_params else 0.0,
    }


def resolve_model_input_device(model: Any) -> Any:
    embeddings = model.get_input_embeddings()
    if embeddings is not None:
        return embeddings.weight.device
    return next(model.parameters()).device
