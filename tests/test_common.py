from src.common import trim_prompt_text


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(tokens)


def test_trim_prompt_keeps_contiguous_conversation_tail():
    tokenizer = CharacterTokenizer()
    prefix = "지시\n\n[통화 내용]\n"
    conversation = "ABCDEFGHIJ"
    suffix = "\n\n[답변]\n"
    prompt = f"{prefix}{conversation}{suffix}"
    tail_budget = 4

    trimmed, was_truncated = trim_prompt_text(
        tokenizer,
        prompt,
        len(prefix) + len(suffix) + tail_budget,
    )

    assert was_truncated is True
    assert trimmed == f"{prefix}GHIJ{suffix}"
    assert "ABCD" not in trimmed
    assert "[중략]" not in trimmed


def test_trim_prompt_leaves_short_prompt_unchanged():
    tokenizer = CharacterTokenizer()
    prompt = "지시\n\n[통화 내용]\n짧은 통화\n\n[답변]\n"

    trimmed, was_truncated = trim_prompt_text(tokenizer, prompt, len(prompt))

    assert was_truncated is False
    assert trimmed == prompt
