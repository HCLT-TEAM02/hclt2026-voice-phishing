# Checklist

Binary questions for judging the quality of augmented calls. This is an artifact written by the
authors and may be reused freely under the repository license.

| File | Contents |
|---|---|
| `chk_v3.jsonl` | Final 48 questions (4 dimensions / 17 sub-dimensions) |
| `PROVENANCE.md` | Version history: how the 48 questions were selected from chk_v2 (112 questions) |

Construction procedure and validation results are documented in
[`../docs/checklist.md`](../docs/checklist.md).

## Usage

Each line is one question. All 48 questions are applied to one augmentation and answered Yes or No
while viewing the original call side by side. The items are written in Korean because the calls they
judge are Korean.

```jsonc
{
  "question_id": "D1-S1-Q01",
  "dimension_id": "D1",
  "dimension": "핵심 의미 및 사기 전략 보존",
  "sub_dimension_id": "D1-S1",
  "sub_dimension": "사기 시나리오 보존",
  "question": "증강 통화는 원본과 동일한 보이스피싱 시나리오를 유지하는가?",
  "sources": ["seed"]
}
```

`sources` records where a question came from: `seed` means it was defined directly by the authors,
`elaboration` means it was derived during the LLM elaboration step.

> **Caution.** Question IDs were renumbered per sub-dimension, so the same ID refers to different
> questions in chk_v2 and chk_v3. Never mix judgments across the two versions.
