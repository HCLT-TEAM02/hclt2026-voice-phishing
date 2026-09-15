# Augmentation Quality Checklist

The checklist verifies that an augmented call preserves the original scam strategy while varying only
its surface form, and this page records how the checklist was built and how the LLM judge was
validated.

## 1. Checklist Construction

The construction procedure of CheckEval was adapted to this dataset. The idea is to decompose an
abstract quality criterion into concrete binary questions so that LLM judgments become consistent
and interpretable.

**Procedure**

1. The authors defined the evaluation dimensions and seed questions.
2. Claude Opus 5 diversified and elaborated the seed questions and filtered them
   (prompts: [`../prompts/checklist_construction/`](../prompts/checklist_construction/)), producing
   118 candidate questions.
3. The authors then removed questions that apply only to specific calls, questions that resist a
   binary answer, and questions where a "Yes" answer does not always indicate good augmentation
   quality, leaving the final **48 questions**.

**Dimensions: 4 dimensions / 17 sub-dimensions / 48 questions**

| Dimension | Focus | Questions |
|---|---|---:|
| D1 | Preservation of core meaning and scam strategy | 18 |
| D2 | Dialogue coherence | 12 |
| D3 | Naturalness | 14 |
| D4 | Augmentation diversity | 4 |

The full checklist is in [`../checklist/chk_v3.jsonl`](../checklist/chk_v3.jsonl) (items are written
in Korean, matching the data they judge). Version history is in
[`../checklist/PROVENANCE.md`](../checklist/PROVENANCE.md).

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

## 2. LLM Judge Validation

50 candidate augmentations were sampled and all 48 questions were applied to each. Three human raters
and Claude Opus 5 reviewed the original and the augmentation side by side and judged the resulting
**2,400 augmentation-question pairs** independently.

### Human inter-rater agreement

| Metric | Value |
|---|---|
| 3-way full agreement | 97.42% |
| Mean pairwise agreement | 98.29% |
| Fleiss' kappa | 0.36 |

Fleiss' kappa looks low, but "No" makes up only 0.83% of all pairs under the human majority vote.
With a response distribution this skewed, chance-corrected kappa is heavily penalized, which explains
the gap between raw agreement and kappa.

### Human majority vs LLM

| Metric | Value |
|---|---|
| Raw agreement | 98.92% |
| Cohen's kappa | 0.60 |
| Recall on "No" | 100.00% |
| Precision on "No" | 43.48% |
| F1 on "No" | 60.61% |
| Score correlation (sum of "Yes" per augmentation) | r = 0.84, p < 0.001 |

With "No" as the detection target, the LLM caught every question the human majority marked "No",
at the cost of extra "No" judgments of its own. Since most augmentations score 47 or 48 out of 48,
the LLM judge is used as a **conservative filter that flags clear quality problems**, not as a ranker
of already-good augmentations.

## 3. Decision Rule and Application

| Judgment | Action | Rationale |
|---|---|---|
| 0 or 1 "No" | Keep | A single "No" can come from item interpretation or minor stylistic issues |
| 2 or more "No" | Exclude | Multiple independent quality problems confirmed |

The validated judge then screened every candidate with the same checklist and rule
(judge prompts: [`../prompts/checklist_judge_v2/`](../prompts/checklist_judge_v2/)). **917 of the
1,104 candidates passed** and became the final augmented data.

| | Count | Share |
|---|---:|---:|
| Candidate augmentations | 1,104 | 100.0% |
| Excluded (2+ "No") | 187 | 16.9% |
| **Passed** | **917** | **83.1%** |

## 4. Limitations

The screening verifies preservation and variation **between each augmentation and its original**.
Diversity among the three augmentations of one original call was not separately evaluated.

The full screening is the judgment of a single LLM judge (Claude Opus 5). On the validation sample
the LLM sat between the three human raters in strictness, but rater-to-rater variance exists and
should be kept in mind when interpreting the pass rate.
