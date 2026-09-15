# Augmentation quality filter

`excluded_augmentations.csv` lists the augmented calls excluded by the quality filter described in
section 3.2 of the paper: a candidate judged "No" on two or more of the 48 checklist questions.

| Column | Meaning |
|---|---|
| `conversation_id` | Id of the excluded augmented call |
| `no_count` | Number of questions judged "No" out of 48 |

| | Count |
|---|---:|
| Candidate augmentations | 1,104 |
| Excluded (two or more "No") | 187 |
| Kept | 917 |

Judgments were produced with Claude Opus 5 against
[`../../checklist/chk_v3.jsonl`](../../checklist/chk_v3.jsonl), using the per-dimension prompts in
[`../../prompts/checklist_judge_v2/`](../../prompts/checklist_judge_v2/). Four candidates failed
the judging step and have no verdict; they are absent from this list and therefore treated as
passing, which is why 1,104 - 187 = 917.

The file carries ids and counts only, no call content.
[`../../scripts/build_samples.py`](../../scripts/build_samples.py) reads it when assembling the
splits.
