# Prompts

Every prompt used to build and screen the dataset, exactly as run in the pipeline. Prompt versions
below match the `prompt_version` values recorded in the pipeline database, so each judgment and
label in the released data can be traced to the prompt that produced it. Prompts are written in
Korean, matching the data they operate on.

| File | Stage | Version tag | Model | Temperature |
|---|---|---|---|---|
| `augmentation_v3.txt` | Dialogue augmentation (3 variants per call) | `aug_v3` | Gemini 3.5 Flash | 0.9 |
| `labeling_v1.txt` | Utterance-level risk labeling | `label_v1` | Gemini 3.1 Flash-Lite | 0.0 |
| `checklist_judge_v2/D1.txt` .. `D4.txt` | Checklist judging, one prompt per dimension | `checklist_judge_v2` | Claude Opus 5 | n/a (agent) |
| `checklist_construction/diversification/*` | Checklist construction, question diversification | chk_v2 pool | Claude Opus 5 | n/a (agent) |
| `checklist_construction/elaboration/*` | Checklist construction, question elaboration | chk_v2 pool | Claude Opus 5 | n/a (agent) |
| `checklist_construction/filtering/*` | Checklist construction, question filtering | chk_v2 pool | Claude Opus 5 | n/a (agent) |

Notes:

- The construction prompts produced the chk_v2 question pool (112 questions) from which the final
  48-question chk_v3 was selected by hand; see `../checklist/PROVENANCE.md`.
- The judge answers each checklist question with Yes or No only. Judgments are stored per question,
  so the exclusion threshold can be changed without re-judging.
- A `labeling_v2.txt` variant was written and rejected during development because it missed core
  triggers; the released labels all come from `labeling_v1.txt`.
- The labeling prompt carries the temporal rule (its ABSOLUTE TEMPORAL RULE section): the full call
  is provided in one request, and the model is instructed to judge utterance i on utterances 1..i
  only. See [`../docs/data.md`](../docs/data.md).
