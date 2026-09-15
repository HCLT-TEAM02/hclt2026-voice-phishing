# Method Walkthrough

This page describes how the pipeline was run, stage by stage, so the method can be followed and
reimplemented. The call data is not released (see [`data.md`](data.md)), so this is a description of
the procedure rather than a turnkey script.

```
1. Source collection and cleaning
        |
        v
2. Augmentation and quality screening
        |
        v
3. Utterance-level labeling and sample construction
        |
        v
4. Fine-tuning and evaluation
```

## Stage 1: Source data

Voice phishing calls come from the FSS voice phishing experience gallery, normal calls from three
AI-Hub counseling corpora. Both sources restrict redistribution, so the corpus is not included here.
[`data.md`](data.md) records the sources, the cleaning rules, and the resulting composition.

## Stage 2: Augmentation and quality screening

1. Generate three candidate augmentations per phishing call with Gemini 3.5 Flash, using
   [`../prompts/augmentation_v3.txt`](../prompts/augmentation_v3.txt).
2. Judge every candidate against the 48-item checklist
   ([`../checklist/chk_v3.jsonl`](../checklist/chk_v3.jsonl)) with Claude Opus 5, using the
   per-dimension prompts in [`../prompts/checklist_judge_v2/`](../prompts/checklist_judge_v2/).
3. Exclude candidates judged "No" on two or more questions. Of 1,104 candidates, 917 passed.

Judgments are recorded per question and per candidate, so the threshold can be changed and the
filter recomputed without re-judging.

## Stage 3: Labeling and sample construction

Label each utterance with Gemini 3.1 Flash-Lite using
[`../prompts/labeling_v1.txt`](../prompts/labeling_v1.txt). The temporal rule in that prompt is
essential: the full call is provided, and the model is instructed to judge utterance i on
utterances 1..i only.

Split at the call level first (stratified by genre group and length bucket, seed 42), then build the
cumulative-context samples (every 5 utterances, T = floor((N-1)/5)) and one whole-call sample per
call within each split. Augmented calls follow their origin call's split and never enter valid or
test.

[`../scripts/build_samples.py`](../scripts/build_samples.py) performs this step. It reads the
labeled calls from `data/labeled/`, applies the quality filter listed in
`data/quality/excluded_augmentations.csv`, and writes `data/final/{train,valid,test}.jsonl`:

```bash
python scripts/build_samples.py
```

On the corpus used in the paper it reports 917 of 1,104 augmentations kept and the split sizes given
in [`data.md`](data.md).

## Stage 4: Fine-tuning and evaluation

Each model is fine-tuned with QLoRA under the same data, objective, and hyperparameters, listed in
[`experiments.md`](experiments.md). The code is in this repository:
[`../scripts/train_lora.py`](../scripts/train_lora.py) and
[`../scripts/evaluate_model.py`](../scripts/evaluate_model.py), with per-model settings under
[`../configs/`](../configs/). Evaluation applies the identical input format and decision rule before
and after fine-tuning: the next-token log probabilities of "0" and "1" are compared directly, with no
generated text parsed.

Table 2 is computed from each model's evaluation output over the cumulative-context samples.

Table 3 is the early-detection analysis, produced by
[`../scripts/analyze_early_detection.py`](../scripts/analyze_early_detection.py):

```bash
python scripts/analyze_early_detection.py \
    --test data/final/test.jsonl \
    --predictions outputs/finetuned_eval_<model>/predictions.jsonl
```

It orders each phishing call's cumulative-context samples, takes the first sample whose gold label
is 1 as the first risk sample, and reports immediate detection, detection within two intervals,
pre-signal positives, final misses, and reversals. Predictions are joined to the test metadata by
the unique `sample_id` rather than row order.

## Validation experiments

| Experiment | Sample |
|---|---|
| Augmentation quality, human vs LLM agreement | 50 augmentations x 48 questions = 2,400 judgments |
| Utterance labels, human vs LLM agreement | 300 utterances |
| Alternative Annotator Test | the same 300 utterances, epsilon = 0.15 |

Human judgments were collected with a browser-based labeling tool in which raters cannot see each
other's answers.
