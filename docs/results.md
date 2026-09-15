# Results

The evaluation set is the test split: 4,514 rows (574 whole-call + 3,940 cumulative-context).
Augmented calls are used for training only and never appear in valid or test. Predictions are made
by comparing the next-token log probabilities of the candidates "0" and "1" and taking the larger,
with no generated text parsed, so the same rule applies before and after fine-tuning and to all
four models. Voice phishing is the positive class. The cumulative-context samples are the primary
result.

## Table 2. Cumulative-context performance, before and after fine-tuning (%)

| Model | Condition | Precision | Recall | F1 | AUPRC |
|---|---|---:|---:|---:|---:|
| Qwen3.5 | before | 13.93 | 99.35 | 24.44 | 42.18 |
| Qwen3.5 | after | 93.02 | 94.62 | 93.82 | 97.88 |
| Ministral | before | 38.59 | 41.08 | 39.79 | 39.34 |
| Ministral | after | 96.69 | 94.19 | 95.42 | 98.38 |
| EXAONE | before | 14.00 | 99.78 | 24.56 | 54.59 |
| EXAONE | after | 95.16 | 93.12 | 94.13 | 97.93 |
| Mi:dm | before | 53.84 | 98.06 | 69.51 | 90.80 |
| Mi:dm | after | 99.30 | 91.18 | 95.07 | 98.30 |

Before fine-tuning, Qwen3.5 and EXAONE detected most risk samples but also flagged normal samples,
a strong positive bias; Ministral under-detected risk samples; Mi:dm had high recall but low
precision. These differing initial biases follow from the base models' next-token probabilities not
being aligned to the detection task. After fine-tuning on the original and augmented data, all four
models improved consistently in F1 and AUPRC. Mi:dm reached the highest precision and Qwen3.5 the
highest recall, while Ministral was comparatively balanced, so the choice in deployment depends on
whether false positives or false negatives are the priority.

## Table 3. Early detection, before and after fine-tuning (before -> after, number of calls)

Among the 59 phishing calls in the test split, 53 have at least one positive cumulative-context
sample and are analyzed; the other 6, with five or fewer utterances and thus no cumulative-context
sample, are excluded. The sample where the ground-truth label first turns 1 is the first risk
sample, and a positive prediction there counts as immediate detection. One interval is five
utterances; a positive prediction anywhere from the first risk sample through two intervals later
counts as cumulative detection. Positive predictions before the first risk sample are counted as
pre-signal positives, and calls never detected before the call ends are final misses.

| Model | Immediate detection | Within two intervals | Pre-signal positive | Final miss | Reversal |
|---|---:|---:|---:|---:|---:|
| Qwen3.5 | 53 -> 46 | 53 -> 47 | 13 -> 4 | 0 -> 6 | 2 -> 0 |
| Ministral | 8 -> 44 | 22 -> 47 | 0 -> 4 | 20 -> 6 | 19 -> 0 |
| EXAONE | 52 -> 44 | 53 -> 47 | 12 -> 3 | 0 -> 6 | 0 -> 1 |
| Mi:dm | 51 -> 37 | 52 -> 46 | 10 -> 2 | 0 -> 7 | 2 -> 0 |

Before fine-tuning, Qwen3.5, EXAONE, and Mi:dm had high immediate detection but also predicted
positive before any risk signal, reflecting the positive bias seen in Table 2, while Ministral
often detected late or not at all and reversed its judgment frequently. After fine-tuning,
Ministral's detection speed and stability improved markedly, and the other models, though changing
somewhat in immediate detection and final miss, showed clearly fewer pre-signal positives. All four
models detected most calls within a short span after the risk signal appeared and kept their
judgment stable after the first detection. Fine-tuning thus eased the models' initial biases and
reduced unnecessary early warnings while keeping fast and consistent responses to risk signals.
