# Data Construction

## 1. Source Data

### Voice phishing calls

Real call recordings and transcripts were collected from the Financial Supervisory Service (FSS)
voice phishing archive "Voice of the Scammer" (그놈 목소리), published on the
[FSS experience gallery board](https://www.fss.or.kr/fss/bbs/B0000206/list.do?menuNo=200690). The raw collection contains 410 calls:
180 loan fraud and 230 government agency impersonation. 192 calls came with transcripts; 218 were
audio only.

For the audio-only calls, Whisper API and Naver ClovaNote were both applied to a subset and compared
qualitatively. ClovaNote transcribed the dialogue content and speaker turns more accurately and was
adopted as the transcription tool.

**Cleaning rules**

- Existing transcripts: drop calls with fewer than 60 or more than 7,000 whitespace-separated words.
- ASR transcripts: drop calls with fewer than 5 utterances, calls with almost no substantive dialogue,
  and calls with severe noise, missing utterances, or speaker identification errors.
- When multiple speakers were merged into one utterance or a speaker ID was wrong, only the utterance
  boundary and the speaker ID were fixed. The utterance text itself was never edited.

368 phishing calls remained after cleaning.

| | Loan fraud | Agency impersonation | Total |
|---|---:|---:|---:|
| Existing transcripts | 130 | 46 | 176 |
| ASR | 32 | 160 | 192 |
| **Total** | **162** | **206** | **368** |

### Normal calls

Normal calls were collected from three AI-Hub corpora:
[Call Center Q&A](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=98) (민원(콜센터) 질의-응답 데이터),
[Counseling Speech](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=100) (상담 음성), and
[Civil Counseling LLM Pretraining and Instruction Tuning](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71844)
(민간 민원 상담 LLM 사전학습 및 Instruction Tuning 데이터). AI-Hub requires account registration
before download. Ten
counseling domains were selected (shopping, public administration, disease control, education,
travel, telecom, card services, and others) to keep domain diversity.

Normal calls that contain finance and personal-information expressions were deliberately included so
that the model cannot rely on keywords alone. From 43,836 candidates, 3,500 calls were selected to
keep the domain mix while matching the word-count distribution of the phishing calls.

**Final source corpus: 368 phishing + 3,500 normal = 3,868 calls.**

> **Redistribution.** The source call data is not included in this repository. Obtain it directly
> from the FSS and AI-Hub under their respective terms.

## 2. Dialogue Augmentation

Gemini 3.5 Flash generated three candidate variants (prompt: [`../prompts/augmentation_v3.txt`](../prompts/augmentation_v3.txt)) for each of the 368 phishing calls, giving
1,104 candidates.

**Preserved:** scam intent, core tactics, dialogue progression, speaker role relations. Actions that
matter for risk judgment, such as agency impersonation, money demands, app installation, and personal
information requests, are preserved semantically. No new incidents or scam tactics are introduced.

**Varied:** vocabulary, sentence structure, tone, and reaction style.

The three variants per call are: (1) close to the original, (2) with the victim's suspicion
strengthened, (3) with the scammer's speaking style changed.

Quality screening uses the 48-item checklist described in [`checklist.md`](checklist.md). A candidate
judged "No" on **two or more items is excluded**; a single "No" can come from item interpretation
differences or minor stylistic issues and is kept. Of the 1,104 candidates, **917 passed** and became
the final augmented data.

## 3. Utterance-Level Labeling

Each utterance of the original calls and the surviving augmented calls received a context-aware risk
label. Automatic labeling used Gemini 3.1 Flash-Lite (prompt: [`../prompts/labeling_v1.txt`](../prompts/labeling_v1.txt)).

**Temporal rule.** The full call is sent in a single request, and the prompt instructs the model
to judge utterance i using only utterances 1..i, without considering later utterances (see the
ABSOLUTE TEMPORAL RULE section of
[`../prompts/labeling_v1.txt`](../prompts/labeling_v1.txt)). This is an instruction-level
constraint rather than an input-level one; the reliability of the resulting labels is verified
against human raters below.

**Label definition**

| Label | Meaning |
|---|---|
| `1` (Risk Signal) | The current utterance carries information that directly or indirectly indicates an ongoing phishing attempt, given the context so far |
| `0` (No Signal) | The current utterance contributes nothing to the risk judgment |

The label describes the **current utterance's own contribution**, not a cumulative state. Even after
a risk signal appeared earlier, a plain acknowledgment or greeting is labeled `0`.

Risk signals include money demands (transfers, cash handoffs), remote-control or malicious app
installation and phishing site prompts, requests for financial or personal information, and
psychological pressure such as agency impersonation or arrest threats. Judgments consider the
accumulated context rather than keyword presence, and no call-type or speaker metadata is given to
the labeler.

### Annotator validation

300 utterances were sampled with call type and utterance position taken into account. Three human
raters and the LLM labeled them independently under the same definition.

| Metric | Value |
|---|---|
| Human 3-way full agreement | 91.7% |
| Mean pairwise Cohen's kappa (human) | 0.82 |
| Krippendorff's alpha | 0.82 |
| Human majority vs LLM agreement | 95.3% |
| Human majority vs LLM Cohen's kappa | 0.84 |

**Alternative Annotator Test.** To test whether the LLM can replace one human rater, each human was
held out in turn and the held-out human and the LLM were compared on how well each represents the
remaining two raters. The tolerance for skilled annotators, epsilon = 0.15, was applied, and the
repeated tests across the three raters were corrected with the Benjamini-Yekutieli procedure. The
difference between the LLM and the human raters stayed within the tolerance, so the LLM labels were
judged reliable enough to stand in for one human rater under this guideline and sample distribution.

After validation, the same definition and temporal restriction were applied to every original and
augmented call.

## 4. Sample Construction

Define a call as N ordered utterances C = (u_1, ..., u_N), each with a risk label y_i in {0, 1}.

**Cumulative-context samples.** Starting from the beginning of the call, a sample is emitted every
5 utterances:

```
P_t = (u_1, u_2, ..., u_5t),    1 <= t <= T,    T = floor((N-1)/5)
Y_t = max(y_1, ..., y_5t)
```

Because T = floor((N-1)/5), **no cumulative-context sample covers the final utterance.** The full
call is represented once, by the whole-call sample below. Dropping this condition would duplicate
every whole call as an extra cumulative-context sample with identical text.

Y_t = 0 means no risk signal has been observed up to that point; Y_t = 1 means at least one
contributing utterance has appeared.

**Whole-call samples.** One per call, labeled with the call-level ground truth (phishing 1, normal 0).

Training on both sample types teaches the model early judgment during the call and final judgment
after it ends.

### Split statistics

| Split | Rows | Cumulative-context | Whole-call | Normal rows | Phishing rows |
|---|---:|---:|---:|---:|---:|
| train | 26,760 | 23,425 | 3,335 | 19,721 | 7,039 |
| valid | 4,892 | 4,306 | 586 | 4,182 | 710 |
| test | 4,514 | 3,940 | 574 | 3,998 | 516 |

The test split contains 59 phishing calls; 53 of them have at least one cumulative-context sample
with ground-truth label 1 and are usable for the early-detection analysis.

**Split composition.** To prevent leakage, the 3,868 original calls are split at the call level
first, into train 2,708, valid 586, and test 574, and samples are built within each split.
Augmented calls are used only in train: of the 917 that passed quality screening, the 627 generated
from train-split original calls are added, giving a training set of 3,335 calls. The remaining 290
augmentations, whose origin calls fall in valid or test, are excluded to avoid leakage. Splitting is
stratified by genre group and length bucket (seed 42), and an augmented call always follows the
split of its origin call, so valid and test contain original calls only.

## 5. Schema

Each row is one JSON object with 9 fields.

```jsonc
{
  "sample_id": "conversation_200031__p10",  // prefix: conversation_{id}__p{e}; whole call: conversation_{id}
  "sample_type": "prefix",                  // "prefix" | "conversation"
  "parent_conversation_id": 200031,         // call ID (7 digits for augmented calls)
  "conversation_label": 1,                  // call-level label (0 normal, 1 phishing)
  "target_label": 1,                        // ground truth for this sample
  "prefix_end_idx": 10,                     // utterances included; null for whole-call samples
  "instruction": "다음 통화 내용을 읽고 보이스피싱 여부를 분류하라.",
  "input": "S1: ...\nS2: ...",              // speakers normalized to S1/S2/S3 in order of appearance
  "output": "1"                             // str(target_label)
}
```

Training uses `instruction` + `input` (prompt assembly) and `output` (answer). The remaining fields
are metadata for analysis: grouping by `parent_conversation_id` and sorting by `prefix_end_idx`
reconstructs each call's detection timeline.
