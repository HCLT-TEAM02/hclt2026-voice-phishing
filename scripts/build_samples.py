"""Build the train/valid/test JSONL files from labeled calls.

Implements sections 4.1 and 5.1 of the paper.

Input:  data/labeled/labeled_{id}.json, one per call, each utterance carrying a risk label.
Output: data/final/{train,valid,test}.jsonl.

- Splitting happens at the call level to prevent leakage: the original calls are stratified by
  (genre group x length bucket S/M/L) and divided 0.70 / 0.15 / 0.15 with seed 42.
- Augmented calls (phishing only, id = origin * 10 + {1,2,3}) are added to train only when their
  origin call is in train. Valid and test contain original calls only.
- Augmentation quality filter: a candidate judged "No" on two or more of the 48 checklist questions
  is excluded. The ids are listed in data/quality/excluded_augmentations.csv.
- Samples: cumulative context every k = 5 utterances, e_t = 5t, Y_t = max(y_i for i <= e_t),
  T = floor((N - 1) / 5). The span ending at the last utterance is never emitted as a cumulative
  sample; it is used once as the whole-call sample.
- Speakers are renamed S1/S2/S3 in order of appearance, so speaker naming cannot leak the label.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELED_DIR = ROOT / "data" / "labeled"
OUT_DIR = ROOT / "data" / "final"
EXCLUDED_AUG_CSV = ROOT / "data" / "quality" / "excluded_augmentations.csv"

SEED = 42
K = 5
RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}
SPLIT_NAMES = ("train", "valid", "test")
INSTRUCTION = "다음 통화 내용을 읽고 보이스피싱 여부를 분류하라."

# Genre groups defined by id range.
PHISHING_GROUPS = {
    "text_base_loan_fraud": (200000, 200129),
    "stt_base_loan_fraud": (200130, 200161),
    "text_base_investigation_agency": (200162, 200207),
    "stt_base_investigation_agency": (200208, 200367),
}
NORMAL_GROUPS = {
    "kshopping_as": (100000, 100220),
    "kshopping_payment_exchange_return": (100221, 100579),
    "dasan_public_transport": (100580, 100800),
    "dasan_sewer": (100801, 101021),
    "dasan_general_administration": (101022, 101242),
    "disease_control_other": (101243, 101462),
    "counseling_education": (101463, 102561),
    "travel_agency": (102562, 102781),
    "lg_uplus": (102782, 103140),
    "hana_card": (103141, 103499),
}


def is_phishing(cid: int) -> bool:
    return cid >= 200000


def is_augmented(cid: int) -> bool:
    return cid >= 2000000  # 7 digits marks an augmented call


def origin_of(cid: int) -> int:
    return cid // 10  # an augmentation's origin id


def genre_group(cid: int) -> str:
    groups = PHISHING_GROUPS if is_phishing(cid) else NORMAL_GROUPS
    for name, (lo, hi) in groups.items():
        if lo <= cid <= hi:
            return name
    raise ValueError(f"id {cid} falls in no genre group")


def length_bucket(word_count: int, phishing: bool) -> str:
    if phishing:
        return "S" if word_count <= 239 else ("M" if word_count <= 579 else "L")
    return "S" if word_count <= 240 else ("M" if word_count <= 600 else "L")


def word_count(utts: list[dict]) -> int:
    return sum(len(u["text"].split()) for u in utts)


def clean_text(text: str) -> str:
    """Collapse newlines and tabs inside an utterance so one utterance stays one line."""
    return " ".join(text.split())


def normalize_speakers(utts: list[dict]) -> dict[str, str]:
    order: dict[str, str] = {}
    for u in utts:
        if u["speaker"] not in order:
            order[u["speaker"]] = f"S{len(order) + 1}"
    return order


def split_sizes(n: int) -> dict[str, int]:
    exact = {k: n * RATIOS[k] for k in SPLIT_NAMES}
    sizes = {k: math.floor(exact[k]) for k in SPLIT_NAMES}
    rem = n - sum(sizes.values())
    order = sorted(SPLIT_NAMES, key=lambda k: (-(exact[k] - sizes[k]), SPLIT_NAMES.index(k)))
    for k in order[:rem]:
        sizes[k] += 1
    return sizes


def load_all() -> dict[int, dict]:
    convs: dict[int, dict] = {}
    for path in LABELED_DIR.glob("labeled_*.json"):
        d = json.load(open(path, encoding="utf-8"))
        convs[int(d["conversation_id"])] = d
    return convs


def load_excluded_augmentations() -> set[int]:
    """Ids of augmentations excluded by the quality filter.

    An augmentation is excluded when it is judged "No" on two or more of the 48 checklist
    questions. A single "No" can come from question interpretation or a minor wording issue, so
    those are kept. Candidates without a judgment are absent from the list and therefore pass.
    """
    if not EXCLUDED_AUG_CSV.exists():
        raise FileNotFoundError(
            f"Quality filter list not found: {EXCLUDED_AUG_CSV}\n"
            "See docs/data.md for how this list is produced."
        )
    excluded: set[int] = set()
    with open(EXCLUDED_AUG_CSV, encoding="utf-8") as f:
        header = next(f).strip().split(",")
        if header[0] != "conversation_id":
            raise ValueError(f"unexpected header: {header}")
        for line in f:
            line = line.strip()
            if line:
                excluded.add(int(line.split(",")[0]))
    if not excluded:
        raise ValueError("the quality filter list is empty; check that this is intended")
    return excluded


def stratified_split(originals: list[int], convs: dict[int, dict], rng: random.Random) -> dict[int, str]:
    """Map each original call id to a split, 0.7/0.15/0.15 within every (genre, length) cell."""
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for cid in originals:
        ph = is_phishing(cid)
        length = length_bucket(word_count(convs[cid]["utterances"]), ph)
        cells[(genre_group(cid), length)].append(cid)

    assignment: dict[int, str] = {}
    for cell in sorted(cells):
        ids = sorted(cells[cell])
        rng.shuffle(ids)
        sizes = split_sizes(len(ids))
        i = 0
        for name in SPLIT_NAMES:
            for cid in ids[i : i + sizes[name]]:
                assignment[cid] = name
            i += sizes[name]
    return assignment


def make_rows(cid: int, conv: dict) -> list[dict]:
    """One call becomes its cumulative-context samples plus a single whole-call sample."""
    utts = conv["utterances"]
    n = len(utts)
    spk = normalize_speakers(utts)
    labels = [int(u["label"]) for u in utts]
    conv_label = int(conv["conversation_label"])

    def render(k: int) -> str:
        return "\n".join(f"{spk[u['speaker']]}: {clean_text(u['text'])}" for u in utts[:k])

    rows = []
    # Cumulative-context samples: e_t = 5t, Y_t = max(y_i for i <= e_t), T = floor((N-1)/5).
    #
    # The span ending at the last utterance is never emitted here; the whole-call sample below
    # covers it. Emitting both would duplicate one character-identical sample per call, so e_t is
    # always a multiple of 5 and always below N.
    t_max = (n - 1) // K
    for t in range(1, t_max + 1):
        e = K * t
        y = max(labels[:e])
        rows.append({
            "sample_id": f"conversation_{cid}__p{e}",
            "sample_type": "prefix",
            "parent_conversation_id": cid,
            "conversation_label": conv_label,
            "target_label": y,
            "prefix_end_idx": e,
            "instruction": INSTRUCTION,
            "input": render(e),
            "output": str(y),
        })
    # One whole-call sample; its target is the call-level label.
    rows.append({
        "sample_id": f"conversation_{cid}",
        "sample_type": "conversation",
        "parent_conversation_id": cid,
        "conversation_label": conv_label,
        "target_label": conv_label,
        "prefix_end_idx": None,
        "instruction": INSTRUCTION,
        "input": render(n),
        "output": str(conv_label),
    })
    return rows


def main() -> None:
    convs = load_all()
    all_ids = set(convs)
    originals = sorted(i for i in all_ids if not is_augmented(i))
    augs = sorted(i for i in all_ids if is_augmented(i))
    print(f"loaded {len(all_ids)} calls ({len(originals)} original, {len(augs)} augmented)")

    excluded_aug = load_excluded_augmentations()
    missing = excluded_aug - all_ids
    if missing:
        print(f"warning: {len(missing)} excluded ids are not present under labeled/")
    n_excluded = len([a for a in augs if a in excluded_aug])
    print(f"quality filter: {n_excluded} of {len(augs)} augmentations excluded, {len(augs) - n_excluded} kept")

    rng = random.Random(SEED)
    assign = stratified_split(originals, convs, rng)

    # Originals follow their split assignment; augmentations join train only via a train origin.
    split_convs: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    for cid in originals:
        split_convs[assign[cid]].append(cid)
    dropped_aug = 0       # dropped because the origin call is in valid or test
    dropped_quality = 0   # dropped by the quality filter
    for aug in augs:
        if aug in excluded_aug:
            dropped_quality += 1
            continue
        o = origin_of(aug)
        if assign.get(o) == "train":
            split_convs["train"].append(aug)
        else:
            dropped_aug += 1

    # Leakage check: an origin and its augmentations must never span two splits.
    origin_to_splits: dict[int, set] = defaultdict(set)
    for name in SPLIT_NAMES:
        for cid in split_convs[name]:
            origin_to_splits[cid if not is_augmented(cid) else origin_of(cid)].add(name)
    leaks = {o: s for o, s in origin_to_splits.items() if len(s) > 1}
    assert not leaks, f"leakage detected: {list(leaks.items())[:5]}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'split':6} {'calls':>7} {'cumul':>8} {'whole':>7} {'rows':>8} {'label0':>8} {'label1':>8}")
    grand = 0
    for name in SPLIT_NAMES:
        ids = sorted(split_convs[name])
        # A per-split RNG. Sharing one generator would make valid/test row order depend on how
        # many training calls there are, changing those files whenever train composition changes.
        random.Random(f"{SEED}:{name}").shuffle(ids)  # shuffle row order within the file
        n_pref = n_conv = 0
        lab = Counter()
        with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for cid in ids:
                for row in make_rows(cid, convs[cid]):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if row["sample_type"] == "prefix":
                        n_pref += 1
                    else:
                        n_conv += 1
                    lab[row["target_label"]] += 1
        total = n_pref + n_conv
        grand += total
        print(f"{name:6} {len(ids):6d} {n_pref:8d} {n_conv:6d} {total:8d} {lab[0]:8d} {lab[1]:8d}")
    print(f"{'total':6} {'':>7} {'':>8} {'':>7} {grand:8d}")
    print(f"\naugmentations dropped by quality filter: {dropped_quality}")
    print(f"augmentations dropped for valid/test origin: {dropped_aug}")
    print(f"augmentations used: {len(augs) - dropped_quality - dropped_aug}")
    print(f"written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
