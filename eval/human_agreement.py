"""Judge-vs-human agreement check.

Methodology: the author independently re-read all 33 auto-handled examples
from the last `eval/run_eval.py` run (message + retrieved grounding + draft
reply) and assigned the author's own 1-5 scores on the same 4 axes the
LLM judge uses (groundedness, correctness, tone, actionability), checking
in particular whether the reply actually answers what the customer asked
- not just whether it "sounds right" - since that's the failure mode an
LLM judge is most likely to rubber-stamp past.

This is a full-coverage check (all 33 judged rows), not a random
subsample, because 33 examples is already small enough to review by hand
in full and a subsample would just add sampling noise on top of an
already-small n.

Run: python -m eval.human_agreement
(Requires outputs/eval_details.csv to already exist, i.e. run_eval.py must
have been run first.)
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"

# Human (author) scores: id -> (groundedness, correctness, tone, actionability)
# Assigned by reading each message/grounding/reply independently, see
# eval/human_agreement_notes.md for the reasoning behind the disagreements.
HUMAN_SCORES = {
    14: (5, 5, 5, 5), 69: (5, 5, 5, 5), 197: (5, 5, 5, 5), 105: (5, 5, 5, 5),
    203: (5, 5, 5, 5), 28: (5, 5, 5, 5), 10: (5, 5, 5, 4), 98: (5, 5, 5, 5),
    138: (5, 5, 5, 5), 144: (4, 3, 5, 4), 88: (2, 2, 5, 3), 204: (5, 5, 5, 5),
    41: (4, 3, 5, 4), 35: (3, 5, 5, 5), 87: (5, 5, 5, 5), 193: (2, 4, 5, 4),
    180: (4, 3, 5, 3), 63: (5, 5, 5, 5), 42: (3, 5, 5, 5), 1: (5, 5, 5, 5),
    202: (5, 5, 4, 5), 153: (5, 5, 5, 5), 18: (5, 5, 5, 5), 2: (3, 3, 5, 4),
    81: (5, 5, 5, 5), 115: (5, 5, 5, 5), 12: (5, 5, 5, 5), 24: (5, 5, 5, 5),
    37: (5, 5, 5, 5), 6: (5, 2, 4, 1), 75: (5, 5, 5, 5), 111: (5, 5, 5, 5),
    147: (1, 5, 5, 4),
}
AXES = ["groundedness", "correctness", "tone", "actionability"]


def main():
    with open(OUT_DIR / "eval_details.csv", encoding="utf-8") as f:
        rows = {int(r["id"]): r for r in csv.DictReader(f) if r["judge_scores"]}

    missing = set(HUMAN_SCORES) - set(rows)
    if missing:
        print(f"Warning: {len(missing)} human-scored ids not found in eval_details.csv: {missing}")

    per_axis_diffs = {a: [] for a in AXES}
    exact_match_all_axes = 0
    within_one_pairs = 0
    total_pairs = 0
    n = 0

    for eid, human in HUMAN_SCORES.items():
        if eid not in rows:
            continue
        judge = json.loads(rows[eid]["judge_scores"])
        n += 1
        row_exact = True
        for axis, h_score in zip(AXES, human):
            j_score = judge.get(axis)
            if j_score is None:
                continue
            diff = abs(h_score - j_score)
            per_axis_diffs[axis].append(diff)
            total_pairs += 1
            if diff <= 1:
                within_one_pairs += 1
            if diff != 0:
                row_exact = False
        if row_exact:
            exact_match_all_axes += 1

    mad = {a: (sum(v) / len(v) if v else None) for a, v in per_axis_diffs.items()}
    exact_match_rate_per_axis = {
        a: (sum(1 for d in v if d == 0) / len(v) if v else None) for a, v in per_axis_diffs.items()
    }

    summary = {
        "n_examples": n,
        "exact_match_rate_all_4_axes": exact_match_all_axes / n if n else None,
        "within_1_point_agreement_rate": within_one_pairs / total_pairs if total_pairs else None,
        "mean_absolute_difference_per_axis": mad,
        "exact_match_rate_per_axis": exact_match_rate_per_axis,
    }

    with open(OUT_DIR / "human_agreement.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
