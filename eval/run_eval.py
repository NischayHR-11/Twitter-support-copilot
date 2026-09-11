"""Main evaluation harness.

Produces outputs/results.json (headline metrics) and outputs/eval_details.csv
(per-example rows, for auditing/failure analysis).

Two tiers of evaluation, on purpose (see REPORT.md "misleading headline
number"):
  1. Baselines (Trivial, Simple TF-IDF+LogReg) run on the FULL 204-row
     golden set - they're free and instant.
  2. The Gemini classifier + full agent pipeline (classify -> decide ->
     retrieve -> draft -> judge) runs on a fixed random SUBSAMPLE (default
     40) to stay within free-tier rate limits and the 15-minute repro
     budget. All three classifiers are also compared head-to-head on that
     same subsample for a fair apples-to-apples number.

Run: python -m eval.run_eval --sample 40
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

from src.classify import TrivialBaseline, SimpleBaseline, GeminiClassifier
from src.decide import decide
from src.retrieve import HistoryStore
from src.draft_reply import GeminiDrafter
from eval.judge import GeminiJudge

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
SLEEP_BETWEEN_CALLS = 1.5  # pace under free-tier RPM limits


def load_golden():
    with open(DATA_DIR / "golden_eval.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["should_escalate"] = r["should_escalate"] == "True"
    return rows


def prf1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main(sample_size: int, seed: int):
    OUT_DIR.mkdir(exist_ok=True)
    golden = load_golden()
    all_texts = {r["customer_text"] for r in golden}

    print(f"Loaded {len(golden)} golden examples.")

    trivial = TrivialBaseline()
    simple = SimpleBaseline(exclude_texts=all_texts)

    # --- Tier 1: baselines on the FULL golden set (cheap) ---
    full_results = {"trivial": {"correct": 0}, "simple": {"correct": 0}}
    for r in golden:
        if trivial.predict(r["customer_text"])["intent"] == r["true_intent"]:
            full_results["trivial"]["correct"] += 1
        if simple.predict(r["customer_text"])["intent"] == r["true_intent"]:
            full_results["simple"]["correct"] += 1
    n_full = len(golden)
    baseline_full_accuracy = {
        "n": n_full,
        "trivial_accuracy": full_results["trivial"]["correct"] / n_full,
        "simple_accuracy": full_results["simple"]["correct"] / n_full,
    }
    print("Full-set baseline accuracy:", baseline_full_accuracy)

    # --- Tier 2: fixed random subsample, all classifiers + full pipeline ---
    rng = random.Random(seed)
    subsample = rng.sample(golden, min(sample_size, len(golden)))

    gemini_clf = GeminiClassifier()
    history = HistoryStore()
    drafter = GeminiDrafter()
    judge = GeminiJudge()

    details = []
    correct = {"trivial": 0, "simple": 0, "gemini": 0}
    esc_tp = esc_fp = esc_fn = esc_tn = 0
    quality_scores = {"groundedness": [], "correctness": [], "tone": [], "actionability": []}

    for i, row in enumerate(subsample):
        text = row["customer_text"]
        true_intent = row["true_intent"]
        true_escalate = row["should_escalate"]

        t_pred = trivial.predict(text)["intent"]
        s_pred = simple.predict(text)["intent"]
        g_pred = gemini_clf.predict(text)
        time.sleep(SLEEP_BETWEEN_CALLS)

        correct["trivial"] += (t_pred == true_intent)
        correct["simple"] += (s_pred == true_intent)
        correct["gemini"] += (g_pred["intent"] == true_intent)

        decision = decide(text, g_pred["intent"], g_pred["confidence"])
        pred_escalate = decision["escalate"]

        if pred_escalate and true_escalate:
            esc_tp += 1
        elif pred_escalate and not true_escalate:
            esc_fp += 1
        elif not pred_escalate and true_escalate:
            esc_fn += 1
        else:
            esc_tn += 1

        row_detail = {
            "id": row["id"], "customer_text": text, "true_intent": true_intent,
            "true_escalate": true_escalate,
            "trivial_pred": t_pred, "simple_pred": s_pred, "gemini_pred": g_pred["intent"],
            "gemini_confidence": g_pred["confidence"],
            "pred_escalate": pred_escalate, "escalate_reason": decision["reason"],
            "draft_reply": None, "judge_scores": None,
        }

        if not pred_escalate:
            examples = history.top_k(text, k=3, intent_filter=g_pred["intent"])
            reply = drafter.draft(text, examples)
            time.sleep(SLEEP_BETWEEN_CALLS)
            judged = judge.score(text, reply, examples)
            time.sleep(SLEEP_BETWEEN_CALLS)
            row_detail["draft_reply"] = reply
            row_detail["judge_scores"] = judged
            for axis in quality_scores:
                if judged.get(axis) is not None:
                    quality_scores[axis].append(judged[axis])

        details.append(row_detail)
        print(f"[{i+1}/{len(subsample)}] true={true_intent} gemini={g_pred['intent']} escalate={pred_escalate}")

    n_sub = len(subsample)
    esc_precision, esc_recall, esc_f1 = prf1(esc_tp, esc_fp, esc_fn)

    results = {
        "golden_set_size": len(golden),
        "baseline_full_set_accuracy": baseline_full_accuracy,
        "subsample_size": n_sub,
        "subsample_seed": seed,
        "classifier_accuracy_on_subsample": {
            "trivial": correct["trivial"] / n_sub,
            "simple": correct["simple"] / n_sub,
            "gemini": correct["gemini"] / n_sub,
        },
        "escalation_policy": {
            "true_positive": esc_tp, "false_positive": esc_fp,
            "false_negative": esc_fn, "true_negative": esc_tn,
            "precision": esc_precision, "recall": esc_recall, "f1": esc_f1,
        },
        "reply_quality_mean_scores": {
            axis: (sum(v) / len(v) if v else None) for axis, v in quality_scores.items()
        },
        "reply_quality_n": len(quality_scores["groundedness"]),
    }

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(OUT_DIR / "eval_details.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(details[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in details:
            d = dict(d)
            d["judge_scores"] = json.dumps(d["judge_scores"]) if d["judge_scores"] else ""
            w.writerow(d)

    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    main(args.sample, args.seed)
