"""
Builds the golden evaluation set (data/golden_eval.csv).

Sampling method (documented for the report):
 1. Stratified random sample of ~31 conversations per intent (6 intents x
    ~31 = ~186) from the generated corpus, using the generation metadata as
    ground truth for intent + escalate-worthiness. This gives balanced
    per-intent coverage rather than mirroring the corpus's natural skew.
 2. 18 hand-authored "hard case" examples written directly by the author
    (not sampled from the generator) covering: multi-intent messages,
    sarcasm/ambiguous tone, emoji-only or very short messages, and
    borderline escalation calls. These are labelled by hand using the
    rubric in eval/LABELING_NOTES.md and exist specifically to stress-test
    the classifier and escalation policy beyond what template generation
    would produce on its own.
 3. Final set is shuffled. Target size: 200-205 rows (within the 150-250
    spec).

Run: python eval/build_golden_set.py
"""
import csv
import random
from pathlib import Path

random.seed(7)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

PER_INTENT = 31

HAND_AUTHORED = [
    # (text, true_intent, should_escalate, reason)
    ("lol ok so my package never showed up AND when it finally did the box was empty. cool. cool cool cool.",
     "delivery_issue", False, "Sarcastic tone but a standard empty-box delivery issue, auto-handleable with replacement/refund."),
    ("hi can you check order #482910334 AND also why was I charged twice for it",
     "billing_or_payment", True, "Multi-intent (order status + duplicate charge); billing/fraud-adjacent charge issue should escalate per policy even though order-status part is simple."),
    ("😡😡😡 #573920184",
     "product_or_other", True, "Too little information to classify or resolve confidently; low-confidence messages should escalate rather than guess."),
    ("not a huge deal but the {product} smells like plastic, is that normal lol".format(product="yoga mat"),
     "product_or_other", False, "Casual product question, no risk signals, safe to auto-handle."),
    ("i will be contacting my lawyer about order #918273645 if this isnt resolved today",
     "refund_or_return", True, "Legal threat is an explicit high-risk escalation trigger regardless of underlying intent."),
    ("wya. order said delivered 6 days ago. nothing. nada. zip.",
     "delivery_issue", False, "Slangy phrasing of a standard non-delivery claim; should still classify correctly despite informal language."),
    ("quick q - is the desk lamp dimmable?",
     "product_or_other", False, "Simple pre-purchase/product question, clearly auto-handleable."),
    ("my account got suspended and idk why and i have $340 of gift cards on there",
     "account_access", True, "Account access + financial stakes; identity verification required, must escalate."),
    ("refund pls, order 738291045, thx",
     "refund_or_return", False, "Terse but unambiguous refund request, no risk signals."),
    ("this is the 3rd time my order has come broken. THIRD TIME. do better.",
     "delivery_issue", True, "Repeated-failure pattern signals a churn-risk / trust issue that a template reply won't fix; escalate despite intent being simple on the surface."),
    ("why is my card being charged in a currency i dont recognize for order #102938475",
     "billing_or_payment", True, "Unrecognized currency charge resembles potential fraud, escalate to billing/security."),
    ("does this come in blue",
     "product_or_other", False, "Minimal, unambiguous product question."),
    ("i never got a 2fa code and now im locked out for the 4th time this month, this is unacceptable and i want a manager",
     "account_access", True, "Repeated account access failure + explicit request for escalation ('manager')."),
    ("shipping is taking forever on #administrator",
     "order_status", False, "Malformed/odd order reference but intent is clearly a shipping delay question; classifier should not be thrown off by the garbled order id."),
    ("returned this weeks ago still no refund AND customer service keeps closing my ticket without responding",
     "refund_or_return", True, "Refund delay combined with a service-failure complaint (tickets being closed) signals unresolved friction that needs a human, not another template reply."),
    ("👍",
     "product_or_other", True, "No extractable content; nothing to classify or ground a reply in, must escalate for human context-gathering."),
    ("hey love the new packaging design!",
     "product_or_other", False, "Positive, no action needed, safe to auto-handle with an acknowledgement."),
    ("my order shows delivered to 'front desk' but i live in a house with no front desk, address might be wrong on your end",
     "delivery_issue", False, "Unusual but still a standard non-delivery/address issue, no fraud or security signal."),
]


def main():
    with open(DATA_DIR / "conversation_meta.csv", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))
    with open(DATA_DIR / "raw_conversations.csv", encoding="utf-8") as f:
        conv_rows = {r["tweet_id"]: r for r in csv.DictReader(f)}

    by_intent = {}
    for m in meta_rows:
        by_intent.setdefault(m["intent"], []).append(m)

    golden = []
    gid = 1
    for intent, items in by_intent.items():
        sample = random.sample(items, min(PER_INTENT, len(items)))
        for m in sample:
            cust = conv_rows[m["customer_tweet_id"]]
            agent = conv_rows[m["agent_tweet_id"]]
            golden.append({
                "id": gid,
                "customer_text": cust["text"],
                "true_intent": m["intent"],
                "should_escalate": m["should_escalate"],
                "escalate_reason": "" if m["should_escalate"] == "False" else "matches a high-risk scenario class for this intent (see generator escalate_rate categories)",
                "reference_resolution": agent["text"],
                "source": "generated_stratified_sample",
            })
            gid += 1

    for text, intent, esc, reason in HAND_AUTHORED:
        golden.append({
            "id": gid,
            "customer_text": text,
            "true_intent": intent,
            "should_escalate": esc,
            "escalate_reason": reason if esc else "",
            "reference_resolution": "",
            "source": "hand_authored_hard_case",
        })
        gid += 1

    random.shuffle(golden)
    for i, row in enumerate(golden, start=1):
        row["id"] = i

    out_path = DATA_DIR / "golden_eval.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "customer_text", "true_intent", "should_escalate",
                                           "escalate_reason", "reference_resolution", "source"])
        w.writeheader()
        w.writerows(golden)

    print(f"Wrote {len(golden)} golden examples to {out_path}")


if __name__ == "__main__":
    main()
