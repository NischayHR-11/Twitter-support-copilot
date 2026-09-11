"""Intent taxonomy for the support agent. Kept deliberately small (6 classes)
- see REPORT.md decision log for why these six and not Banking77's 77."""

INTENT_DEFINITIONS = {
    "order_status": "Customer is asking where their order is, shipping/tracking status, or when it will arrive. No damage/loss/missing item yet.",
    "delivery_issue": "Order arrived damaged, missing, wrong item, or marked delivered but never received.",
    "refund_or_return": "Customer wants to return an item, cancel an order, or is asking about a refund's status/amount.",
    "billing_or_payment": "Duplicate charges, unrecognized charges, promo code issues, or payment/charge disputes.",
    "account_access": "Login issues, password resets, account suspension, 2FA problems, or suspected account compromise.",
    "product_or_other": "Pre-purchase product questions, compliments, warranty questions, or anything not covered above.",
}

INTENT_LABELS = list(INTENT_DEFINITIONS.keys())

# Risk signals that force escalation regardless of classifier confidence.
# See REPORT.md decision log for rationale on a hybrid rule+LLM policy
# instead of a pure LLM "should I escalate?" call.
HIGH_RISK_KEYWORDS = [
    "lawyer", "legal action", "sue", "attorney", "lawsuit",
    "fraud", "unauthorized", "hacked", "hack", "compromised",
    "suicide", "kill myself", "self harm",
    "manager", "third time", "3rd time", "again and again",
]

CLASSIFIER_CONFIDENCE_ESCALATION_THRESHOLD = 0.6
