"""Escalation policy: hybrid of hard rules (safety/fraud/legal keywords,
always-escalate intents) and classifier-confidence threshold. Returns a
decision plus a human-readable reason - the assignment explicitly asks for
a *stated* reason, not just a boolean."""
from src.intents import HIGH_RISK_KEYWORDS, CLASSIFIER_CONFIDENCE_ESCALATION_THRESHOLD

# Intents where even a "resolved-sounding" message often hides something a
# template reply shouldn't touch (money movement, identity/security).
ALWAYS_ESCALATE_INTENTS = {"account_access"}


def decide(text: str, intent: str, confidence: float):
    lowered = text.lower()

    matched_keywords = [kw for kw in HIGH_RISK_KEYWORDS if kw in lowered]
    if matched_keywords:
        return {
            "escalate": True,
            "reason": f"High-risk keyword(s) detected: {', '.join(matched_keywords)}. Routed to human for safety/fraud/legal review.",
        }

    if intent in ALWAYS_ESCALATE_INTENTS:
        return {
            "escalate": True,
            "reason": f"Intent '{intent}' involves identity/account security which requires human verification by policy.",
        }

    if confidence < CLASSIFIER_CONFIDENCE_ESCALATION_THRESHOLD:
        return {
            "escalate": True,
            "reason": f"Classifier confidence ({confidence:.2f}) is below the auto-handle threshold ({CLASSIFIER_CONFIDENCE_ESCALATION_THRESHOLD}); too risky to auto-reply.",
        }

    return {
        "escalate": False,
        "reason": f"Intent '{intent}' classified with confidence {confidence:.2f}, no risk signals detected; safe to auto-handle.",
    }
