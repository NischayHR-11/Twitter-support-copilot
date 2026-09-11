"""End-to-end orchestration: classify -> decide -> (retrieve + draft if
auto-handled). Escalated tickets don't get a drafted reply - see REPORT.md
decision log for why we don't draft for escalations."""
from src.classify import GeminiClassifier
from src.decide import decide
from src.retrieve import HistoryStore
from src.draft_reply import GeminiDrafter


class SupportAgent:
    def __init__(self):
        self.classifier = GeminiClassifier()
        self.history = HistoryStore()
        self.drafter = GeminiDrafter()

    def handle(self, message: str):
        cls = self.classifier.predict(message)
        intent, confidence = cls["intent"], cls["confidence"]

        decision = decide(message, intent, confidence)

        result = {
            "message": message,
            "intent": intent,
            "confidence": confidence,
            "escalate": decision["escalate"],
            "escalate_reason": decision["reason"],
            "draft_reply": None,
            "grounding": [],
        }

        if not decision["escalate"]:
            examples = self.history.top_k(message, k=3, intent_filter=intent)
            result["grounding"] = examples
            result["draft_reply"] = self.drafter.draft(message, examples)

        return result


if __name__ == "__main__":
    agent = SupportAgent()
    for msg in [
        "my order #123456789 arrived damaged, the blender is broken in the box",
        "i think my account was hacked, orders i didnt place are showing up",
    ]:
        print(agent.handle(msg))
