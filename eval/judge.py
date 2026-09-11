"""LLM-as-judge for grounded reply quality. Scores 1-5 on four axes.
See REPORT.md for the human-agreement check on a 30-example subsample."""
import json
import os

from src.gemini_utils import call_with_retry

JUDGE_PROMPT = """You are grading a customer support reply for quality. Score each axis 1-5 (5=best).

Customer message: "{message}"
Historical resolutions this reply should be grounded in:
{grounding}
Drafted reply: "{reply}"

Axes:
- groundedness: does the reply's resolution match the pattern shown in the historical examples (not inventing a new policy)?
- correctness: is the reply factually consistent with the customer's message (no contradictions, right item/order referenced)?
- tone: is it warm, brief, on-brand (matches the historical examples' style)?
- actionability: does the customer know what happens next?

Respond with ONLY JSON: {{"groundedness": <1-5>, "correctness": <1-5>, "tone": <1-5>, "actionability": <1-5>, "rationale": "<one sentence>"}}"""


class GeminiJudge:
    def __init__(self, model_name="gemini-flash-lite-latest"):
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def score(self, message: str, reply: str, grounding: list):
        grounding_str = "\n".join(
            f"- Customer: \"{g['customer_text']}\" -> Agent: \"{g['agent_text']}\"" for g in grounding
        ) or "(none retrieved)"
        prompt = JUDGE_PROMPT.format(message=message, grounding=grounding_str, reply=reply)
        try:
            resp = call_with_retry(lambda: self.model.generate_content(prompt))
            raw = resp.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            return {"groundedness": None, "correctness": None, "tone": None, "actionability": None, "rationale": f"judge error: {e}"}
