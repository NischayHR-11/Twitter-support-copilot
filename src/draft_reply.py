"""Drafts a reply grounded in retrieved historical resolutions for similar
past issues. The prompt explicitly instructs the model to base its answer
on the retrieved examples rather than inventing a policy."""
import os

from src.gemini_utils import call_with_retry

DRAFT_PROMPT = """You are drafting a customer support reply for the Twitter handle @AmazonHelp.
Match this brand's real tone: warm, brief (1-3 sentences), uses first-person plural,
signs off with an agent initial tag like " ^AG" at the end.

Here is how this brand has historically resolved similar customer issues (most similar first):
{examples}

Now draft a reply to this NEW customer message. Ground your reply in the
pattern shown above (offer the same kind of resolution style) but don't
copy an example verbatim - adapt it to this specific message.

Customer message: "{message}"

Respond with ONLY the reply text, no quotes, no markdown."""


class GeminiDrafter:
    def __init__(self, model_name="gemini-flash-lite-latest"):
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def draft(self, message: str, retrieved_examples: list):
        if retrieved_examples:
            examples_str = "\n".join(
                f"{i+1}. Customer: \"{ex['customer_text']}\" -> Agent: \"{ex['agent_text']}\" (similarity: {ex['similarity']:.2f})"
                for i, ex in enumerate(retrieved_examples)
            )
        else:
            examples_str = "(no closely similar past resolution found)"

        prompt = DRAFT_PROMPT.format(examples=examples_str, message=message)
        try:
            resp = call_with_retry(lambda: self.model.generate_content(prompt))
            return resp.text.strip()
        except Exception as e:
            return f"[draft generation failed: {e}]"
