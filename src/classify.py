"""Three intent classifiers, cheapest to most capable:
 - TrivialBaseline: always predicts the majority class.
 - SimpleBaseline: TF-IDF + Logistic Regression trained on the generated
   corpus (with golden-eval rows excluded from training to avoid leakage).
 - GeminiClassifier: few-shot LLM classification with the intent
   definitions in the prompt, returns a label + self-reported confidence.
"""
import csv
import json
import os
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.intents import INTENT_DEFINITIONS, INTENT_LABELS
from src.gemini_utils import call_with_retry

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_training_pairs(exclude_texts=None):
    exclude_texts = exclude_texts or set()
    with open(DATA_DIR / "conversation_meta.csv", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    with open(DATA_DIR / "raw_conversations.csv", encoding="utf-8") as f:
        tweets = {r["tweet_id"]: r for r in csv.DictReader(f)}

    texts, labels = [], []
    for m in meta:
        text = tweets[m["customer_tweet_id"]]["text"]
        if text in exclude_texts:
            continue
        texts.append(text)
        labels.append(m["intent"])
    return texts, labels


class TrivialBaseline:
    name = "trivial_majority_class"

    def __init__(self):
        texts, labels = _load_training_pairs()
        self.majority = Counter(labels).most_common(1)[0][0]

    def predict(self, text: str):
        return {"intent": self.majority, "confidence": 1.0}


class SimpleBaseline:
    name = "tfidf_logreg"

    def __init__(self, exclude_texts=None):
        texts, labels = _load_training_pairs(exclude_texts)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, labels)

    def predict(self, text: str):
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        idx = proba.argmax()
        return {"intent": self.model.classes_[idx], "confidence": float(proba[idx])}


CLASSIFY_PROMPT = """You are an intent classifier for a customer support system.
Classify the customer's message into exactly one of these intents:

{intent_list}

Message: "{message}"

Respond with ONLY a JSON object, no markdown, no explanation:
{{"intent": "<one of the intent names above>", "confidence": <float 0-1, your genuine confidence>}}"""


class GeminiClassifier:
    name = "gemini_few_shot"

    def __init__(self, model_name="gemini-flash-lite-latest"):
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def predict(self, text: str):
        intent_list = "\n".join(f"- {name}: {desc}" for name, desc in INTENT_DEFINITIONS.items())
        prompt = CLASSIFY_PROMPT.format(intent_list=intent_list, message=text)
        try:
            resp = call_with_retry(lambda: self.model.generate_content(prompt))
            raw = resp.text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            intent = parsed.get("intent", "product_or_other")
            if intent not in INTENT_LABELS:
                intent = "product_or_other"
            confidence = float(parsed.get("confidence", 0.5))
            return {"intent": intent, "confidence": confidence}
        except Exception as e:
            return {"intent": "product_or_other", "confidence": 0.0, "error": str(e)}
