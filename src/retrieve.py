"""Retrieves historically-similar resolved conversations to ground reply drafting.
Simple TF-IDF cosine similarity over customer messages - deliberately not an
embedding/vector-DB setup, see REPORT.md decision log for why."""
import csv
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class HistoryStore:
    def __init__(self):
        with open(DATA_DIR / "conversation_meta.csv", encoding="utf-8") as f:
            self.meta = list(csv.DictReader(f))
        with open(DATA_DIR / "raw_conversations.csv", encoding="utf-8") as f:
            self.tweets = {r["tweet_id"]: r for r in csv.DictReader(f)}

        self.conversations = []
        for m in self.meta:
            cust = self.tweets[m["customer_tweet_id"]]
            agent = self.tweets[m["agent_tweet_id"]]
            self.conversations.append({
                "customer_text": cust["text"],
                "agent_text": agent["text"],
                "intent": m["intent"],
            })

        self.corpus = [c["customer_text"] for c in self.conversations]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def top_k(self, query_text: str, k: int = 3, intent_filter: str = None):
        q_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        results = []
        for i in ranked:
            conv = self.conversations[i]
            if intent_filter and conv["intent"] != intent_filter:
                continue
            results.append({**conv, "similarity": float(sims[i])})
            if len(results) >= k:
                break
        return results
