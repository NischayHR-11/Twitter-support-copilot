"""Shared helper: retry once on rate-limit (429) errors instead of silently
defaulting, since a bare except previously masked quota errors as
low-confidence predictions and quietly corrupted eval results."""
import time


def call_with_retry(fn, retries=2, backoff_seconds=20):
    last_err = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if "429" in str(e) and attempt < retries:
                time.sleep(backoff_seconds)
                continue
            raise last_err
