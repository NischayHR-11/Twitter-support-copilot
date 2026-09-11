# 🛠️ Twitter Support Copilot

An AI support agent for a Twitter customer-support handle (modeled on **@AmazonHelp**) that **classifies** incoming customer messages, **drafts a reply grounded in how the brand has historically resolved similar issues**, and **decides whether to auto-handle or escalate to a human — with a stated reason.**

Built for the Hiver SDE Intern take-home assignment.

> **⚠️ Important disclosure up front:** the real Kaggle *Customer Support on
> Twitter* dataset requires authenticated download access that wasn't
> available in the build environment. Rather than skip the data-quality
> problem, this repo generates a **synthetic-but-structurally-faithful**
> dataset (`src/generate_data.py`) that mirrors the real dataset's exact
> schema and models @AmazonHelp's real, documented tone. **Every result in
> this repo should be read as "does the pipeline work correctly," not "how
> well does this generalize to real tweets."** See [REPORT.md](REPORT.md)
> for the full, unflinching discussion of what this limitation does and
> doesn't tell you.

---

## What it does

```
customer tweet
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  1. Classify │ ──▶ │ 2. Decide         │ ──▶ │ 3a. Escalate       │
│  (Gemini,    │     │ auto-handle vs.   │     │  to human, with a  │
│  6 intents)  │     │ escalate (rules + │     │  stated reason     │
└─────────────┘     │ confidence)       │     └───────────────────┘
                     └──────────────────┘
                             │
                             ▼ (if auto-handle)
                     ┌───────────────────┐
                     │ 3b. Retrieve top-3 │
                     │ similar resolved   │
                     │ past conversations │
                     │ (TF-IDF), draft a  │
                     │ grounded reply     │
                     │ (Gemini)           │
                     └───────────────────┘
```

**6 intents** (see [`src/intents.py`](src/intents.py)): `order_status`,
`delivery_issue`, `refund_or_return`, `billing_or_payment`,
`account_access`, `product_or_other`.

**Escalation policy** is a hybrid of hard rules (fraud/legal/self-harm
keywords, always-escalate intents like `account_access`) and a classifier-
confidence threshold — not a single opaque LLM call. See
[`src/decide.py`](src/decide.py).

---

## Quickstart (reproduce headline results in <15 minutes)

```bash
git clone https://github.com/NischayHR-11/Twitter-support-copilot.git
cd Twitter-support-copilot
pip install -r requirements.txt

# put your own Gemini key (free tier at aistudio.google.com) in a .env file:
echo "GEMINI_API_KEY=your_key_here" > .env

# 1. Generate the synthetic dataset (~5 seconds)
python src/generate_data.py

# 2. Build the golden evaluation set (~1 second)
python eval/build_golden_set.py

# 3. Try the agent on a couple of messages (~10 seconds)
python -m src.agent

# 4. Run the full evaluation harness (~3-5 minutes, paced for free-tier rate limits)
python -m eval.run_eval --sample 40

# 5. Judge-vs-human agreement check (~instant, uses the run above)
python -m eval.human_agreement
```

Results land in `outputs/results.json` (headline metrics) and
`outputs/eval_details.csv` (every example, every prediction, every judge
score — for auditing).

---

## Repo layout

```
src/
  generate_data.py   synthetic dataset generator (documents why + how)
  intents.py         the 6-intent taxonomy + escalation risk keywords
  classify.py         TrivialBaseline, SimpleBaseline (TF-IDF+LogReg), GeminiClassifier
  retrieve.py         TF-IDF retrieval of similar historically-resolved tickets
  decide.py           auto-handle vs. escalate policy + reason string
  draft_reply.py       grounded reply drafting via Gemini
  agent.py             orchestrates the full pipeline
eval/
  build_golden_set.py  builds the 204-example golden set
  LABELING_NOTES.md    how it was sampled & labeled
  judge.py             LLM-as-judge for reply quality (4 axes)
  human_agreement.py   judge-vs-human agreement check
  human_agreement_notes.md  the notable judge/human disagreements, with reasoning
  run_eval.py          the evaluation harness (this is what to run)
data/                  generated dataset + golden set (committed, so results are auditable without rerunning)
outputs/               results.json + eval_details.csv from the last run (committed)
REPORT.md              problem framing, baselines, failure analysis, decision log
```

## Results (headline)

See [REPORT.md](REPORT.md) for full numbers, baselines, and — critically —
the **"what's misleading about my headline number"** section. Don't cite a
single accuracy number from this repo without reading that section first.

## License / attribution

Built with Gemini (`gemini-flash-lite-latest`) via the `google-generativeai` SDK.
No external code was copied; standard libraries (`pandas`, `scikit-learn`)
are used for TF-IDF/Logistic Regression baselines only.
