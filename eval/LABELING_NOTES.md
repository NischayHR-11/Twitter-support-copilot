# Golden set: sampling & labeling methodology

**Size:** 204 examples (spec asked for 150-250).

**Composition:**
1. **186 stratified samples** (~31 per intent x 6 intents) drawn randomly
   from the generated conversation corpus. Ground-truth intent and
   escalate-worthiness come from the generation metadata, which is exact
   by construction (each template belongs to exactly one intent and one
   escalation-scenario class).
2. **18 hand-authored "hard case" examples**, written directly by the
   author (not sampled from templates), specifically to cover patterns
   templates don't produce well:
   - Multi-intent messages ("check my order AND why was I charged twice")
   - Sarcasm / informal tone that could confuse a keyword-sensitive model
   - Emoji-only or near-empty messages (nothing to ground a reply in)
   - Borderline escalation calls (e.g. a repeated-failure complaint that's
     intent-simple but should still go to a human)
   Each was labeled by hand against the rubric below.

**Labeling rubric (applied by the author):**
- *Intent*: which of the 6 definitions in `src/intents.py` best matches the
  primary actionable request in the message. Multi-intent messages are
  labeled by the intent requiring more specialized handling (e.g. billing
  over order-status).
- *Should escalate*: yes if the message contains (a) a safety/legal/fraud
  signal, (b) an account-security concern, (c) evidence of repeated
  unresolved failure (churn risk a template can't fix), or (d) too little
  information to act on confidently. Otherwise no.

**Known limitation:** because the underlying corpus is synthetic (see
`src/generate_data.py` docstring for why), the "hand-labeling" for the 186
templated rows is really "the label the generator assigned, verified by
the author," not independent human annotation of naturally-occurring
tweets. The 18 hand-authored hard cases are the only rows with genuinely
independent human judgment applied after the fact. This is called out
again in `REPORT.md`.
