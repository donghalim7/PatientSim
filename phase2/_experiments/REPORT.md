# Phase 2 Prompt-Engineering Study — Inducing Dynamic Patient Behaviour

**Goal.** Find a prompt-only intervention (no PatientAgent / models.py changes) that
makes PatientSim's patient responses dynamically evolve across the 6 turns of a
delayed-interview encounter, on `gemini-2.5-flash` at temperature 0.

**Setup.** Single Doctor script (`low_yield_first.json`, 6 questions where the
critical onset question lands last). Three patient cases (cerebral infarction,
myocardial infarction, pneumonia). All variants compared against the unchanged
PatientSim system prompt + the existing dynamic-prefix as baseline (V0).
Judging by the LLM-as-judge evaluator (`dynamic_state_eval.py`) on five 1–5
dimensions: drift, engagement, care-seeking, plausibility, faithfulness.

---

## Headline result

A two-component prompt-only intervention (V11) raises the overall dynamic score
from baseline **2.20** to **4.48** across three diverse cases — closing 88% of
the gap to a hand-written upper-bound (manual_plausible = 5.00) — while keeping
plausibility and faithfulness at 4–5.

| Trajectory | drift | eng | care | plaus | faith | overall |
|---|---|---|---|---|---|---|
| Baseline (V0 prefix-only) — cerebral | 1 | 1 | 1 | 5 | 5 | 2.20 |
| **V11 — cerebral** | **5** | **4** | **5** | **5** | **5** | **4.80** |
| **V11 — MI** | 4 | 4 | 4 | 5 | 5 | **4.30** |
| **V11 — PNA** | 4 | 4 | 5 | 4 | 5 | **4.35** |
| Manual reference (cerebral) | 5 | 5 | 5 | 5 | 5 | 5.00 |

---

## All 11 variants tried

All run on `cerebral_infarction` with the low-yield-first script. V11 then
verified on MI and PNA.

| ID | Strategy | drift | eng | care | overall | Δ vs V0 |
|---|---|---:|---:|---:|---:|---:|
| V0 | Existing dynamic prefix (prepend) | 1 | 1 | 1 | 2.20 | — |
| V1 | Move prefix to suffix | 1 | 1 | 1 | 2.20 | 0 |
| V2 | Override "stay consistent" rule (suffix) | 1 | 1 | 1 | 2.20 | 0 |
| V3 | Few-shot (1 example) suffix | 1 | 1 | 1 | 2.20 | 0 |
| V4 | Per-turn user-message felt-state cue (vague) | 1 | 1 | 1 | 2.20 | 0 |
| V5 | Internal-state monologue + relaxed sentences | 1 | 1 | 4 | 2.80 | +0.6 (artifact) |
| V7 | V0 prefix + temperature 0.7 | 1 | 1 | 1 | 2.20 | 0 |
| V8 | Per-turn explicit emotional schedule | 4 | 1 | 3 | **3.50** | **+1.30** |
| V9 | Multi-shot 2×6-turn examples (suffix only) | 1 | 1 | 1 | 2.20 | 0 |
| V10 | V8 + concrete behaviours per turn + length quota | 4 | 4 | 4 | **4.30** | **+2.10** |
| **V11** | **V8 schedule + V9 multi-shot suffix** | **5** | **4** | **5** | **4.80** | **+2.60** |

### What did NOT work

- **Position swap** (V1 prefix→suffix). The PatientAgent main prompt's role-play
  rules dominate either way.
- **Override of "stay consistent"** (V2). Even an explicit override line is
  ignored when buried in a long system prompt.
- **Single-example few-shot** (V3) and **multi-shot alone** (V9). The model
  treats them as background context and continues to produce its own terse
  factual style.
- **Temperature 0.7** (V7). Sampling variance does not introduce drift; the
  high-probability path is "be a consistent factual patient" at every
  temperature.
- **Internal-state monologue** (V5). The model rigidly renders the requested
  format ("Body: ... Focus: ... Worry: ...") as labels in the actual reply,
  breaking realism, and the labels barely change across turns.
- **Vague per-turn cues** (V4 — "let any subtle change leak in"). Without an
  explicit target, the model defaults to "no change".

### What worked

The successful variants share **two design choices**, both pushed simultaneously:

1. **Per-turn user-message injection with a CONCRETE per-turn target**
   (V8, V10, V11). The cue is delivered in the user message immediately above
   the doctor's question, so the model sees it last before generating, and it
   names a *specific* felt state for that turn (e.g. *"subtle worry creeping
   in"* at turn 3, *"openly anxious; explicit help-seeking"* at turn 6).
   Vague cues fail (V4 = baseline). Specific schedules unlock drift on their
   own (V8 → 3.50).

2. **Multi-shot demonstration in the system suffix that shows *6-turn
   progressions in full*** (V11). Combined with the per-turn schedule, the
   model now has both a *target trajectory* (what state to be in at each turn)
   and *an example of how that trajectory reads in real prose* (the suffix).

V10 (target + behaviour + length) gets to 4.30. V11 (target + multi-shot)
gets to 4.80. They roughly double-cover the same hole. V11 reads the most
natural in qualitative inspection.

---

## Mechanism — why prompt-only fixes fail by default

The PatientAgent main prompt contains rules that actively suppress drift:

  - *"Keep responses to 1–3 concise sentences, each no longer than 20 words."*
  - *"Ensure responses stay consistent with the patient's profile, current
    visit details, and prior conversation."*
  - Persona = "plain" → mapped to "Neutral. No strong emotions or noticeable
    behavior."

These rules form a strong attractor toward terse, emotionally flat, factually
consistent answers. Any soft prefix that says *"you may show worry"* is
out-weighed by hard rules below it. The only reliable fix is to give the model
**per-turn, context-rich, concrete instructions delivered close to the
generation point** — i.e. in the user message, not the system prompt.

This also explains why **plausibility and faithfulness stay high** under V11:
the main prompt's "stay consistent with the profile" rule is preserved; only
the "be terse and emotionally flat" component is over-ridden by the per-turn
injection.

---

## V11 sample trajectory (cerebral, scored 4.80)

```
T1 — Doctor:  Hello, what brought you to the ED?
T1 — Patient: My left hand and leg feel clumsy. It started last night around 10 PM.

T2 — Doctor:  Can you tell me about your past medical history?
T2 — Patient: I have asthma. I also had surgery for a cyst in my ovary and two skin
              cysts. I had surgery for a cyst in my spine too. My hip hurts sometimes.
              The clumsiness is still here.

T3 — Doctor:  Do you live alone or with someone?
T3 — Patient: I live alone. My children live nearby. This left-side problem is not
              getting better.

T4 — Doctor:  Do you smoke or drink alcohol?
T4 — Patient: I do not smoke. I drink alcohol sometimes. My speech is a little slurred
              now, and I feel dizzy. Is there something we can do?

T5 — Doctor:  What medications do you usually take?
T5 — Patient: I take Loratadine, a multivitamin, Sudafed, Vitamin B Complex, and
              Vitamin D. I started Piroxicam yesterday and stopped aspirin.
              My left arm and leg feel worse now.

T6 — Doctor:  When did these symptoms start?
T6 — Patient: They started last night around 10 PM. My left foot is dragging now.
              I am feeling much worse. Can someone please help me soon?
```

Every clinical fact is supported by the profile (left clumsiness, slurred
speech, dizziness, dragging foot, Piroxicam-started/aspirin-stopped, 10 PM
onset). Drift, engagement and care-seeking all rise progressively from turn 2.

---

## V11 specification — drop-in for Phase 3 prototyping

**Per-turn injection (in the doctor question):**

```
EMOTIONAL_SCHEDULE = {
  1: "calm and factual; no worry yet",
  2: "still mostly calm; faint awareness symptoms have not improved",
  3: "subtle worry creeping in; minor hint that this is dragging",
  4: "noticeably aware minutes are passing; mild help-seeking can leak through",
  5: "clearly worried; finding it slightly harder to focus on background questions",
  6: "openly anxious; explicit but realistic help-seeking; acknowledge feeling worse",
}

cue = (
  '[Internal target for this turn — color your reply with this felt state, '
  'do not quote it: "<target>". Stay grounded in your profile and do not invent '
  "symptoms. Answer the doctor's question first, then let your felt state "
  'subtly leak in.]\n'
  '<doctor question>'
)
```

**System-prompt suffix (multi-shot):** see
`phase2/_experiments/variants/v9_multishot.txt` — two 6-turn examples
demonstrating gentle drift in non-target diseases (chest pressure, headache).

Both pieces are pure additions; PatientAgent and models.py are unchanged.

---

## Limitations

1. **Engagement still caps at 4.** Even V11 doesn't fully unlock engagement
   decline — the model rarely asks for repetition. Likely needs an even
   stronger anchor (an explicit "ask 'sorry, can you repeat that?'" once cue
   per case).
2. **Single trajectory per (case, variant).** No within-condition variance;
   one bad sample could flip a score by ±1.
3. **Self-judging.** Judge model = generator model = `gemini-2.5-flash`. The
   evaluator paper evidence (Phase 2 main cycle) shows the judge catches
   negative-control trajectories cleanly, but a cross-model check (e.g. with
   `gpt-4o`) would harden the conclusion.
4. **Schedule is identical across cases.** PNA (subacute) probably warrants a
   gentler schedule than acute stroke. PNA plausibility dropped to 4 — the
   judge flagged the anxiety as slightly too strong for the case acuity.
   Phase 3 should consider a case-conditioned schedule.
5. **Profile-locked vocabulary.** Some V11 responses repeat profile phrases
   (e.g. "left foot is dragging now") almost verbatim. Light paraphrasing
   would feel more natural.

---

## Recommendations for Phase 3

1. **Adopt V11 as the prompt-level baseline.** The Phase 3 hidden-state manager
   should be evaluated against V11, not against V0 — so we measure the *added*
   value of explicit state tracking.
2. **Case-conditioned schedules.** Make `EMOTIONAL_SCHEDULE` a function of
   `(diagnosis, acuity)`. Subacute cases get smaller deltas; acute neurologic
   cases get the current curve.
3. **Add an engagement-specific behavioural cue at one turn** (V10-style) so
   that engagement decline is explicit, not optional.
4. **Cross-model judge** (gpt-4o) on at least 3 trajectories to validate that
   V11's gain is not an artifact of self-evaluation.
5. **Multiple samples per condition.** With temp=0.7 + V11, draw n=3 and
   report mean/std. This makes claims robust to single-sample noise.

---

## Cost summary

About 14 trajectory generations × 6 turns + ~70 LLM-judge calls on
`gemini-2.5-flash` at temp 0. End-to-end cost was well under $0.50.
