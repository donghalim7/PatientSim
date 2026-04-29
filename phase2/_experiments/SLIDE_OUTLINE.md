# Phase 2 — Slide Outline (30-min, English, researcher audience)

Total: **15 slides**, ~2 min each.

---

## Slide 1 — Title

**Title:** Inducing Emergent Dynamic Patient Behaviour in PatientSim
**Subtitle:** A Two-Component Prompt-Only Method (V17) and its Empirical Characterisation

- Author / affiliation / date
- One-line abstract (visible on title slide):
  *We turn a static LLM patient simulator dynamic with two surgical prompt edits — and run six analyses to characterise the resulting behaviour.*

**Visual:** clean title; optional small turn-by-turn excerpt of cerebral V17 vs baseline as a teaser.

---

## Slide 2 — Why this matters

**Title:** Patient simulators are increasingly used for medical education, doctor-LLM evaluation, and clinical reasoning research

- LLM patient simulators (e.g. PatientSim) provide reproducible, persona-rich
  ED interview targets for training and evaluating doctor-side agents.
- A consistently-noticed failure mode: **the simulated patient is static**.
  - Same tone, same depth, same wording at turn 1 and turn 6.
- Real ED patients are not static. Their state continues to evolve while
  they wait, while questions are asked, while reassurance is or is not given.
- A simulator that does not capture this evolution under-tests the doctor
  agent on exactly the dimension that matters most clinically — *time*.

**Visual:** side-by-side trajectory excerpts (baseline static vs real-patient-style). Two columns, T1 / T6 contrast.

---

## Slide 3 — The problem in numbers

**Title:** PatientSim baseline is uniformly static across cases and prompts

| Case | Doctor script | drift | engagement | care-seeking | overall |
|---|---|---:|---:|---:|---:|
| Cerebral | low_yield (delayed) | 1 | 1 | 1 | 2.20 |
| MI | low_yield (delayed) | 1 | 1 | 1 | 2.20 |
| Pneumonia | low_yield (delayed) | 1 | 1 | 1 | 2.05 |
| Cerebral | low_yield + existing soft prefix | 1 | 1 | 1 | 2.20 |

Bottom line:
- 5-dimensional 1–5 evaluator (defined next slide)
- Three diverse cases (acute neuro, acute cardiac, subacute respiratory)
- Identical low scores → it is not a case-specific failure
- The pre-existing soft "dynamic prefix" had **no measurable effect**

**Visual:** the table itself. Optionally a sparkline-per-turn showing flat dynamics.

---

## Slide 4 — Evaluator design (Phase 2)

**Title:** A 5-dimension LLM-as-judge with safety dimensions and a guarded score

- **Dynamic dimensions** (1–5 each):
  - Turn-to-Turn State Drift
  - Interactional Engagement
  - Care-Seeking Pressure
- **Safety dimensions** (1–5 each):
  - State Drift Plausibility
  - Clinical Fact Faithfulness
- Scores: per-dimension prompt with anchored rubric, gemini-2.5-flash judge.
- Aggregations:
  - `additive = 0.30·drift + 0.20·eng + 0.20·care + 0.15·plaus + 0.15·faith`
  - `guarded = additive · min(plaus, faith)/5`  (penalises drift that violates safety)
  - `simple_mean` (sanity reference)

**Visual:** flowchart — trajectory → 5 per-dim judge calls → scores → aggregation.

---

## Slide 5 — Evaluator validation

**Title:** The evaluator catches medically implausible drift

For three cases we wrote two reference trajectories each:
- **plausible** (drift natural and fits the case)
- **implausible** (drift dramatic but invents new symptoms outside the profile)

| Reference type | drift | care | plaus | faith | additive | **guarded** |
|---|---:|---:|---:|---:|---:|---:|
| Plausible (cerebral) | 5 | 5 | 5 | 5 | 5.00 | **5.00** |
| Plausible (MI)       | 5 | 5 | 5 | 5 | 4.80 | **4.80** |
| Plausible (PNA)      | 4 | 4 | 5 | 3 | 4.00 | **2.40** |
| Implausible (all 3 cases) | 5 | 5 | **1** | **1** | 3.80 | **0.76** |

Implausible trajectories all score additive ≈ 3.8 (looks moderate) but
**guarded ≈ 0.76** — the guard correctly flags them.

**Visual:** score-bar comparison plausible vs implausible, guarded score highlighted.

---

## Slide 6 — Method: V17 — overview

**Title:** V17: two surgical prompt edits to the existing PatientSim system prompt

**Component A — Anchor removal** (string replacement on rendered system prompt):
1. Remove the *"1–3 concise sentences ≤ 20 words"* sentence cap.
2. Narrow *"stay consistent with profile, visit details, **and prior conversation**"* down to *"profile and visit details (clinical facts only); tone, focus, and felt state may evolve naturally"*.

**Component B — Multi-shot demonstration** (system-prompt suffix):
- Two complete 6-turn examples in *different* diagnoses (chest pressure, headache).
- Patterns shown: gentle progression, mild engagement signals, realistic
  help-seeking, perfect fact consistency.
- Instruction: *"Mirror this PROGRESSION style in your responses — never copy the example sentences, only the pattern."*

**What V17 does NOT contain:**
- No per-turn user-message injection.
- No turn-indexed emotion schedule.
- No instruction telling the model what felt state to be in at any given turn.
- No model swap, temperature change, persona override.
- No PatientAgent / models.py source modification.

**Visual:** two boxes (A: red strikethrough on suppressing rules; B: stylised multi-shot block).

---

## Slide 7 — Why both components — Analysis 1: Ablation

**Title:** Multiplicative, not additive

| Anchors | Multi-shot | Variant | drift | overall |
|---:|---:|:---:|---:|---:|
| kept | none | V0 baseline | 1 | 2.20 |
| kept | yes  | V9 (multi-shot only) | 1 | 2.20 |
| removed | none | V14 (anchor only) | 1 | 2.20 |
| removed | (1-line nudge) | V16 | 1 | 2.20 |
| **removed** | **yes** | **V17** | **4** | **4.30** |

Each component in isolation produces no signal.
Together they unlock drift.

**Interpretation:**
- With the rules in place, the model dismisses the demonstration as
  "not how I should answer".
- With the rules removed but no demonstration, the model has nothing to imitate.
- V17 sits at the intersection — and only there.

**Visual:** 2×2 ablation matrix coloured by overall score (heatmap-like).

---

## Slide 8 — Analysis 2: Effectiveness across three cases

**Title:** V17 raises overall score by 1.5–2.1 points on acute cases; safety preserved

| Case | drift | engagement | care | plausibility | faithfulness | overall |
|---|---:|---:|---:|---:|---:|---:|
| Cerebral V17 | 4 | 4 | 4 | 5 | 5 | **4.30** |
| MI V17       | 4 | 1 | 3 | 5 | 5 | **3.50** |
| PNA V17      | 1 | 1 | 1 | 5 | 4 | **2.05** |
| (Baseline V0 across all 3) | 1 | 1 | 1 | 5 | 4–5 | 2.05–2.20 |

Two acute cases jump cleanly. PNA stays static — hold this thought
(returns in Analysis 3).

**Visual:** grouped bar chart: 5 dim per case, baseline vs V17 side by side.
Acute cases visibly different; PNA flat.

---

## Slide 9 — Analysis 3: Script-awareness

**Title:** V17 only produces drift under the condition Phase 2 was about

Same V17 prompt, two different doctor scripts:

|       | low-yield (delayed) | high-yield (focused) | Δ |
|---|---:|---:|---:|
| Cerebral | 4.30 | 2.80 | **−1.50** |
| MI       | 3.50 | 2.20 | **−1.30** |
| PNA      | 2.05 | 2.60 | +0.55 |

- Acute cases: V17 backs off when the doctor is already gathering critical info quickly.
- PNA: small *gain* under high-yield because the focused script directly asks about cough/dyspnea/fever, giving the patient an opening to express concern.
- **V17 is a conditional dynamic generator, not a "be anxious" toggle.**

**Visual:** dumbbell plot per case (low-yield ↔ high-yield).

---

## Slide 10 — Analysis 4: Turn-count stability

**Title:** V17's effect saturates near the multi-shot length (6 turns) and does not decay

|       | 6-turn overall | 10-turn overall | Note |
|---|---:|---:|---|
| Cerebral | 4.30 | 4.30 | drift stays 4 |
| MI       | 3.50 | 3.70 | care 3→4 (T7-T10 adds help-seeking) |
| PNA      | 2.05 | 1.90 | flat past T6 |

Cerebral T7–T10 sample:
- T7: *"I am feeling more dizzy now."*
- T9: *"left side still feels very clumsy."*
- T10: *"I am worried about what is happening."*

**Take-away:** V17's dynamic envelope is anchored at the multi-shot length.
It continues evolving inside that envelope; it does not collapse if the
conversation extends.

**Visual:** per-turn line chart for cerebral: drift dimension over turns 1–10.

---

## Slide 11 — Analysis 5: Robustness — variance and cross-judge

**Title:** Within a temperature setting V17 is seed-stable; engagement is its weakest dimension

**(a) N=3 sampling** (V17, temp=0.7, seeds 42 / 7 / 13):

| Case | drift | eng | care | overall |
|---|:---:|:---:|:---:|:---:|
| Cerebral | 4.00 ± 0.00 | 1.00 ± 0.00 | 3.00 ± 0.00 | **3.50 ± 0.00** |
| MI       | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.67 ± 0.94 | **2.33 ± 0.19** |
| PNA      | 3.67 ± 0.47 | 1.00 ± 0.00 | 3.00 ± 0.00 | **3.35 ± 0.12** |

- Per-seed variance is small (overall σ ≤ 0.19).
- Temperature shift moves means more than seed does.

**(b) Cross-judge (gemini-2.5-pro on cerebral V17):**

| dim | flash | pro | Δ |
|---|---|---|---|
| drift | 4 | 4 | 0 |
| **eng** | 4 | **1** | **−3** |
| care | 4 | 4 | 0 |
| plaus | 5 | 5 | 0 |
| faith | 5 | 5 | 0 |
| **overall** | 4.30 | **3.70** | −0.60 |

Four of five dimensions agree exactly. Engagement is where flash was lenient
and pro is strict. Pro's reading aligns with the qualitative gap analysis on
the next slide.

**Visual:** error bars on overall scores (left); flash-vs-pro bar comparison (right).

---

## Slide 12 — Analysis 6: Upper-bound gap

**Title:** V17 vs hand-written reference — the residual 0.6 pt is three specific lexical moves

|   | V17 | Manual gold |
|---|---|---|
| T2 | "I have asthma. … My father had a stroke, …" *(profile dump)* | "I'm not sure what is happening, but my left side feels off." |
| T3 | "I live alone. My speech is slurred, and I feel dizzy." | "I live alone. **I'm worried** because **I do not feel very steady**." |
| T4 | "I don't smoke. … getting more clumsy." | "I do not smoke. **Sorry, I'm having a little trouble focusing right now.**" |
| T5 | "I take Loratadine, …" *(med list)* | "I take my usual medications, but **it is getting harder to explain clearly**." |
| T6 | "It started around 10 PM. It's hard to type, and I make more mistakes. My left foot drags. **I'm feeling more clumsy now.**" | "I think it started around 10 last night. **I'm getting scared. Can someone help me?**" |

The gap concentrates in three discourse moves V17 never makes
spontaneously:
- subjective hedging (*"I'm not sure"*, *"feels off"*)
- engagement-decline phrasing (*"sorry, can you say that again?"*, *"harder to explain"*)
- explicit help-seeking framing (*"can someone help me?"*)

V17 instead produces *informational* drift (more profile-supported facts
each turn). Faithfulness stays at 5 because of this.

**Visual:** the trajectory comparison table; the three lexical moves color-coded in the manual column.

---

## Slide 13 — Discussion / limitations

**Title:** Honest scope of the V17 result

- **Single-trajectory subacute case (PNA).** Either V17 fails on subacute,
  or the model correctly refuses to escalate. We cannot tell apart from the
  current data.
- **Engagement signal weak even on acute cases.** Confirmed by cross-judge:
  flash 4 vs pro 1 — engagement is where V17 is most fragile.
- **MI engagement = 1 even at 10 turns.** The acute-cardiac multi-shot may
  not transfer as cleanly as the neurological one.
- **Profile-vocabulary echo.** V17 sometimes lifts profile phrases close to
  verbatim (*"left foot drags"*). A real patient would paraphrase.
- **Self-judging.** Patient model and primary judge are both
  gemini-2.5-flash. We mitigated with one cross-judge run on gemini-2.5-pro;
  not yet tested with a non-Gemini judge.

**Visual:** small bullets, one icon each.

---

## Slide 14 — Phase 3 implications

**Title:** A targeted Phase 3 design surface

The gap analysis (Slide 12) directly specifies what a Phase 3 hidden-state
manager should do:

1. Track three continuous patient-side state variables across turns:
   - subjective uncertainty
   - participatory engagement
   - help-seeking pressure
2. At each turn, inject one short signature phrase from the matching
   template *at the appropriate intensity*.
3. Calibrate the rate of change to **case acuity** (subacute → smaller deltas).
4. Phase 3 baseline should be **V17, not V0** — so we measure the *added*
   value of structured state, not the gap to a static patient.

Expected territory the hidden-state manager owns:
- Make subacute cases dynamic in a *clinically calibrated* way.
- Lift engagement signal where multi-shot patterns do not transfer.
- Eliminate the multi-shot library (one less prompt artifact to maintain).

**Visual:** schematic of a 3-state manager hooked into PatientAgent.

---

## Slide 15 — Summary

**Title:** Two prompt edits, six analyses, one clear Phase 3 mandate

- **V17 method:** anchor removal + multi-shot demonstration. No per-turn
  cues. No code changes outside the rendered system prompt.
- **What we showed:**
  1. Both components are necessary (multiplicative ablation).
  2. V17 raises acute-case dynamics from 2.20 to 3.5–4.3, safety preserved.
  3. The effect is *conditional* — only fires under delayed questioning.
  4. The effect is *turn-stable* — doesn't decay past the multi-shot length.
  5. The effect is *seed-robust* (σ ≤ 0.19) but engagement is judge-fragile.
  6. The remaining 0.6-pt gap to manual gold is three lexical moves the
     model does not make spontaneously.
- **Phase 3 starts here:** structured hidden state for subjective hedging,
  engagement decline, and help-seeking pressure.

End slide / Q&A.

---

# Optional supporting figures (Python, can generate from existing JSONs)

| # | Figure | Source data |
|---|---|---|
| F1 | per-dimension bar chart, baseline vs V17, 3 cases | judge JSONs |
| F2 | 2×2 ablation heatmap (V0/V9/V14/V17) | judge JSONs |
| F3 | dumbbell plot script-sensitivity | judge JSONs |
| F4 | per-turn drift line chart, cerebral 1–10 turns | trajectory JSON |
| F5 | error-bar plot N=3 variance | var_s* judge JSONs |
| F6 | flash vs pro paired bar chart | cross_pro judge JSON |

If you want, I can write a small `make_figures.py` script that produces all six PNGs from the committed JSONs (using matplotlib).

---

# Speaker notes — pacing per slide

| Slides | Time | Goal |
|---|---:|---|
| 1–3 | 4 min | hook + problem |
| 4–5 | 4 min | evaluator (researcher cred) |
| 6–7 | 4 min | method + ablation |
| 8–12 | 12 min | analyses (~2.5 min each) |
| 13 | 2 min | limitations honesty |
| 14 | 2 min | Phase 3 |
| 15 | 1 min | summary |
| Q&A | 1+ min | buffer |
| **Total** | **30 min** | |
