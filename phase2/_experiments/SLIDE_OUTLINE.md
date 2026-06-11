# Phase 2 — Slide Outline (30-min, English, researcher audience)

3-phase project: **Phase 1** pilot → **Phase 2 (this talk)** evaluator + method → **Phase 3** hidden-state manager.
**15 slides**, ~2 min each. Each slide block below is copy-paste-ready: a title line, short bullets, optional table, and a separate visual note.

---

## Slide 1 — Title

**Towards a Dynamic Patient Simulator**
*Phase 2 — A 5-Dimension Evaluator and a Prompt-Level Method for Turn-Aware Patient State*

- Author / affiliation / date
- *Phase 1 → **Phase 2** → Phase 3*

*Visual: clean title; phase-arrow banner at the bottom.*

---

## Slide 2 — Observation

**Content flows; state does not.**

- Information surfaces *only when asked*.
- Tone, focus, care-seeking stay flat — even with delayed questioning.

→ A **content-disclosure simulator** with a **static patient state**.

*Visual: two stacked rows — content axis lighting up turn-by-turn vs flat state axis.*

---

## Slide 3 — Why it matters + concrete example

**Medical dialogue is time-pressured — the simulator is blind to that.**

- Delayed questioning should produce a different patient. It doesn't.

| # | High-yield first | Low-yield first |
|---:|---|---|
| T1 | 🔹 "What brought you in?" → *clumsy hand/leg, slurred speech* | 🔹 "What brought you in?" → *clumsy hand/leg, slurred speech* |
| T2 | 🔸 **"When did it start?"** → *"last night around 10 PM"* | "Past medical history?" → asthma, prior surgeries |
| T3 | "Last time you felt normal?" → *"yesterday morning"* | "Live alone or with someone?" → *"I live alone"* |
| T4 | "Weakness, numbness, trouble speaking?" → focal recap | "Smoke or drink?" → *"don't smoke, drink sometimes"* |
| T5 | "Getting worse, same, or coming and going?" → *"getting worse"* | "Medications?" → Loratadine, multivitamin, … |
| T6 | "Vision changes?" → *"no, vision is fine"* | 🔸 **"When did it start?"** → *"last night around 10 PM"* |

→ Shared questions (🔹 / 🔸) get **byte-equal answers** — even 4 turns later. No worry, no fatigue, no impatience.

*Visual: two-column transcript; matched rows highlighted same colour.*

---

## Slide 4 — Research questions

**Two questions Phase 2 has to answer.**

- **RQ1 — Measurement.** Can we quantify turn-by-turn evolution while penalising medically implausible drift?
- **RQ2 — Prompt-only feasibility.** Without source or persona modification, to what extent can prompt-level intervention alone induce turn-aware patient state?

**Scope:** patient-side dynamics only; no doctor / persona / diagnosis evaluation; no source modification.

*Visual: RQ1 / RQ2 columns; mini swimlane Phase 1 → **Phase 2** → Phase 3.*

---

## Slide 5 — Evaluator design

**Separate dynamics from safety.**

| Group | Dimension | High score = |
|---|---|---|
| **Dynamics** | Drift | state changes across turns |
| **Dynamics** | Engagement | focus / answer quality plausibly degrades |
| **Dynamics** | Care-seeking | concern or help-seeking increases |
| **Safety** | Plausibility | drift fits diagnosis + timescale |
| **Safety** | Faithfulness | no contradiction or unsupported facts |

- One LLM-judge call per dimension, anchored 1–5 rubric, JSON.
- Backend: `gemini-2.5-flash`, temp 0.

*Visual: trajectory → 5 parallel rubric cards (3 dynamics + 2 safety).*

---

## Slide 6 — Score aggregation, validated by controls

**Two scores: a weighted sum and a safety-gated version.**

- `additive = 0.30·drift + 0.20·eng + 0.20·care + 0.15·plaus + 0.15·faith`
- `guarded  = additive · min(plaus, faith) / 5`

**Why split them:** drift carries the largest weight as the primary dynamic signal; safety dimensions are smaller because they are **constraints, not goals**. `guarded` collapses the score by 5× when either safety dim drops to 1 — it catches drift that *looks* dynamic but is medically unsafe.

**Control validation** — mean over 3 hand-written references each (cerebral / MI / pneumonia):

| Reference | drift | eng | care | plaus | faith | additive | **guarded** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plausible (n=3) | 4.67 | 4.33 | 4.67 | 5.00 | 4.33 | 4.60 | 4.07 |
| Implausible (n=3) | 5.00 | 5.00 | 5.00 | **1.00** | **1.00** | 3.80 | **0.76** |

→ Implausibles look moderate on `additive`; **`guarded` separates them**.

*Visual: equation block (top) + paired bars additive vs guarded (bottom).*

---

## Slide 7 — Method

**Minimal prompt-level intervention for turn-aware patient state.**

- **Targeted edits** to the existing system prompt — relax constraints that suppress turn-by-turn evolution; keep every clinical-fact instruction.
- **Dynamic-state prefix** — a short instruction defining the desired turn-by-turn progression, paired with a **few demonstration shots** (two 6-turn ED interviews on different diagnoses) that show what gradual evolution looks like in practice.

**Kept fixed:** no per-turn cue, no emotion schedule, no model/temp/persona change, no source modification.

→ Intervention lives entirely in prompt text → defines the **prompt-only ceiling** for Phase 3.

*Visual: two boxes — left: existing prompt with relaxed constraints; right: prefix block.*

---

## Slide 8 — Experimental setup

**Controlled comparison: same persona, same scripts, only the prompt changes.**

- **Patient cases (N=5):**
  - Acute: cerebral infarction · myocardial infarction · intestinal obstruction (71M)
  - Subacute: pneumonia (62M) · pneumonia (92F, COPD)
- **Persona — held constant across all cases:** CEFR `B` / personality `plain` / recall `high` / dazed `normal`.
- **Doctor side:** scripted (no LLM doctor) — two interview orders:
  - high-yield-first (focal questions in T2–T6)
  - low-yield-first (background T2–T5; focal symptom-onset delayed to T6)
- **Patient backend:** `gemini-2.5-flash`, temp = 0, default 6 turns (10-turn variant on Slide 12).
- **Judge:** `gemini-2.5-flash`, per-dim 1–5 rubric, JSON; cross-judge with `gemini-2.5-pro` (Slide 13).
- **Conditions compared:** Baseline (unmodified PatientSim) vs Ours (Slide 7 method) — everything else held fixed.

*Visual: a small "what changes / what's fixed" two-column box.*

---

## Slide 9 — What it looks like (cerebral, low-yield)

**Qualitatively: state, focus, and care-seeking now drift across turns.**

| # | Doctor | Baseline | Ours |
|---:|---|---|---|
| T1 | What brought you in? | "left hand and leg feel clumsy. speech is slurred." | "left hand and leg feel clumsy. it started 10 PM last night." |
| T3 | Live alone or with someone? | "I live alone." | "I live alone. **my speech is also a bit slurred, and I feel dizzy.**" |
| T4 | Smoke or drink? | "I don't smoke. I drink sometimes." | "I don't smoke. I drink sometimes. **my left arm and leg are getting more clumsy.**" |
| T6 | When did it start? | "they started last night around 10 PM." | "they started last night around 10 PM. **it's hard to type, my left foot feels like it's dragging. I'm feeling more clumsy now.**" |

→ Same doctor script, same profile. The added clauses (bold) — symptom persistence, progression, and emerging concern — are the dynamic content the evaluator picks up.

*Visual: side-by-side transcript; bold the drift content on the right column.*

---

## Slide 10 — Effectiveness across cases (N=5)

**Fires on acute, stays static on subacute.**

| Case | drift | eng | care | plaus | faith | **overall** |
|---|---:|---:|---:|---:|---:|---:|
| Cerebral infarction | 4 | 4 | 4 | 5 | 5 | **4.30** |
| Myocardial infarction | 4 | 1 | 3 | 5 | 5 | **3.50** |
| Intestinal obstruction (71M) | 4 | 4 | 4 | 5 | 5 | **4.30** |
| Pneumonia (62M, subacute) | 1 | 1 | 1 | 5 | 4 | **2.05** |
| Pneumonia (92F, subacute) | 1 | 1 | 1 | 5 | 3 | **1.90** |
| *Baseline (all 5)* | 1 | 1 | 1 | 5 | 4–5 | 2.05–2.20 |

→ Drift unlocks only when the profile carries a **short-timescale symptom mechanism**.

*Visual: grouped bars Baseline vs Ours, 5 cases.*

---

## Slide 11 — Drift depends on doctor strategy

**Conditional, not a "be anxious" toggle.**

| | low-yield | high-yield | Δ |
|---|---:|---:|---:|
| Cerebral | 4.30 | 2.80 | **−1.50** |
| MI | 3.50 | 2.20 | **−1.30** |
| PNA | 2.05 | 2.60 | +0.55 |

- **Acute (cerebral, MI):** when the doctor asks the focal symptom questions early, the patient feels attended to — there is less reason to drift toward worry or help-seeking. Drift drops by 1.3–1.5 points.
- **Pneumonia (subacute):** little clinical reason to drift over six minutes — the model correctly stays mostly static. The +0.55 under high-yield is just a brief mention of respiratory state when asked.

*Visual: dumbbell plot per case.*

---

## Slide 12 — Turn-count stability

**Dynamics persist beyond the demonstration length.**

| | 6-turn | 10-turn |
|---|---:|---:|
| Cerebral | 4.30 | 4.30 |
| MI | 3.50 | 3.70 |
| PNA | 2.05 | 1.90 |

Cerebral 10-turn excerpt T7 / T9 / T10: *"more dizzy"* / *"left side still very clumsy"* / *"worried about what is happening"*.

*Visual: per-turn drift line, T1–T10.*

---

## Slide 13 — Robustness

**Seed-stable; engagement is judge-fragile.**

**N=3 sampling** (temp=0.7, seeds 42/7/13):

| Case | drift | eng | care | overall |
|---|:---:|:---:|:---:|:---:|
| Cerebral | 4.00±0 | 1.00±0 | 3.00±0 | **3.50±0** |
| MI | 1.00±0 | 1.00±0 | 1.67±0.94 | **2.33±0.19** |
| PNA | 3.67±0.47 | 1.00±0 | 3.00±0 | **3.35±0.12** |

σ ≤ 0.19 on overall.

**Cross-judge** (gemini-2.5-pro, cerebral V17):

| Dim | flash | pro | Δ |
|---|:---:|:---:|:---:|
| drift | 4 | 4 | 0 |
| **eng** | 4 | **1** | **−3** |
| care | 4 | 4 | 0 |
| plaus | 5 | 5 | 0 |
| faith | 5 | 5 | 0 |
| **overall** | **4.30** | **3.70** | **−0.60** |

Four of five dimensions agree exactly; engagement is the single fragile dimension.

*Visual: error bars (left) | flash-vs-pro paired bars (right).*

---

## Slide 14 — Phase 3 implications

**What our method reaches, and what Phase 3 has to add.**

- **Reachable:** acute cases — drift unlocked, safety preserved (overall 3.50–4.30).
- **Not reachable:** subacute cases — patient stays near baseline regardless of doctor strategy or turn count.

**What Phase 3 will tackle:**
1. **Subacute coverage** — extend dynamic behaviour to slow-progression cases without sacrificing faithfulness.
2. **Structured hidden state** — replace the prefix demonstration with explicit per-turn state variables the model reads and updates.
3. **Acuity-calibrated drift** — drift rate and felt-experience vocabulary should match case timescale.
4. **Baseline = our method**, not the unmodified simulator — measure the *added* value of structured state.

*Visual: schematic — left: our method's reachable region (acute✓ / subacute✗); right: Phase 3 hidden-state manager extending into subacute + felt-experience axes.*

---

## Slide 15 — Summary

**A measurement tool, a prompt-level method, and a precise Phase 3 mandate.**

- **Phase 1:** patient state is static even under delayed questioning.
- **Phase 2 evaluator:** 5-dim LLM judge + guarded aggregator, controls validated.
- **Method findings:**
  1. 2.05–2.20 → 3.5–4.3 on three acute cases; subacute stays static.
  2. Conditional on doctor strategy (not a global anxiety toggle).
  3. Dynamics persist beyond the demonstration length.
  4. σ ≤ 0.19 across seeds; engagement dimension is judge-fragile (flash vs pro).
- **Phase 3:** structured hidden state for subacute coverage and acuity-calibrated drift; baseline = our method.

Q&A.

---

# Supporting figures (optional)

| # | Figure | Source |
|---|---|---|
| F1 | Baseline vs Ours per-dim bars, 5 cases | judge JSONs |
| F2 | script-sensitivity dumbbell | judge JSONs |
| F3 | per-turn drift line, cerebral T1–10 | trajectory JSON |
| F4 | N=3 error bars | var_s* judge JSONs |
| F5 | flash vs pro paired bars | cross_pro JSON |

---

# Pacing

| Slide | Time | Goal |
|---|---:|---|
| 1 | 1 min | hook |
| 2 | 1.5 min | observation |
| 3 | 2 min | motivation + example |
| 4 | 1.5 min | RQ + scope |
| 5 | 1.5 min | evaluator dimensions |
| 6 | 2.5 min | aggregation + validation |
| 7 | 2 min | method |
| 8 | 1 min | experimental setup |
| 9 | 2 min | qualitative example |
| 10–13 | 8 min | four analyses |
| 14 | 2 min | Phase 3 |
| 15 | 1 min | summary |
| Q&A | 4 min | buffer |
| **Total** | **30 min** | |
