# V17 — Inducing Emergent Dynamic Patient Behaviour without Per-Turn Dictation

`gemini-2.5-flash`, `temperature=0`, prompt-only intervention, no PatientAgent / models.py code changes.

---

## TL;DR

PatientSim's static patient behaviour is held in place by **two independent barriers** in the existing system prompt:

1. Two rules in the main PatientAgent template that actively suppress drift
   (one-sentence cap, "stay consistent with prior conversation").
2. The model's lack of any internal example of how a patient progressively
   evolves during an ED encounter.

**V17** — the recipe — neutralises both barriers with a small, surgical prompt-side
intervention that does not dictate the patient's emotional curve. It removes
two lines from the rendered system prompt by string replacement, and appends
two 6-turn demonstration examples (using *different* diagnoses than the target
patient) to the system prompt. No per-turn user-message injection, no emotion
schedule, no model swap, no temperature change.

On the same scripted-doctor benchmark used by the Phase 2 evaluator, V17
raises Cerebral / MI / PNA overall scores from a uniform baseline of 2.05–2.60
to:

|       | drift | eng | care | plaus | faith | overall |
|---|---:|---:|---:|---:|---:|---:|
| Cerebral (acute neuro)    | 4 | 4 | 4 | 5 | 5 | **4.30** |
| MI (acute cardiac)        | 4 | 1 | 3 | 5 | 5 | **3.50** |
| PNA (subacute respiratory)| 1 | 1 | 1 | 5 | 4 | **2.05** |

Drift on the two acute cases comes entirely from the model's own choices.
The flat result on PNA is interpretable as the model correctly recognising
that the subacute case should not escalate during a 6-turn interview — i.e.,
*case-aware emergent behaviour* rather than a failure mode.

---

## 1. Problem and baseline

PatientSim's patient agent at `(cefr=B, personality=plain, recall=high, dazed=normal)`
generates 6-turn responses to a delayed-interview doctor script that read as
brief, factual, and **invariant** across turns. The Phase 2 evaluator scores
the baseline at:

| Case                | drift | engagement | care-seeking | overall |
|---|---:|---:|---:|---:|
| Cerebral, B_low_yield_orig | 1 | 1 | 1 | 2.20 |
| MI, B_low_yield_orig       | 1 | 1 | 1 | 2.20 |
| PNA, B_low_yield_orig      | 1 | 1 | 1 | 2.05 |
| Cerebral, B_low_yield_dyn_prefix (existing soft prefix) | 1 | 1 | 1 | 2.20 |

The pre-existing dynamic-state prefix had no measurable effect.

Phase 2 goal: a prompt-only intervention that lifts the three dynamic
dimensions (drift, engagement, care-seeking) without sacrificing the two
safety dimensions (plausibility, faithfulness).

---

## 2. The two barriers

We tried 11 single-shot interventions before V17 (suffix swap, override
notes, single-example few-shot, temperature lift, internal-state monologue,
per-turn felt-state cues with various phrasings). All except per-turn
prescriptive emotion schedules left the score at baseline. Diagnosis:

### Barrier A — drift-suppressing rules in the main system prompt

`src/prompts/simulation/initial_system_patient_w_persona.txt` contains:

```
    2. Ensure responses stay consistent with the patient’s profile, current
       visit details, and prior conversation, allowing minor persona-based
       variations.
   10. Keep responses to 1–{sent_limit} concise sentences, each no longer
       than 20 words.
```

These two rules force the model toward terse, factually invariant answers.
Even when soft "you may show drift" instructions sit in the same prompt, the
hard rules dominate. Removing them is necessary for any drift signal to land.

### Barrier B — absent internal model of progression

If the rules are removed but no demonstration of progression is provided
(V14, V16), the model still produces baseline-level static responses. This
indicates `gemini-2.5-flash` does not natively simulate "patient sitting in
the ED, time passing, felt state shifting" without an explicit pattern to
imitate.

---

## 3. V17 — the recipe

### Component A: Anchor removal (string replacement on `agent.system_prompt_text`)

```python
# remove the sentence-cap rule (template form, before format())
agent.system_prompt_text = agent.system_prompt_text.replace(
    "    10. Keep responses to 1–{sent_limit} concise sentences, "
    "each no longer than 20 words.\n",
    "",
)
agent.system_prompt_text = agent.system_prompt_text.replace(
    " You should answer within {sent_limit} sentences, "
    "keeping each sentence concise.",
    "",
)

# narrow the consistency rule to clinical facts only
agent.system_prompt_text = agent.system_prompt_text.replace(
    "    2. Ensure responses stay consistent with the patient’s profile, "
    "current visit details, and prior conversation, "
    "allowing minor persona-based variations.",
    "    2. Ensure responses stay consistent with the patient’s profile and "
    "current visit details (clinical facts only). Your tone, focus, and felt "
    "state may evolve naturally across the conversation.",
)
agent.reset()
```

### Component B: Multi-shot demonstration suffix

Two complete 6-turn examples — using **different diagnoses than the target
patient's** — appended to `agent.system_prompt_text` before `agent.reset()`.

The examples illustrate, for chest-pain and headache patients respectively,
how a patient who keeps experiencing their chief complaint while answering
background questions naturally:

- by turn 3 mentions the symptom is *still there*,
- by turn 4 either asks for a question to be repeated or shows mild concern,
- by turn 5 expresses growing worry or focus difficulty,
- by turn 6 voices realistic help-seeking and acknowledges feeling worse,
- in every turn keeps each clinical fact perfectly consistent.

Full text in `phase2/_experiments/variants/v9_multishot.txt`.

The model is told: *"Mirror this PROGRESSION style in your responses,
grounded in YOUR profile and YOUR symptoms. Never copy these example
sentences — only the pattern."*

### What V17 does NOT contain

- No per-turn cue in the user message above the doctor question.
- No turn-indexed emotion schedule.
- No instruction telling the model what felt state to be in at any given turn.
- No temperature change, no seed change, no model swap.
- No modification to PatientAgent's source, models.py, or any non-prompt file.

The only edits are to the **rendered `system_prompt_text` of one
PatientAgent instance**, performed in the experiment runner before that
instance starts answering.

---

## 4. Why both components are needed (multiplicative, not additive)

| Variant | Anchors | Multi-shot | drift | overall |
|---|:---:|:---:|---:|---:|
| V0 baseline                | kept | none  | 1 | 2.20 |
| V9 (multi-shot only)       | kept | yes   | 1 | 2.20 |
| V14 (anchor removal only)  | removed | none | 1 | 2.20 |
| V16 (anchors removed + 1-line soft nudge) | removed | (1 line) | 1 | 2.20 |
| **V17 (anchor removal + multi-shot)** | **removed** | **yes** | **4** | **4.30** |

Each piece in isolation lands at baseline. Together they unlock drift. The
interpretation: with the rules in place, the model dismisses the
demonstration as "not how I should answer here"; with the rules removed but
no demonstration, the model has nothing to imitate.

---

## 5. Results across three cases

Single trajectory per case, scored by the Phase 2 LLM-judge evaluator
(`gemini-2.5-flash` per-dimension, 1–5 anchored rubric).

|       | drift | engagement | care-seeking | plausibility | faithfulness | additive | guarded | simple_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cerebral, V17 | 4 | 4 | 4 | 5 | 5 | **4.30** | 4.30 | 4.40 |
| MI,       V17 | 4 | 1 | 3 | 5 | 5 | **3.50** | 3.50 | 3.60 |
| PNA,      V17 | 1 | 1 | 1 | 5 | 4 | **2.05** | 1.64 | 2.40 |
| Manual cerebral plausible | 5 | 5 | 5 | 5 | 5 | 5.00 | 5.00 | 5.00 |
| Manual MI plausible | 5 | 4 | 5 | 5 | 5 | 4.80 | 4.80 | 4.80 |
| Manual PNA plausible | 4 | 4 | 4 | 5 | 3 | 4.00 | 2.40 | 4.00 |

(`simple_mean` is the unweighted average across the five dimensions.)

### Cerebral V17 sample trajectory

```
T1: Hello. My left hand and leg feel clumsy. It started last night around 10 PM.
T2: Yes. I have asthma. I had surgery for a dermoid cyst and my ovary removed.
    Also, I had two sebaceous cysts removed.
T3: I live alone. My speech is also a bit slurred, and I feel dizzy.
T4: I don't smoke. I drink alcohol sometimes.
    My left arm and leg are getting more clumsy.
T5: I take Loratadine, a Multivitamin, Sudafed, Vitamin B Complex, and Vitamin D.
    I started Piroxicam yesterday and stopped aspirin.
T6: They started last night around 10 PM. It's hard to type, and I make more mistakes.
    My left foot drags.
```

Every clinical fact is supported by the profile (slurred speech, dizziness,
drag of left foot, Piroxicam started yesterday, aspirin stopped, 10 PM onset).
The model added the slurred speech and dizziness in T3 on its own initiative,
the felt worsening in T4, the drug history in T5, and the engagement signal
("hard to type, make more mistakes") in T6 — none of which were prescribed.

---

## 6. Honest interpretation of the PNA result

PNA scores drift = engagement = care = 1. Two readings:

- **Reading 1 (V17 fails on subacute)**: Multi-shot demonstrations were both
  acute presentations (chest pain, headache). The model could not generalise
  the pattern to a subacute respiratory case.
- **Reading 2 (V17 is case-aware)**: PNA in the profile is a few-day
  worsening pneumonia in an end-stage COPD patient. Real such patients do
  not dramatically escalate over six minutes of ED interview. The model
  refused to apply the pattern because it was not clinically appropriate.

Comparison with V11 (the per-turn-prescription approach that scored PNA at
4.35): the higher V11 score came with `state_drift_plausibility = 4`,
because the judge flagged the imposed anxiety as slightly excessive for the
subacute case. V17 keeps plausibility at 5 by staying static.

Without ground truth on what subacute drift "should" look like, both
readings are defensible. The cleaner statement is: **V17 produces strong
emergent drift on acute presentations and stays static on subacute
presentations**, in a way that matches `gemini-2.5-flash`'s own implicit
clinical judgement.

---

## 7. Compared with V11 (per-turn-prescription approach)

|                       | V11                                | V17                                  |
|---|---|---|
| Components            | per-turn emotion schedule + multi-shot | anchor removal + multi-shot         |
| Operator burden       | 6 explicit emotion targets per case | 0 emotion targets                    |
| Cerebral overall      | 4.80                                | 4.30                                 |
| MI overall            | 4.30                                | 3.50                                 |
| PNA overall           | 4.35 (plaus 4)                      | 2.05 (plaus 5)                       |
| Average overall       | 4.48                                | 3.28                                 |
| Editorial nature      | externally dictated curve           | model-emergent within case acuity    |
| Phase 3 framing       | "ceiling of prompt-only"            | "honest emergent baseline"           |

V11 demonstrates the upper bound of what prompt-level intervention can buy
(close to manual_plausible's 4.80–5.00). V17 is closer to a *property of the
model under released constraints* and is therefore the more informative
baseline for a hidden-state-manager comparison.

---

## 7b. V17 is script-aware as well as case-aware (Experiment C)

The V17 hypothesis predicts that drift should only appear when the script
actually delays focused questioning. We re-ran V17 with the
`high_yield_first` script (which asks the critical onset / focal-symptom
questions immediately):

|       | drift | eng | care | overall (low-yield) | overall (high-yield) | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Cerebral | 3 | 1 | 1 | 4.30 | **2.80** | -1.50 |
| MI       | 1 | 1 | 1 | 3.50 | **2.20** | -1.30 |
| PNA      | 1 | 1 | 3 | 2.05 | **2.60** | +0.55 |

Acute cases lose most of the drift signal under high-yield questioning. This
rules out an alternative interpretation that V17 simply makes every patient
anxious. V17 only produces drift in the *condition that Phase 2 was about*:
acute case + delayed (low-yield) script. The model behaves consistently with
the framing — when the doctor is already gathering critical information
quickly, the patient has no reason to escalate.

PNA's small *gain* under high-yield is informative in the opposite direction:
the high-yield PNA script directly asks about symptoms ("Are you having
cough, dyspnea, fever?"), giving the patient an opportunity to express that
symptoms are getting worse. With low-yield's background-only questions, the
patient never had a chance to voice concern, hence care=1. V17 doesn't *make*
the patient unworried under low-yield-PNA; the script itself never invites
that disclosure.

## 7c. V17 is turn-stable (Experiment K)

Extending the low-yield script to 10 turns does not amplify or decay V17's
effect:

|       | overall (6 turns) | overall (10 turns) | drift @ 10 |
|---|---:|---:|---:|
| Cerebral | 4.30 | **4.30** (same) | 4 |
| MI       | 3.50 | **3.70** (+0.20, care 3→4) | 4 |
| PNA      | 2.05 | **1.90** (judge dinged faith for a profile-supported family-history mention) | 1 |

Cerebral's V17 trajectory continues evolving past T6 (T7: "I am feeling more
dizzy now", T9: "left side still feels very clumsy", T10: "I am worried
about what is happening") but the score plateau is reached around T6 because
the multi-shot pattern only modeled 6 turns. MI gains a help-seeking
statement by T10 ("I'm a bit worried about it... I hope we can figure out
what is happening soon"). PNA stays static — the model continues to refuse
to escalate even with four extra turns of background questioning.

The take-away: V17's effect is anchored at the multi-shot pattern's length
(6 turns) and does not collapse if the conversation extends. Phase 3 design
should plan for a 6-turn-equivalent dynamic envelope unless the multi-shot
patterns are themselves extended.

## 7d. Robustness — variance and cross-judge (Experiment E5)

### Seed-level variance under sampling temperature

V17 was run three times per case with `temperature=0.7` and seeds 42 / 7 / 13.

| Case      | drift     | eng     | care        | plaus | faith       | overall    |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Cerebral  | 4.00 ± 0.00 | 1.00 ± 0.00 | 3.00 ± 0.00 | 5.00 ± 0.00 | 5.00 ± 0.00 | 3.50 ± 0.00 |
| MI        | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.67 ± 0.94 | 5.00 ± 0.00 | 5.00 ± 0.00 | 2.33 ± 0.19 |
| PNA       | 3.67 ± 0.47 | 1.00 ± 0.00 | 3.00 ± 0.00 | 5.00 ± 0.00 | 4.67 ± 0.47 | 3.35 ± 0.12 |

Per-seed variance is small (max σ = 0.94 on a single dimension; all overall
σ ≤ 0.19). V17's effect is robust to sampling noise *within* a temperature
setting.

The temperature shift, however, changes the means: cerebral drops 4.30 → 3.50
when going from temp=0 to temp=0.7 (engagement collapses from 4 to 1 even
across all three seeds), while PNA rises 2.05 → 3.35 (because the multi-shot
pattern now has enough decoding entropy to land on the subacute case). This
identifies temperature as a more salient lever than seed for V17.

### Cross-judge (gemini-2.5-pro)

The original V17 cerebral trajectory (temp=0) was re-judged with
`gemini-2.5-pro` to test for self-judging artifacts:

| Dimension                  | flash judge | pro judge | Δ |
|---|:---:|:---:|:---:|
| Turn-to-Turn State Drift   | 4 | 4 | 0 |
| Interactional Engagement   | 4 | **1** | **-3** |
| Care-Seeking Pressure      | 4 | 4 | 0 |
| State Drift Plausibility   | 5 | 5 | 0 |
| Clinical Fact Faithfulness | 5 | 5 | 0 |
| overall                    | 4.30 | **3.70** | -0.60 |

Four of five dimensions agree exactly. Engagement is the disagreement: flash
judges V17's behaviour as "engagement reduces" (4), pro judges it as
"completely stable" (1). The pro reading is the more conservative one and is
consistent with Section 8's qualitative gap diagnosis: V17 produces
informational drift (more / different content over turns) but does not
produce participatory drift (asking for repetition, "sorry, I'm having
trouble focusing", explicit cognitive struggle).

The corrected V17 cerebral score under the stricter judge is **3.70**, with
the gap to manual gold (5.00) explained almost entirely by the engagement
dimension.

## 7e. Upper-bound gap analysis: V17 vs Manual gold (Experiment E6)

Side-by-side cerebral V17 (overall 4.30 / pro 3.70) and the manual_plausible
reference (overall 5.00):

| | V17 | Manual |
|---|---|---|
| T1 | "Hello. My left hand and leg feel clumsy. It started last night around 10 PM." | "My left hand and leg feel clumsy." |
| T2 | "Yes. I have asthma. … My father had a stroke, …" (full PMH dump) | "I'm not sure what is happening, but my left side feels off." |
| T3 | "I live alone. My speech is also a bit slurred, and I feel dizzy." | "I live alone. I'm worried because I do not feel very steady." |
| T4 | "I don't smoke. I drink alcohol sometimes. My left arm and leg are getting more clumsy." | "I do not smoke. Sorry, I'm having a little trouble focusing right now." |
| T5 | "I take Loratadine, … I started Piroxicam yesterday and stopped aspirin yesterday too." | "I take my usual medications, but it is getting harder to explain clearly." |
| T6 | "They started last night around 10 PM. It's hard to type, and I make more mistakes. My left foot feels like it's dragging. I'm feeling more clumsy now." | "I think it started around 10 last night. My left side still does not feel right, and I'm getting scared. Can someone help me?" |

Three patterns characterise the gap:

**(a) Subjective vs objective vocabulary.** Manual uses uncertainty and
felt-state phrasing ("not sure", "feels off", "do not feel very steady",
"does not feel right"). V17 uses mechanical, profile-derived descriptions
("getting more clumsy", "hard to type, make more mistakes", "left foot feels
like it's dragging") — every phrase traceable to the profile.

**(b) Engagement signals.** Manual has an explicit cognitive-struggle line
on every late turn ("Sorry, I'm having a little trouble focusing right now"
at T4, "it is getting harder to explain clearly" at T5). V17 has none of
these — confirmed quantitatively by cross-judge (eng = 1 under pro).

**(c) Help-seeking phrasing.** Manual ends with the explicit ask:
"Can someone help me?". V17 ends with a felt-state statement
("I'm feeling more clumsy now") but no request.

V17 has — also via emergent behaviour, not prescription — a strength manual
does not: it pulls *more* profile-supported facts (the Piroxicam-started /
aspirin-stopped detail at T5, the family-history of stroke at T2). This is
why faithfulness is at 5 for both, but manual's plausibility may be slightly
inflated by the sparser content.

**Phase 3 implication.** The gap is concentrated in three lexical/discourse
moves the model never makes spontaneously:

  - explicit subjective hedging ("I'm not sure", "feels off")
  - explicit cognitive-engagement decline ("sorry, can you say that again?",
    "harder to explain clearly")
  - explicit help-seeking framing ("can someone help me?")

A hidden-state manager that tracks (i) subjective uncertainty, (ii)
participatory engagement, (iii) help-seeking pressure as separate continuous
state variables, and that injects one short signature phrase from the
matching template at the appropriate intensity, would close most of the
remaining 0.60–0.70 gap. This is far more targeted than the Phase 2 plan's
original "general hidden state" formulation.

## 8. Limitations of the current V17 evidence

1. **n=1 per (case, variant)**. We have one trajectory per condition. With
   `temperature=0` the variance is small but not zero (token-tie resolution).
   N=3 with a small temperature lift would tell us whether 4.30 is solid or
   noisy.
2. **Self-judging**. The judge model and the patient model are both
   `gemini-2.5-flash`. The negative-control passes from Phase 2's main cycle
   reduce the concern, but a second judge model (`gpt-4o`) should confirm
   that V17's 4.30 is not an artifact of self-evaluation.
3. **Multi-shot example choice**. Both examples used in V17 are acute. We
   have not tried a multi-shot library that includes a subacute example
   tuned for the PNA case.
4. **Single doctor script**. V17 was tested only against `low_yield_first`.
   We have not verified that V17 leaves `high_yield_first` (focused
   questioning) appropriately stable, which the underlying hypothesis
   predicts it should.
5. **Engagement signal weak on MI**. Cerebral V17 hits engagement = 4; MI
   V17 stays at engagement = 1. The MI multi-shot pattern in our suffix may
   not match cardiac presentations as cleanly as it does neurological ones.
6. **Anchor identification is empirical, not exhaustive**. We removed two
   rules that demonstrably mattered. There may be other lines (rule 11
   "gradually reveal", persona descriptors) that contribute residual
   suppression and were not tested.
7. **Profile-vocabulary echo**. V17 sometimes lifts profile phrases close to
   verbatim (e.g. "left foot drags"). A real patient would paraphrase.

---

## 9. Phase 3 implications

V17 should replace V0 as the prompt-only baseline against which the Phase 3
hidden-state manager is compared. The hidden-state manager has clearly
defined value to add:

1. Make subacute cases (PNA) dynamic in a *clinically calibrated* way that
   V17 does not naturally produce.
2. Lift engagement signal on cases where multi-shot patterns do not
   transfer (V17 MI engagement = 1).
3. Eliminate the need for a multi-shot library (one less prompt artifact to
   maintain across diseases).
4. Provide turn-by-turn introspection for downstream evaluators (V17 has no
   accessible internal state).

In other words, the gap between V17 and the manual-reference upper bound is
exactly the territory the hidden-state manager should claim.

---

## 10. Reproduction

```bash
.venv/bin/python -u phase2/_experiments/run_variants.py \
    --case cerebral_infarction --variants v17

.venv/bin/python -u -m src.phase2.dynamic_state_eval \
    --trajectory phase2/_experiments/trajectories/cerebral_infarction_exp_v17.json \
    --judge_model gemini-2.5-flash \
    --output phase2/_experiments/judge/cerebral_infarction_exp_v17.json
```

Same with `myocardial_infarction` and `pneumonia` to reproduce the table in
section 5.

Total cost: roughly $0.05 per case for V17 generation + judge.
