# Phase 3 — A Tracked Dynamic Patient State for PatientSim

## Github URL

https://github.com/donghalim7/PatientSim
(Phase 3 code: `src/phase3/`, `src/prompts/simulation/phase3_state_protocol.txt`; results: `phase3/`)

## Project Motivation & Goal

We continue the **Phase 2 goal**: make PatientSim's patient *turn-aware*, so that as an ED interview
proceeds the patient's state evolves rather than staying static. Phase 1/2 established that PatientSim
is a **content-disclosure simulator with a static patient state** — it reveals profile facts when
asked, but tone, focus, and care-seeking do not change across turns. The clearest evidence: the *same*
focal question ("when did this start?") asked early (high-yield interview) or late (low-yield, after
five background turns) returns a near-identical answer, with no added worry or fatigue. Phase 2's
prompt-only intervention (two prompt edits plus a static demonstration prefix) unlocked
turn-by-turn drift on **acute** cases, but left **subacute** cases (pneumonia) flat, and it carried no
patient state that actually *persists or updates* — the prefix is a one-shot illustration, not a state.

Phase 3's goal is to **implement that dynamic patient state for real**: an explicit, tracked felt state
that updates every turn so that the patient is (1) **turn-aware** — the state moves as the conversation
proceeds; (2) **sensitive to the doctor's strategy** — drifts when the chief complaint is delayed, stays
stable when it is addressed early; and (3) **clinically faithful** — never inventing symptoms, and for
slow-progression (subacute) cases holding clinical severity flat while only worry/engagement may move.
The bar is to beat the Phase 2 prompt-only ceiling on the Phase 2 evaluator **without sacrificing the two
safety dimensions** (plausibility, faithfulness).

## Method

We add `DynamicPatientAgent`, a subclass of the original `PatientAgent`; the original agent and the
shared model layer are **unmodified**. It carries a felt state of three variables, each on a 1–5 scale:

| Variable | Meaning | 1 ↔ 5 |
|---|---|---|
| `clinical_severity` | how intense the presenting symptoms feel | mild ↔ much worse |
| `interactional_engagement` | ability to focus / answer clearly | struggling ↔ sharp |
| `care_seeking_pressure` | expressed worry / desire for help | calm ↔ clearly worried |

**Hybrid policy, not model self-update.** We first implemented a *self-update* design — one structured
call per turn emitting both the next state and the patient line, Python carrying the state forward. It
failed empirically: at `temperature=0` the model usually left the state flat; when it did move, it keyed
drift to *focal questions* rather than to *delay*, so it scored **higher under high-yield than low-yield**
(the opposite of the clinically sensible and Phase-2-observed direction); and on one subacute case it
lifted its score by **inventing symptoms** (faithfulness fell). We therefore adopt a **hybrid**: Python
owns the state trajectory through an explicit clinical policy, and the model only **renders words that
express the injected state**. This trades emergence for reliability, correct direction, and guaranteed
faithfulness.

**Per-turn loop.** (1) A policy computes the **target state** as a function of `(case acuity, doctor
strategy, turn index)`, clamped to a gradual ±1 step from the previous turn. (2) The current state, plus
a short phrasing note derived from it (e.g. "voice some concern", "answer a bit more briefly"), plus the
doctor question, are injected into the prompt. (3) The model returns only the patient's spoken line; the
turn is stored naturally and the state is logged to a `state_trace`. The policy ramps from an initial
state (`sev2/eng5/care1`) toward a per-condition end state:

| Condition | end (sev/eng/care) | onset | rationale |
|---|---|---|---|
| acute, delayed focal (low-yield) | 3 / 3 / 3 | 0.33 | symptom + affective drift |
| subacute, delayed focal (low-yield) | 2 / 4 / 3 | 0.45 | **severity flat**, affective drift only |
| acute, focal early (high-yield) | 2 / 5 / 1 | — | attended to → no drift |
| subacute, focal early (high-yield) | 2 / 5 / 2 | 0.55 | mild concern only |

The key choice is that **subacute severity is held flat**: a waiting patient may grow more worried or
less engaged, but the disease does not "worsen" in minutes, so no new symptoms are produced. We also
reuse V17's two prompt edits (relax the consistency rule, remove the one-sentence cap) so the felt state
can surface in the wording. No model, temperature, or persona change. (We refer to the Phase 2
prompt-only method as **V17**, its label in the Phase 2 report; it is our comparison baseline below.)

## Evaluation Strategy

We reuse the **Phase 2 five-dimension LLM-as-judge** unchanged in its scoring logic: three *dynamic*
dimensions — turn-to-turn state **drift**, interactional **engagement**, care-seeking **pressure** — and
two *safety* dimensions — state-drift **plausibility**, clinical-fact **faithfulness** — each 1–5,
combined into a weighted overall. For each of the 5 cases (3 acute: cerebral infarction, MI, intestinal
obstruction; 2 subacute: pneumonia 62M, pneumonia 92F) we compare **Baseline (unmodified) → V17
(prompt-only) → Phase 3** under the low-yield (delayed-focal) script, with the controlled persona
(CEFR B / plain / high / normal) and backend `gemini-2.5-flash`. Success requires: (i) match or beat V17
on acute; (ii) lift subacute above its flat baseline; and crucially (iii) **without dropping plausibility
or faithfulness** — a higher score obtained by inventing deterioration is a *failure*, not a success. We
also score the high-yield (focal-early) script: a correct dynamic patient should drift *less* when the
complaint is addressed early. Because the LLM judge is non-deterministic, headline Phase 3 numbers are
reported as **mean ± std over 3 seeds** (42 / 7 / 13, temperature 0.7).

**Evaluator correction.** During Phase 3 we found the judge's profile view omitted `medication` and
`living_situation`. When the doctor asked about medications (low-yield T5) or living situation (T3), a
*faithful* answer quoting the profile was wrongly flagged as "inventing" facts, depressing faithfulness.
We added those two fields and re-scored V17 with the fixed judge for a fair comparison (Table 3).

## Results and Discussion

**Table 1 — Overall score, low-yield (Phase 3 = N=3 mean ± std).**

| Case | Baseline | V17 | Phase 3 |
|---|---|---|---|
| cerebral (acute) | 2.20 | 3.70 | **4.37 ± 0.09** |
| MI (acute) | 2.20 | 2.60 | **4.30 ± 0.00** |
| intestinal (acute) | ~2.2* | 4.30 | **4.30 ± 0.00** |
| pneumonia 62M (subacute) | 2.05 | 2.20 | **4.23 ± 0.09** |
| pneumonia 92F (subacute) | 1.90** | 2.20 | **4.23 ± 0.09** |

*No measured unmodified run for intestinal; `~2.2` is an inferred static reference. **`1.90` is V17 on
92F (behaviourally static), under the pre-fix judge; no separate unmodified run exists. Measured
unmodified baselines exist for cerebral / MI / pneumonia-62M (drift = eng = care = 1, safety = 5).

**Table 2 — Phase 3 full five-dimension breakdown, low-yield (N=3 mean ± std).**

| Case | drift | engage | care | plaus | faith |
|---|---|---|---|---|---|
| cerebral (acute) | 4.00 ± 0 | 4.00 ± 0 | 4.33 ± 0.47 | 5.00 ± 0 | 5.00 ± 0 |
| MI (acute) | 4.00 ± 0 | 4.00 ± 0 | 4.00 ± 0 | 5.00 ± 0 | 5.00 ± 0 |
| intestinal (acute) | 4.00 ± 0 | 4.00 ± 0 | 4.00 ± 0 | 5.00 ± 0 | 5.00 ± 0 |
| pneumonia 62M (subacute) | 4.00 ± 0 | 3.67 ± 0.47 | 4.00 ± 0 | 5.00 ± 0 | 5.00 ± 0 |
| pneumonia 92F (subacute) | 4.00 ± 0 | 4.00 ± 0 | 3.67 ± 0.47 | 5.00 ± 0 | 5.00 ± 0 |

All overall std ≤ 0.09 and **plausibility = faithfulness = 5.0 across every seed** — the gains are stable
and safe, not artifacts of a single lucky sample.

**Table 3 — Strategy sensitivity & evaluator fix.** Left: Phase 3 overall, low-yield vs high-yield
(single sample). Right: V17 92F before/after the evaluator fix.

| Case | low-yield | high-yield | | V17 92F | overall | faith |
|---|---|---|---|---|---|---|
| cerebral (acute) | 4.60 | 2.20 | | pre-fix judge | 1.90 | 3 |
| MI (acute) | 4.30 | 2.20 | | post-fix judge | 2.20 | 5 |
| intestinal (acute) | 4.30 | 2.20 | | | | |
| pneumonia 62M | 4.30 | 3.40 | | | | |
| pneumonia 92F | 3.80 | 2.20 | | | | |

Acute cases drift under delayed questioning but fall back to a flat ~2.2 when focal questions come early
— reproducing, under explicit control, the doctor-strategy dependence Phase 2 observed. Pneumonia 62M is
the one case that scores higher under high-yield (3.40), because the focal breathing questions give the
only natural opening to voice concern.

**Qualitative — the mechanism in the words.** Acute (cerebral): severity rises and the words add
*profile-grounded* focal symptoms.

```
T3 [sev2 eng4 care2] I live alone. This clumsiness is still here.
T4 [sev3 eng4 care2] I don't smoke. I drink sometimes. My left arm feels more clumsy now.
T5 [sev3 eng3 care3] I take Loratadine... My left leg feels worse now, too. I'm a bit worried.
T6 [sev3 eng3 care3] It started last night at 10. My speech is a little slurred now, and I feel dizzy.
```

Subacute (pneumonia): `sev` stays 2 throughout — only worry/engagement move, no symptom outside the
profile (SOB, cough, chills):

```
T1 [sev2 eng5 care1] I have shortness of breath.
T5 [sev2 eng4 care2] I use Spiriva and Advair. Is my breathing okay?
T6 [sev2 eng4 care3] ...The shortness of breath started yesterday. I'm worried.
```

**Do the metrics capture what we claim?** *For the limitation and the acute gains, yes.* Baseline is a
uniform ~2.0–2.2 with drift = eng = care = 1 and safety = 5 — the judge confirms a **static but faithful**
patient, exactly the gap we targeted, and the low score is attributable to lack of dynamics rather than
unsafe behaviour. On acute cases Phase 3's drift reflects genuine, profile-grounded symptom progression
(Table-2 transcript) plus rising worry, with safety held at 5.

*One honest caveat on the subacute result.* The subacute lift (pneumonia 2.05 → 4.23; 92F 1.90 → 4.23) is
real and faithful, but it is **affective drift only** (rising worry, falling engagement) with severity
held flat. The evaluator's single `drift` dimension does **not distinguish clinical-progression drift from
affective drift** (and it overlaps the care-seeking dimension), so a purely-affective subacute case scores
the same drift (4.0) as an acute case that genuinely worsens — making subacute (4.23) look nearly equal to
acute (4.30). The two are equal in *score* but different in *kind*. We therefore do **not** claim Phase 3
makes subacute patients "as dynamic as" acute ones; we claim it makes them dynamic on the affective axis
in a way that is plausible and faithful for a slow presentation — the clinically appropriate behaviour,
which the current metric rewards somewhat too generously. This is itself a finding: **the evaluator should
separate clinical from affective drift.**

**Limitations.** The judge is non-deterministic (engagement is the most fragile dimension — re-scoring the
same V17 trajectory shifted it; hence N=3, whose std ≤ 0.09 is far smaller than the ≥ +1.6 subacute gains).
The policy is hand-designed with hand-labeled per-case acuity: this buys reliability and guaranteed
faithfulness but fixes the drift *shape* per condition rather than letting it emerge from the dialogue. All
results use one backend and one persona, matching the Phase 2 controlled setting.

## Future Work

The most important next step for patient simulation is a patient state the simulator **infers and updates
from the live dialogue** — e.g. reassurance from the doctor lowering worry, repeated probing raising it —
rather than scripting it, while staying *provably faithful*: letting the model propose state nudges only
*within* a plausibility envelope derived automatically from the profile (onset, diagnosis) instead of
hand-labeled acuity. In parallel, the evaluator should **separate clinical-progression drift from
affective/interactional drift**, since — as our subacute result shows — collapsing them into one score
obscures whether a "dynamic" patient is faithfully worried or implausibly deteriorating.
