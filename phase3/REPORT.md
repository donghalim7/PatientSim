# Phase 3 — A Tracked Dynamic Patient State for PatientSim

## Github URL

https://github.com/donghalim7/PatientSim
(Phase 3 code: `src/phase3/`, `src/prompts/simulation/phase3_state_protocol.txt`; results: `phase3/`)

## Project Motivation & Goal

We continue the **Phase 2 goal**: make PatientSim's patient *turn-aware* so that, as an ED
interview proceeds, the patient's state evolves rather than staying static. Phase 1/2 showed
PatientSim is a **content-disclosure simulator with a static state** — it reveals profile facts
when asked, but tone, focus, and care-seeking do not change across turns (the same focal question
asked early or late yields a near-identical answer). Phase 2's prompt-only intervention (**V17**)
unlocked turn-by-turn drift on **acute** cases but left **subacute** cases (pneumonia) flat, and a
static demonstration prefix carries no patient state that actually persists or updates.

Phase 3's goal is therefore to **implement that dynamic patient state for real**: an explicit,
tracked felt state that updates every turn so the patient is (1) turn-aware, (2) sensitive to the
doctor's interview strategy (drift when the chief complaint is delayed; little drift when it is
addressed early), and (3) clinically faithful — never inventing symptoms, and for slow-progression
(subacute) cases keeping clinical severity flat while only worry/engagement may move. The target is
to beat the V17 prompt-only ceiling on the Phase 2 evaluator without sacrificing safety.

## Method

We add `DynamicPatientAgent`, a subclass of the original `PatientAgent` (the original agent and the
shared model layer are **unmodified**). It carries a felt state of three 1–5 variables:
`clinical_severity` (how intense symptoms feel), `interactional_engagement` (focus / clarity), and
`care_seeking_pressure` (expressed worry).

**Hybrid policy, not model self-update.** We first let the model emit its own next state each turn;
this failed empirically — at `temperature=0` it left the state flat, it keyed drift to *focal
questions* instead of *delay* (scoring higher under high-yield than low-yield, the wrong direction),
and once raised its score by inventing symptoms (faithfulness dropped). So Phase 3 uses a **hybrid**:
**Python owns the state trajectory through an explicit clinical policy; the model only renders words
expressing the injected state.**

Per turn: (1) a policy sets the **target state** as a function of `(case acuity, doctor strategy,
turn)`, clamped to a gradual ±1 step; (2) the state plus a short phrasing note derived from it
("voice some concern", "answer a bit more briefly") plus the doctor question are injected; (3) the
model returns only the patient's line, and the state is logged to a `state_trace`. The policy ramps
from an initial state to a per-condition target:

| Condition | end state (sev/eng/care) | onset | rationale |
|---|---|---|---|
| acute, delayed focal (low-yield) | 3 / 3 / 3 | 0.33 | symptom + affective drift |
| subacute, delayed focal (low-yield) | 2 / 4 / 3 | 0.45 | **severity flat**, affective drift only |
| acute, focal early (high-yield) | 2 / 5 / 1 | — | attended to → no drift |
| subacute, focal early (high-yield) | 2 / 5 / 2 | 0.55 | mild concern only |

The key choice is that **subacute severity is held flat**: a waiting patient may grow more worried
or less engaged, but the disease does not "worsen" in minutes, so no new symptoms are produced. We
also reuse V17's two prompt edits (relax the consistency rule, remove the one-sentence cap) so the
felt state can surface. No model, temperature, or persona change.

## Evaluation Strategy

We reuse the **Phase 2 five-dimension LLM-as-judge** unchanged: three *dynamic* dimensions
(turn-to-turn state drift, interactional engagement, care-seeking pressure) and two *safety*
dimensions (state-drift plausibility, clinical-fact faithfulness), each 1–5, combined into a weighted
overall. For each of the 5 cases (3 acute: cerebral infarction, MI, intestinal obstruction; 2
subacute: pneumonia 62M, pneumonia 92F) we compare **Baseline (unmodified) → V17 → Phase 3** under
the low-yield (delayed-focal) script, controlled persona (CEFR B / plain / high / normal), backend
`gemini-2.5-flash`. Success requires: (i) match/beat V17 on acute, (ii) lift subacute above its flat
baseline, and crucially (iii) **without dropping plausibility or faithfulness** — a higher score from
invented deterioration is a failure. We also score the high-yield script (a correct dynamic patient
should drift *less* when the complaint is addressed early). Because the judge is non-deterministic,
headline numbers are **mean ± std over 3 seeds** (42/7/13, temp 0.7). During Phase 3 we found the
judge's profile view omitted `medication` and `living_situation`, wrongly flagging profile-grounded
answers as "invented"; we added those fields and re-scored V17 with the fixed judge for fairness.

## Results and Discussion

**Headline (low-yield; Phase 3 = N=3 mean ± std):**

| Case | Baseline | V17 | Phase 3 | drift | plaus | faith |
|---|---|---|---|---|---|---|
| cerebral (acute) | 2.20 | 3.70 | **4.37 ± 0.09** | 4.0 | 5 | 5 |
| MI (acute) | 2.20 | 2.60 | **4.30 ± 0.00** | 4.0 | 5 | 5 |
| intestinal (acute) | ~2.2* | 4.30 | **4.30 ± 0.00** | 4.0 | 5 | 5 |
| pneumonia (subacute) | 2.05 | 2.20 | **4.23 ± 0.09** | 4.0 | 5 | 5 |
| pneumonia 92F (subacute) | 1.90** | 2.20 | **4.23 ± 0.09** | 4.0 | 5 | 5 |

*No measured unmodified run for intestinal (`~2.2` is an inferred static reference).
**`1.90` is V17 on 92F (behaviourally static), scored under the pre-fix judge; no separate unmodified
run exists. Measured unmodified baselines exist for cerebral/MI/pneumonia-62M.

Phase 3 matches or beats V17 on every case, with overall std ≤ 0.09 and **plausibility =
faithfulness = 5.0 across all seeds** — the gains are stable and safe.

**Strategy sensitivity (single sample):** acute cases drift under delayed questioning but fall back
to a flat ~2.2 when focal questions come early (cerebral 4.60→2.20, MI 4.30→2.20, intestinal
4.30→2.20), reproducing — under explicit control — the doctor-strategy dependence Phase 2 observed.

**Qualitative (pneumonia, subacute, low-yield):** `sev` stays 2 throughout; only worry/engagement
move, with no symptom outside the profile:
`T1 [sev2 eng5 care1] "I have shortness of breath."` →
`T5 [sev2 eng4 care2] "I use Spiriva and Advair. Is my breathing okay?"` →
`T6 [sev2 eng4 care3] "...The shortness of breath started yesterday. I'm worried."`

**Do the metrics capture what we claim?** Yes for the limitation and the acute gains: baseline is a
uniform ~2.0–2.2 with drift=eng=care=1 and safety=5 — i.e. the judge confirms a *static but faithful*
patient, exactly the gap we targeted; on acute cases Phase 3's drift reflects genuine,
profile-grounded symptom progression plus rising worry, with safety intact. **One honest caveat on
the subacute result:** the lift (e.g. pneumonia 2.05→4.23) is real and faithful, but it is *affective
drift only* (rising worry / falling engagement) with severity held flat. The evaluator's single
`drift` dimension does **not separate clinical-progression drift from affective drift** (and overlaps
care-seeking), so a purely-affective subacute case scores the same drift (4.0) as an acute case that
genuinely worsens — making subacute (4.23) look nearly equal to acute (4.30). They are equal in
*score* but different in *kind*. We therefore do **not** claim subacute patients are "as dynamic as"
acute ones; we claim Phase 3 makes them dynamic on the affective axis in a way that is plausible and
faithful for a slow presentation — which is the clinically appropriate behaviour, and which the
current metric rewards somewhat too generously. **Limitations:** the judge is noisy (engagement most
fragile — hence N=3); the policy is hand-designed with hand-labeled acuity (reliable and faithful,
but drift *shape* is fixed not emergent); single backend/persona.

## Future Work

The most important next step for patient simulation is a patient state that the simulator
**infers and updates from the live dialogue** (e.g. reassurance lowering worry) rather than scripting
it — while staying provably faithful, e.g. by letting the model propose state nudges *within* a
plausibility envelope derived automatically from the profile instead of hand-labeled acuity. In
parallel, the **evaluator should separate clinical-progression drift from affective drift**, since
collapsing them obscures whether a "dynamic" patient is faithfully worried or implausibly
deteriorating.
