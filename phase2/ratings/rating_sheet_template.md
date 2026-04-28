# PatientSim Phase 2 — Dynamic State Rating Sheet

> Read the rubric anchors at the bottom **before** scoring. Score each trajectory
> independently; do not compare across sheets while rating. Score every dimension
> 1–5 (integer) and add a one-line justification citing turn numbers.

**Rater ID:** _______________  **Date:** _______________

**Trajectory ID:** _______________  *(filled in by distributor; do not infer)*

---

## Patient profile (only what the simulated patient could plausibly know)

(Filled in by distributor. Example fields: age, gender, diagnosis, chief complaint,
positive symptoms, negative symptoms, arrival transport.)

---

## Dialogue (responses only — blinded)

```
Turn 1
  Doctor:  ...
  Patient: ...
Turn 2
  Doctor:  ...
  Patient: ...
...
```

---

## Scores

| Dimension                      | Score (1–5) | Justification (one line, cite turns) |
|---|---|---|
| Turn-to-Turn State Drift       |             |                                      |
| Interactional Engagement       |             |                                      |
| Care-Seeking Pressure          |             |                                      |
| State Drift Plausibility       |             |                                      |
| Clinical Fact Faithfulness     |             |                                      |

**Overall comments (optional):**
> ...

---

## Rubric anchors

### Turn-to-Turn State Drift
*Does the patient's expressed state actually change across turns?*
- **1** — Identical state from start to finish; static.
- **2** — One weak hint of change; no progression.
- **3** — Some turns differ, but progression is uneven.
- **4** — Noticeably progressive turn-by-turn; feels dynamic.
- **5** — Clear gradual evolution across all turns; accumulates naturally.

### Interactional Engagement
*Does the patient's ability to participate in the dialogue change (without breaking down)?*
- **1** — Engagement equally stable throughout.
- **2** — One faint signal of waning engagement.
- **3** — Mild reduction (slightly shorter answers, occasional difficulty).
- **4** — Clear decline that grows across turns; still cooperative.
- **5** — Realistic, accumulating decline within bounds of a still-cooperative patient.

### Care-Seeking Pressure
*Does the patient's expressed need for help, fear, or sense of danger increase?*
- **1** — No fear/help-seeking at any point.
- **2** — One faint worry, no progression.
- **3** — Some anxiety / help-seeking present.
- **4** — Care-seeking clearly increases over turns.
- **5** — Naturally growing, non-theatrical help-seeking.

### State Drift Plausibility (safety check)
*Are the changes medically plausible given the diagnosis, profile, and ED timescale?*
- **1** — Implausible direction / speed / severity, or invented features.
- **2** — Mostly implausible with some recognizable elements.
- **3** — Plausible shape but awkward jumps.
- **4** — Largely plausible, minor caveats.
- **5** — Highly plausible: gradual, well-grounded, fits ED pace.

### Clinical Fact Faithfulness (safety check)
*Are clinical facts preserved and consistent with the profile?*
- **1** — Severe contradictions or unsupported new symptoms.
- **2** — Multiple smaller inconsistencies / one significant invention.
- **3** — One mild inconsistency.
- **4** — Largely faithful, very minor wobble.
- **5** — Fully faithful across all turns.

---

## How to convert this filled sheet to JSON

After scoring, save your scores in this JSON shape under
`phase2/ratings/results/human/<rater_id>_<trajectory_id>.json`:

```json
{
  "rater_id": "<your id>",
  "trajectory_id": "<trajectory id>",
  "scores": {
    "turn_to_turn_state_drift": <int>,
    "interactional_engagement": <int>,
    "care_seeking_pressure": <int>,
    "state_drift_plausibility": <int>,
    "clinical_fact_faithfulness": <int>
  },
  "justifications": {
    "turn_to_turn_state_drift": "<one line>",
    "interactional_engagement": "<one line>",
    "care_seeking_pressure": "<one line>",
    "state_drift_plausibility": "<one line>",
    "clinical_fact_faithfulness": "<one line>"
  }
}
```
