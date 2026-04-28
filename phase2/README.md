# Phase 2: Dynamic State Transition Evaluator

This directory holds everything needed to run Phase 2 of the PatientSim project:

- **scripted-doctor trajectory generation** for 3 patient cases × 3 conditions
- **LLM-as-judge automatic scoring** on 5 dimensions
- **human rating sheet** for evaluator validation
- **comparison report** between human and LLM judge

The plan it implements lives at `~/.claude/plans/optimized-bubbling-moon.md`.

## Layout

```
phase2/
├── patient_profiles/                   # you copy from playground (3 files)
│   ├── cerebral_infarction.json
│   ├── myocardial_infarction.json
│   └── pneumonia.json
├── scripts/
│   ├── high_yield_first/{case}.json    # focal-symptom questions (case-specific)
│   └── low_yield_first.json            # background-first questions (shared)
├── manual_trajectories/                # gold-reference trajectories (checked in)
│   ├── cerebral_infarction_manual_plausible.json
│   └── cerebral_infarction_manual_implausible.json
├── trajectories/                       # generated (gitignored)
├── ratings/
│   ├── rating_sheet_template.md        # for human raters
│   └── results/
│       ├── human/                      # one JSON per (rater, trajectory)
│       └── llm_judge/                  # one JSON per trajectory
└── README.md
```

Code lives in `src/phase2/`:

- `run_scripted_dialogue.py` — driver for trajectory generation
- `dynamic_state_eval.py` — LLM judge (Phase 2 evaluator)
- `aggregate.py` — score averaging utility
- `compare_ratings.py` — human ↔ LLM agreement report
- `config.py` — backend, persona, dimensions, weights

Prompts:
- `src/prompts/simulation/dynamic_state_prefix.txt`
- `src/prompts/eval/phase2_judge/{dimension}.txt` (×5)

## Prerequisites

```bash
export GENAI_API_KEY="..."
export GOOGLE_GENAI_USE_VERTEXAI="True"
```

The default backend is `gemini-2.5-flash` at `temperature=0`, matching `demo/config.json`.

## End-to-end flow

```bash
# 0. Drop 3 patient profiles into phase2/patient_profiles/
#    Each must contain the 33 fields validated by run_scripted_dialogue
#    (PatientAgent prompt requires age, gender, race, social hx, present
#    illness, cefr_*/med_* word lists, etc.)

# 1. Validate profiles
for case in cerebral_infarction myocardial_infarction pneumonia; do
    python -m src.phase2.run_scripted_dialogue \
        --profile phase2/patient_profiles/${case}.json --validate_only
done

# 2. Generate 9 trajectories (3 cases × 3 conditions)
for case in cerebral_infarction myocardial_infarction pneumonia; do
    python -m src.phase2.run_scripted_dialogue \
        --profile phase2/patient_profiles/${case}.json \
        --case_id ${case} --run_all_conditions \
        --output_dir phase2/trajectories
done

# 3. LLM judge on every trajectory (LLM-generated + manual references)
python -m src.phase2.dynamic_state_eval \
    --trajectories "phase2/*trajectories*/*.json" \
    --output_dir phase2/ratings/results/llm_judge

# 4. Distribute rating_sheet_template.md to human raters.
#    Have them save filled JSONs under phase2/ratings/results/human/
#    using filename "{rater_id}_{trajectory_id}.json"

# 5. Aggregate human ratings
python -m src.phase2.aggregate \
    --input "phase2/ratings/results/human/*.json" \
    --output phase2/ratings/results/human_aggregated.json

# 6. Compare
python -m src.phase2.compare_ratings \
    --human phase2/ratings/results/human_aggregated.json \
    --llm "phase2/ratings/results/llm_judge/*.json" \
    --output phase2/ratings/results/comparison.md
```

## Conditions

| Condition ID | Doctor script | Dynamic prefix | Use |
|---|---|---|---|
| `A_high_yield_original`  | high-yield, case-specific | off | Exp 1 Cond A |
| `B_low_yield_original`   | low-yield, shared | off | Exp 1 Cond B + Exp 2 baseline |
| `B_low_yield_dyn_prefix` | low-yield, shared | on  | Exp 2 main |

Manual references (`cerebral_infarction_manual_plausible` /
`cerebral_infarction_manual_implausible`) are scored alongside but use
hand-written turns, no LLM patient.

## Dimensions and weights

```
overall_dynamic_score = 0.30 * turn_to_turn_state_drift
                      + 0.20 * interactional_engagement
                      + 0.20 * care_seeking_pressure
                      + 0.15 * state_drift_plausibility
                      + 0.15 * clinical_fact_faithfulness
```

Definitions and 1–5 anchor rubrics are in
`src/prompts/eval/phase2_judge/{dimension}.txt`.

## Smoke tests

```bash
# Aggregate arithmetic check (plan §14.2 → §14.3)
python -m src.phase2.aggregate \
    --input phase2/_smoke/example_ratings.json \
    --output /tmp/smoke_agg.json

# Dry-run the dialogue runner — no LLM call, prints system prompt
python -m src.phase2.run_scripted_dialogue \
    --profile phase2/patient_profiles/cerebral_infarction.json \
    --condition B_low_yield_dyn_prefix --dry_run \
    --output_dir /tmp

# Dry-run the judge — prints all 5 prompts for one manual trajectory
python -m src.phase2.dynamic_state_eval \
    --trajectory phase2/manual_trajectories/cerebral_infarction_manual_plausible.json \
    --dry_run
```
