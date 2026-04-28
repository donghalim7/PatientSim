"""Centralized config for Phase 2 experiments.

All scripts read from here so the backend / persona / weighting stay consistent
across (a) trajectory generation, (b) LLM judge, and aggregate utilities.
"""

# Backend (matches demo/config.json)
BACKEND_MODEL = "gemini-2.5-flash"
BACKEND_API_TYPE = "genai"
BACKEND_PARAMS = {"temperature": 0, "seed": 42}

# Controlled persona baseline (계획서 §7.2)
PERSONA = {
    "cefr_type": "B",
    "personality_type": "plain",
    "recall_level_type": "high",
    "dazed_level_type": "normal",
}

# 5 evaluation dimensions
DIMENSIONS = [
    "turn_to_turn_state_drift",
    "interactional_engagement",
    "care_seeking_pressure",
    "state_drift_plausibility",
    "clinical_fact_faithfulness",
]

# Overall score weights (sum = 1.0)
DIMENSION_WEIGHTS = {
    "turn_to_turn_state_drift":   0.30,
    "interactional_engagement":   0.20,
    "care_seeking_pressure":      0.20,
    "state_drift_plausibility":   0.15,
    "clinical_fact_faithfulness": 0.15,
}

# Conditions for run_scripted_dialogue
CONDITIONS = {
    "A_high_yield_original":   {"script_kind": "high_yield_first", "use_dynamic_prefix": False},
    "B_low_yield_original":    {"script_kind": "low_yield_first",  "use_dynamic_prefix": False},
    "B_low_yield_dyn_prefix":  {"script_kind": "low_yield_first",  "use_dynamic_prefix": True},
}

# Patient cases used in Phase 2
CASES = ["cerebral_infarction", "myocardial_infarction", "pneumonia"]

# Path constants (resolved relative to repo root at runtime)
RANDOM_SEED = 42
NUM_WORD_SAMPLE = 10
