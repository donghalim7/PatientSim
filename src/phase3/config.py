"""Centralized config for Phase 3 (improved PatientSim with tracked state).

Reuses the Phase 2 backend / persona / cases / evaluator settings verbatim so
Phase 3 trajectories are scored by the same judge against the same baseline.
Adds only the dynamic-state schema this phase introduces.
"""

# Reuse Phase 2 settings (backend, persona, 5 cases, dimensions, weights).
from phase2.config import (  # noqa: F401
    BACKEND_MODEL,
    BACKEND_API_TYPE,
    BACKEND_PARAMS,
    PERSONA,
    DIMENSIONS,
    DIMENSION_WEIGHTS,
    RANDOM_SEED,
    NUM_WORD_SAMPLE,
)

# Phase 3 evaluates the same 5 cases used in the Phase 2 V17 analysis.
CASES = [
    "cerebral_infarction",
    "myocardial_infarction",
    "intestinal_obstruction_elderly_male",
    "pneumonia",
    "pneumonia_elderly_female",
]

# Doctor scripts to evaluate each case under (mirrors Phase 2 conditions A/B).
SCRIPTS = ["low_yield_first", "high_yield_first"]

# --- Tracked patient-state schema ------------------------------------------
# Three felt-state variables, each an integer on a 1-5 scale. clinical_severity
# is gated by case timescale; the two affective/interactional axes may drift
# even when severity is flat (this is what closes the subacute gap).
STATE_VARS = [
    "clinical_severity",
    "interactional_engagement",
    "care_seeking_pressure",
]

STATE_MIN = 1
STATE_MAX = 5
MAX_DELTA_PER_TURN = 1  # each variable may change by at most ±1 each turn

# Patient starts the encounter mild, fully engaged, low worry.
INITIAL_STATE = {
    "clinical_severity": 2,
    "interactional_engagement": 5,
    "care_seeking_pressure": 1,
}

# --- Hybrid drift policy ----------------------------------------------------
# Python owns the state trajectory; the model only renders words that express
# the injected target state. The trajectory is an explicit, clinically-motivated
# function of (case acuity, doctor strategy), reproducing the Phase 2 finding
# under control rather than leaving it to noisy model self-update.

# Case acuity: acute presentations may show mild symptom escalation over minutes;
# subacute (slow) presentations should NOT (keeps faithfulness intact).
ACUITY = {
    "cerebral_infarction": "acute",
    "myocardial_infarction": "acute",
    "intestinal_obstruction_elderly_male": "acute",
    "pneumonia": "subacute",
    "pneumonia_elderly_female": "subacute",
}

# Target END state + onset fraction per (acuity, delayed_focal). `delayed_focal`
# is True under the low-yield script (main complaint asked late) and False under
# high-yield (asked early → patient feels attended to → little drift).
#   key: (acuity, delayed_focal) -> {"end": {state}, "onset": fraction-through}
ARCS = {
    # Delayed focal questioning: affective axes drift; acute severity rises late.
    ("acute", True):     {"end": {"clinical_severity": 3, "interactional_engagement": 3, "care_seeking_pressure": 3}, "onset": 0.33},
    ("subacute", True):  {"end": {"clinical_severity": 2, "interactional_engagement": 4, "care_seeking_pressure": 3}, "onset": 0.45},
    # Focal questions early: patient feels attended to -> minimal drift.
    ("acute", False):    {"end": {"clinical_severity": 2, "interactional_engagement": 5, "care_seeking_pressure": 1}, "onset": 0.0},
    ("subacute", False): {"end": {"clinical_severity": 2, "interactional_engagement": 5, "care_seeking_pressure": 2}, "onset": 0.55},
}

DELAYED_FOCAL_SCRIPTS = {"low_yield_first": True, "high_yield_first": False}
