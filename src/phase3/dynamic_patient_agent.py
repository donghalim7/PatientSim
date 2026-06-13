"""Phase 3: PatientAgent with a tracked, policy-driven felt state (hybrid).

DynamicPatientAgent extends PatientAgent with an explicit 3-variable felt state
(clinical_severity / interactional_engagement / care_seeking_pressure). Unlike a
pure model-self-update scheme (which proved noisy and mis-keyed drift to focal
questions), Python OWNS the state trajectory: each turn a clinically-motivated
policy — a function of (case acuity, doctor strategy, turn) — sets the target
state, and the model only produces words that express it. This reproduces the
Phase 2 drift finding under control and keeps subacute cases faithful (their
clinical_severity never rises, so no symptoms are invented).

Two drift-suppressing rules in the base system prompt are relaxed/removed (the
same anchor removal validated in Phase 2's V17) so the felt state can surface.
"""
from __future__ import annotations

import os

from agent.patient_agent import PatientAgent
from models import get_answer
from utils import process_string
from phase3 import config

# --- Anchor removal (identical to Phase 2 V17) ------------------------------
# Strings copied verbatim from
# src/prompts/simulation/initial_system_patient_w_persona.txt (lines 49, 57, 63).
CONSISTENCY_RULE = (
    "    2. Ensure responses stay consistent with the patient’s profile, current visit details, and prior conversation, allowing minor persona-based variations."
)
CONSISTENCY_RULE_RELAXED = (
    "    2. Ensure responses stay consistent with the patient’s profile and current visit details (clinical facts only). Your tone, focus, and felt state may evolve naturally across the conversation."
)
SENTENCE_RULE_TEMPLATE = (
    "    10. Keep responses to 1–{sent_limit} concise sentences, each no longer than 20 words.\n"
)
FINAL_SENT_LIMIT = (
    " You should answer within {sent_limit} sentences, keeping each sentence concise."
)

PROTOCOL_FILENAME = "phase3_state_protocol.txt"


def target_state(turn: int, total: int, init: dict, end: dict, onset: float) -> dict:
    """Linear ramp from init -> end across turns, beginning at `onset` fraction."""
    if total <= 1:
        frac = 1.0
    else:
        raw = (turn / total - onset) / max(1e-9, (1.0 - onset))
        frac = min(1.0, max(0.0, raw))
    return {v: round(init[v] + (end[v] - init[v]) * frac) for v in config.STATE_VARS}


def clamp_step(proposed: dict, prev: dict) -> dict:
    """Clamp each var to [1,5] and to within ±MAX_DELTA_PER_TURN of prev."""
    out = {}
    for k in config.STATE_VARS:
        lo = max(config.STATE_MIN, prev[k] - config.MAX_DELTA_PER_TURN)
        hi = min(config.STATE_MAX, prev[k] + config.MAX_DELTA_PER_TURN)
        out[k] = max(lo, min(hi, proposed[k]))
    return out


def render_state(state: dict) -> str:
    parts = ", ".join(f"{k}={state[k]}/5" for k in config.STATE_VARS)
    return f"[Your current felt state — {parts}]"


def phrasing_hints(state: dict) -> str:
    """Turn the target state into concrete instructions for how the words sound."""
    hints = []
    care = state["care_seeking_pressure"]
    eng = state["interactional_engagement"]
    sev = state["clinical_severity"]
    if care >= 4:
        hints.append("clearly express worry and ask for help")
    elif care == 3:
        hints.append("voice some concern about what is happening")
    elif care == 2:
        hints.append("let a little worry show")
    if eng <= 2:
        hints.append("you are struggling to focus: give a short answer or ask the doctor to repeat")
    elif eng == 3:
        hints.append("answer a bit more briefly, slightly less sharp")
    elif eng == 4:
        hints.append("answer a touch more briefly")
    if sev >= 4:
        hints.append("your main symptom feels worse now (only symptoms already in your profile)")
    elif sev == 3:
        hints.append("your main symptom feels a little worse than at arrival")
    if not hints:
        return "Answer naturally and clearly; you feel stable."
    return "; ".join(hints) + "."


class DynamicPatientAgent(PatientAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_dynamic_prompt()
        # Set by the driver before the dialogue loop.
        self.script_kind = "low_yield_first"
        self.acuity = "acute"
        self.total_turns = 6
        self.reset()

    def configure_run(self, script_kind: str, acuity: str, total_turns: int) -> None:
        self.script_kind = script_kind
        self.acuity = acuity
        self.total_turns = total_turns

    def _apply_dynamic_prompt(self) -> None:
        t = self.system_prompt_text
        t = t.replace(CONSISTENCY_RULE, CONSISTENCY_RULE_RELAXED)
        t = t.replace(SENTENCE_RULE_TEMPLATE, "")
        t = t.replace(FINAL_SENT_LIMIT, "")
        with open(os.path.join(self.prompt_dir, PROTOCOL_FILENAME)) as f:
            protocol = f.read()
        self.system_prompt_text = protocol + "\n\n" + t

    def reset(self) -> None:
        super().reset()
        self.state = dict(config.INITIAL_STATE)
        self.state_trace = []

    def inference(self, question) -> str:
        turn = len(self.state_trace) + 1
        delayed = config.DELAYED_FOCAL_SCRIPTS.get(self.script_kind, True)
        arc = config.ARCS[(self.acuity, delayed)]
        raw = target_state(turn, self.total_turns, config.INITIAL_STATE,
                           arc["end"], arc["onset"])
        new_state = clamp_step(raw, self.state)

        block = (f"[Turn {turn} of {self.total_turns} in an ongoing ED interview] "
                 f"{render_state(new_state)}\n"
                 f"How to speak this turn: {phrasing_hints(new_state)}")
        wrapped = (f"{block}\n\nDoctor: {question}\n\n"
                   "Respond with ONLY what the patient says out loud "
                   "(no narration, no state numbers).")
        messages = self.messages + [{"role": "user", "content": wrapped}]

        response = self.client(messages, model=self.model, **self.client_params)
        patient_response = process_string(get_answer(response))

        self.messages.append({"role": "user", "content": f"{question}"})
        self.messages.append({"role": "assistant", "content": f"{patient_response}"})
        self.state = new_state
        self.state_trace.append({"turn": turn, **new_state})
        self.log_token_usage(response)
        return patient_response
