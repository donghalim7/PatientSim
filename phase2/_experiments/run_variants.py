"""Phase 2 dynamic-state prompt-engineering experiments.

Runs scripted dialogues with each prompt variant and saves trajectories.
Variants are pure prompt-level interventions (no PatientAgent / models.py changes).

Usage:
    python -u phase2/_experiments/run_variants.py --case cerebral_infarction --variants v0,v1,v2,v3,v4,v5
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)

SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from agent.patient_agent import PatientAgent  # noqa: E402
from utils import set_seed  # noqa: E402
from phase2 import config  # noqa: E402

PROMPT_DIR = SRC_DIR / "prompts" / "simulation"
ORIGINAL_PREFIX_PATH = PROMPT_DIR / "dynamic_state_prefix.txt"
VARIANTS_DIR = Path(__file__).resolve().parent / "variants"

# ---- variant definitions ---------------------------------------------------

def v0_baseline_prefix(agent):
    """Current dynamic prefix prepended to system prompt (= existing behaviour)."""
    txt = ORIGINAL_PREFIX_PATH.read_text()
    agent.system_prompt_text = txt + "\n\n" + agent.system_prompt_text
    agent.reset()


def v1_suffix(agent):
    """V1: same kind of dynamic-state rule but appended (suffix) instead of prepended."""
    txt = (VARIANTS_DIR / "v1_suffix.txt").read_text()
    agent.system_prompt_text = agent.system_prompt_text + "\n\n" + txt
    agent.reset()


def v2_override(agent):
    """V2: explicit override of the conflicting 'stay consistent' rule, suffix position."""
    txt = (VARIANTS_DIR / "v2_override.txt").read_text()
    agent.system_prompt_text = agent.system_prompt_text + "\n\n" + txt
    agent.reset()


def v3_fewshot(agent):
    """V3: in-context demonstration of dynamic progression style, suffix position."""
    txt = (VARIANTS_DIR / "v3_fewshot.txt").read_text()
    agent.system_prompt_text = agent.system_prompt_text + "\n\n" + txt
    agent.reset()


def v5_internal_state(agent):
    """V5: internal felt-state monologue + relaxed sentence cap, suffix position."""
    txt = (VARIANTS_DIR / "v5_internal_state.txt").read_text()
    agent.system_prompt_text = agent.system_prompt_text + "\n\n" + txt
    agent.reset()


def v6_combo(agent):
    """V6: best-known combo (V2 override + V5 internal state). Tested after individual results."""
    txt2 = (VARIANTS_DIR / "v2_override.txt").read_text()
    txt5 = (VARIANTS_DIR / "v5_internal_state.txt").read_text()
    agent.system_prompt_text = agent.system_prompt_text + "\n\n" + txt2 + "\n\n" + txt5
    agent.reset()


def v7_temp_high(agent):
    """V7: V0 prefix + temperature 0.7 (loosen deterministic decoding)."""
    txt = ORIGINAL_PREFIX_PATH.read_text()
    agent.system_prompt_text = txt + "\n\n" + agent.system_prompt_text
    agent.reset()
    agent.client_params = dict(agent.client_params)
    agent.client_params["temperature"] = 0.7


def v9_multishot(agent):
    """V9: 6-turn 2-shot demonstration suffix (much stronger than V3)."""
    txt = (VARIANTS_DIR / "v9_multishot.txt").read_text()
    agent.system_prompt_text = agent.system_prompt_text + "\n\n" + txt
    agent.reset()


# Per-turn user-message wrappers (V4, V8 use this instead of system-prompt change)

def v4_turn_wrapper(turn_idx, total_turns, doctor_question):
    """V4: inject a per-turn felt-state cue into the user message itself."""
    if turn_idx == 1:
        return doctor_question
    cue = (
        f"[Note to yourself before answering: it is now turn {turn_idx} of {total_turns} "
        f"in this ED interview. Your chief complaint has been continuing while you sit and talk. "
        f"Let any subtle change in how you feel right now — body, focus, or worry — "
        f"leak into your reply, without inventing new symptoms or contradicting your profile.]\n"
        f"{doctor_question}"
    )
    return cue


# V8: explicit per-turn emotional schedule (much more concrete than V4)
EMOTIONAL_SCHEDULE = {
    1: "calm and factual; no worry yet",
    2: "still mostly calm; faint awareness that the symptoms have not improved",
    3: "subtle worry creeping in; minor hint that this is dragging",
    4: "noticeably aware minutes are passing; mild help-seeking can leak through",
    5: "clearly worried; finding it slightly harder to focus on background questions",
    6: "openly anxious; explicit but realistic help-seeking; acknowledge feeling worse",
}


def v8_turn_wrapper(turn_idx, total_turns, doctor_question):
    """V8: per-turn felt-state TARGET delivered through the user message."""
    target = EMOTIONAL_SCHEDULE.get(turn_idx, "calm and factual")
    cue = (
        f"[Internal target for this turn — color your reply with this felt state, "
        f"do not quote it: \"{target}\". "
        f"Stay grounded in your profile and do not invent symptoms. "
        f"Answer the doctor's question first, then let your felt state subtly leak in.]\n"
        f"{doctor_question}"
    )
    return cue


# V10: V8 + concrete behavioral cue per turn (engagement signal) + relaxed length late
BEHAVIOR_SCHEDULE = {
    1: ("calm and factual; no worry yet",
        "Answer briefly and clearly, 1–2 sentences."),
    2: ("still mostly calm; faint awareness symptoms have not improved",
        "Answer the question, then add ONE short sentence noting the symptoms are still there. 2 sentences."),
    3: ("subtle worry creeping in",
        "Answer the question, then add ONE phrase showing mild worry (e.g., 'this still feels strange'). 2-3 sentences."),
    4: ("noticeably aware time is passing without help; mild help-seeking",
        "Briefly answer the question, then ask ONCE for clarification ('sorry, can you repeat that?') OR mention difficulty focusing. 2-3 sentences."),
    5: ("clearly worried; finding it harder to focus on background questions",
        "Answer briefly, then add a phrase that you are finding it hard to focus, OR a brief help-seeking line. 3-4 sentences."),
    6: ("openly anxious; explicit but realistic help-seeking; acknowledge feeling worse",
        "Give the factual answer, then acknowledge feeling worse and ask for help once. 4-5 sentences allowed here."),
}


def v10_turn_wrapper(turn_idx, total_turns, doctor_question):
    """V10: per-turn target + concrete behavioral instruction + length quota."""
    target, behavior = BEHAVIOR_SCHEDULE.get(turn_idx, ("calm and factual", "1-2 sentences."))
    cue = (
        f"[Internal target for this turn — DO NOT QUOTE: felt state = \"{target}\". "
        f"Behavioral target: {behavior} "
        f"Stay grounded in your profile and do not invent symptoms. "
        f"Answer the doctor's question first, then let the targeted state subtly leak in.]\n"
        f"{doctor_question}"
    )
    return cue


def v11_v8_plus_multishot(agent):
    """V11: V8 turn-wrapper + V9 multi-shot suffix together."""
    txt = (VARIANTS_DIR / "v9_multishot.txt").read_text()
    agent.system_prompt_text = agent.system_prompt_text + "\n\n" + txt
    agent.reset()


# ---- Anchor-removal variants (no per-turn injection, no extra text) -------
# These selectively delete the parts of the existing PatientAgent main prompt
# that suppress drift, and observe whether the model drifts naturally.

SENTENCE_RULE = (
    "    10. Keep responses to 1–3 concise sentences, each no longer than 20 words.\n"
)
SENTENCE_RULE_TEMPLATE = (
    "    10. Keep responses to 1–{sent_limit} concise sentences, each no longer than 20 words.\n"
)
FINAL_SENT_LIMIT = (
    " You should answer within {sent_limit} sentences, keeping each sentence concise."
)

CONSISTENCY_RULE = (
    "    2. Ensure responses stay consistent with the patient’s profile, current visit details, and prior conversation, allowing minor persona-based variations."
)
CONSISTENCY_RULE_RELAXED = (
    "    2. Ensure responses stay consistent with the patient’s profile and current visit details (clinical facts only). Your tone, focus, and felt state may evolve naturally across the conversation."
)


def _strip_sentence_limit(agent):
    t = agent.system_prompt_text
    # Remove rule 10 (with literal {sent_limit} placeholder still in text)
    t = t.replace(SENTENCE_RULE_TEMPLATE, "")
    # Remove the trailing reminder
    t = t.replace(FINAL_SENT_LIMIT, "")
    agent.system_prompt_text = t


def _relax_consistency(agent):
    agent.system_prompt_text = agent.system_prompt_text.replace(
        CONSISTENCY_RULE, CONSISTENCY_RULE_RELAXED
    )


def _replace_plain_persona(agent):
    """Replace the 'plain' personality description in patient_profile so it
    no longer reads 'No strong emotions or noticeable behavior'."""
    permissive = (
        "\n\t\tNeutral baseline tone. Your felt state — body, focus, worry — "
        "may evolve naturally as the encounter progresses, the way it would for any real patient."
    )
    agent.patient_profile["personality"] = permissive


def v12_no_sentence_limit(agent):
    """V12: remove the 1-3 sentences/20 words rule. Nothing else changes."""
    _strip_sentence_limit(agent)
    agent.reset()


def v13_relax_consistency(agent):
    """V13: relax the 'stay consistent with prior conversation' rule. Nothing else."""
    _relax_consistency(agent)
    agent.reset()


def v14_remove_both_anchors(agent):
    """V14: V12 + V13 combined. No per-turn cues. No extra text."""
    _strip_sentence_limit(agent)
    _relax_consistency(agent)
    agent.reset()


def v15_remove_anchors_and_persona(agent):
    """V15: V14 + replace 'plain' persona description with permissive baseline."""
    _strip_sentence_limit(agent)
    _relax_consistency(agent)
    _replace_plain_persona(agent)
    agent.reset()


SOFT_NUDGE = (
    "\n\nNote: As this ED encounter unfolds and the patient's symptoms continue, "
    "the patient's felt state — body sensations, focus, and worry — may naturally evolve "
    "the way it would for any real patient sitting in the ED for several minutes. "
    "Stay grounded in the profile and never invent symptoms outside it."
)


def v16_anchors_plus_soft_nudge(agent):
    """V16: V14 anchor removal + a single soft system-prompt nudge (no per-turn cues)."""
    _strip_sentence_limit(agent)
    _relax_consistency(agent)
    agent.system_prompt_text = agent.system_prompt_text + SOFT_NUDGE
    agent.reset()


def v17_anchors_plus_multishot(agent):
    """V17: V14 anchor removal + V9 multi-shot suffix (NO per-turn injection)."""
    _strip_sentence_limit(agent)
    _relax_consistency(agent)
    txt = (VARIANTS_DIR / "v9_multishot.txt").read_text()
    agent.system_prompt_text = agent.system_prompt_text + "\n\n" + txt
    agent.reset()


def v18_anchors_only(agent):
    """V18: V14 anchor removal alone (system side); per-turn schedule provided via wrapper."""
    _strip_sentence_limit(agent)
    _relax_consistency(agent)
    agent.reset()


def identity_wrapper(turn_idx, total_turns, doctor_question):
    return doctor_question


# Variant registry: id -> (system_modifier, turn_wrapper, description)
VARIANTS = {
    "v0": (v0_baseline_prefix, identity_wrapper, "Baseline: existing dynamic prefix (prepended)"),
    "v1": (v1_suffix,           identity_wrapper, "Suffix instead of prefix"),
    "v2": (v2_override,         identity_wrapper, "Override 'stay consistent' rule (suffix)"),
    "v3": (v3_fewshot,          identity_wrapper, "Few-shot dynamic progression style (suffix)"),
    "v4": (lambda a: None,      v4_turn_wrapper,  "Per-turn user-message felt-state cue (no sysprompt change)"),
    "v5": (v5_internal_state,   identity_wrapper, "Internal-state monologue + relaxed sentences (suffix)"),
    "v6": (v6_combo,            identity_wrapper, "Combo: V2 override + V5 internal state"),
    "v7": (v7_temp_high,        identity_wrapper, "V0 prefix + temperature 0.7"),
    "v8": (lambda a: None,      v8_turn_wrapper,  "Per-turn explicit emotional-schedule target injection"),
    "v9": (v9_multishot,        identity_wrapper, "Multi-shot (2 full 6-turn examples) suffix"),
    "v10": (lambda a: None,     v10_turn_wrapper, "V8 schedule + concrete per-turn behaviors + length quota"),
    "v11": (v11_v8_plus_multishot, v8_turn_wrapper, "V8 turn-wrapper + V9 multi-shot suffix"),
    "v12": (v12_no_sentence_limit, identity_wrapper, "Anchor removal: drop 1-3 sentence cap"),
    "v13": (v13_relax_consistency, identity_wrapper, "Anchor removal: relax 'consistent with prior conversation' rule"),
    "v14": (v14_remove_both_anchors, identity_wrapper, "Anchor removal: V12+V13 (drop cap AND relax consistency)"),
    "v15": (v15_remove_anchors_and_persona, identity_wrapper, "Anchor removal: V14 + replace 'plain' personality text"),
    "v16": (v16_anchors_plus_soft_nudge, identity_wrapper, "V14 anchor removal + 1-line soft nudge (no per-turn)"),
    "v17": (v17_anchors_plus_multishot, identity_wrapper, "V14 anchor removal + multi-shot only (no per-turn)"),
    "v18": (v18_anchors_only,           v8_turn_wrapper,  "V14 anchor removal + V8 per-turn schedule (no multi-shot)"),
}


# ---- runner ---------------------------------------------------------------

def load_profile(case: str) -> tuple[dict, Path]:
    p = REPO_ROOT / "phase2" / "patient_profiles" / f"{case}.json"
    return json.loads(p.read_text()), p


def load_low_yield_script() -> tuple[dict, Path]:
    p = REPO_ROOT / "phase2" / "scripts" / "low_yield_first.json"
    return json.loads(p.read_text()), p


def load_script(script_kind: str, case_id: str) -> tuple[dict, Path]:
    """Resolve script_kind into a file path.

    - 'low_yield_first'           → phase2/scripts/low_yield_first.json (shared)
    - 'high_yield_first'          → phase2/scripts/high_yield_first/<case>.json
    - 'low_yield_first_10turn'    → phase2/scripts/low_yield_first_10turn.json (shared)
    - any other string is treated as a filename under phase2/scripts/
    """
    base = REPO_ROOT / "phase2" / "scripts"
    if script_kind == "low_yield_first":
        p = base / "low_yield_first.json"
    elif script_kind == "high_yield_first":
        p = base / "high_yield_first" / f"{case_id}.json"
    else:
        p = base / script_kind
        if not p.suffix:
            p = p.with_suffix(".json")
    return json.loads(p.read_text()), p


def build_agent(profile: dict) -> PatientAgent:
    return PatientAgent(
        patient_profile=profile,
        backend_str=config.BACKEND_MODEL,
        backend_api_type=config.BACKEND_API_TYPE,
        prompt_dir=str(PROMPT_DIR),
        prompt_file="initial_system_patient_w_persona",
        num_word_sample=config.NUM_WORD_SAMPLE,
        cefr_type=config.PERSONA["cefr_type"],
        personality_type=config.PERSONA["personality_type"],
        recall_level_type=config.PERSONA["recall_level_type"],
        dazed_level_type=config.PERSONA["dazed_level_type"],
        client_params=dict(config.BACKEND_PARAMS),
        verbose=False,
    )


def run_variant(case: str, variant_id: str, output_dir: Path,
                script_kind: str = "low_yield_first",
                output_suffix: str = "",
                seed: int | None = None,
                temperature: float | None = None) -> dict:
    if variant_id not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant_id}")
    sys_modifier, turn_wrapper, description = VARIANTS[variant_id]

    profile, profile_path = load_profile(case)
    script, script_path = load_script(script_kind, case)

    actual_seed = seed if seed is not None else config.RANDOM_SEED
    set_seed(actual_seed)
    agent = build_agent(profile)
    if seed is not None:
        agent.client_params = dict(agent.client_params)
        agent.client_params["seed"] = seed
    if temperature is not None:
        agent.client_params = dict(agent.client_params)
        agent.client_params["temperature"] = temperature
    sys_modifier(agent)

    print(f"=== {case} / {variant_id} ({description}) ===", flush=True)
    start = time.time()
    turns = []
    for i, q in enumerate(script["questions"], start=1):
        wrapped = turn_wrapper(i, len(script["questions"]), q)
        t0 = time.time()
        response = agent.inference(wrapped)
        dt = time.time() - t0
        print(f"  turn {i}/{len(script['questions'])}  ({dt:.1f}s)  patient: "
              f"{response[:90]}{'...' if len(response) > 90 else ''}", flush=True)
        turns.append({"turn": i, "doctor": q, "doctor_wrapped": wrapped, "patient": response})
        time.sleep(0.3)
    elapsed = time.time() - start

    suffix_part = f"_{output_suffix}" if output_suffix else ""
    trajectory = {
        "trajectory_id": f"{case}_exp_{variant_id}{suffix_part}",
        "case_id": case,
        "variant_id": variant_id,
        "variant_description": description,
        "patient_profile_id": str(profile.get("hadm_id")),
        "patient_profile_summary": {k: profile.get(k) for k in
            ["age", "gender", "diagnosis", "chiefcomplaint",
             "present_illness_positive", "present_illness_negative",
             "pain", "arrival_transport", "medical_history"]},
        "persona": dict(config.PERSONA),
        "backend": config.BACKEND_MODEL,
        "use_dynamic_prefix": variant_id != "v4_no_prefix" and variant_id is not None,
        "script_id": script["script_id"],
        "turns": turns,
        "metadata": {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "elapsed_sec": round(elapsed, 2),
            "patient_token_log": agent.token_log,
            "profile_path": str(profile_path),
            "script_path": str(script_path),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{case}_exp_{variant_id}{suffix_part}.json"
    out_path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False))
    print(f"  -> {out_path}  ({len(turns)} turns, {elapsed:.1f}s)", flush=True)
    return trajectory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, choices=config.CASES)
    p.add_argument("--variants", required=True,
                   help=f"Comma-separated list from: {list(VARIANTS)}")
    p.add_argument("--script", default="low_yield_first",
                   help="Doctor script kind: 'low_yield_first', 'high_yield_first', "
                        "'low_yield_first_10turn', or a filename under phase2/scripts/")
    p.add_argument("--output_dir",
                   default=str(REPO_ROOT / "phase2" / "_experiments" / "trajectories"))
    p.add_argument("--output_suffix", default="",
                   help="Tag appended to output filename, e.g. 'highyield' or '10turn'")
    p.add_argument("--seed", type=int, default=None,
                   help="Override the LLM call seed (default: config.RANDOM_SEED=42)")
    p.add_argument("--temperature", type=float, default=None,
                   help="Override the LLM call temperature (default: 0)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    for vid in args.variants.split(","):
        vid = vid.strip()
        if vid:
            run_variant(args.case, vid, output_dir,
                        script_kind=args.script,
                        output_suffix=args.output_suffix,
                        seed=args.seed,
                        temperature=args.temperature)


if __name__ == "__main__":
    main()
