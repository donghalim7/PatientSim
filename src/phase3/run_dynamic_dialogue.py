"""Phase 3 driver: scripted-doctor dialogue with the DynamicPatientAgent.

Produces a trajectory JSON in the SAME schema the Phase 2 evaluator consumes
(turns: [{turn, doctor, patient}], patient_profile_summary, case_id,
trajectory_id), plus an extra `state_trace` field (ignored by the judge) holding
the tracked felt state after each turn.

Usage
-----
    python src/phase3/run_dynamic_dialogue.py \
        --case cerebral_infarction --script low_yield_first \
        --output_dir phase3/trajectories

    python src/phase3/run_dynamic_dialogue.py \
        --case pneumonia --script low_yield_first --dry_run
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

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from utils import set_seed  # noqa: E402
from phase3 import config  # noqa: E402
from phase3.dynamic_patient_agent import DynamicPatientAgent  # noqa: E402

PROMPT_DIR = SRC_DIR / "prompts" / "simulation"
PROFILE_DIR = REPO_ROOT / "phase2" / "patient_profiles"
SCRIPT_DIR = REPO_ROOT / "phase2" / "scripts"

PROFILE_SUMMARY_FIELDS = [
    "age", "gender", "diagnosis", "chiefcomplaint",
    "present_illness_positive", "present_illness_negative",
    "pain", "arrival_transport", "medical_history",
    "medication", "living_situation",
]


def load_profile(case: str) -> tuple[dict, Path]:
    path = PROFILE_DIR / f"{case}.json"
    profile = json.loads(path.read_text())
    if isinstance(profile, list):
        if len(profile) != 1:
            raise ValueError(f"Profile list must hold exactly one patient, got {len(profile)}")
        profile = profile[0]
    return profile, path


def load_script(case: str, script_kind: str) -> tuple[dict, Path]:
    if script_kind == "high_yield_first":
        path = SCRIPT_DIR / "high_yield_first" / f"{case}.json"
    elif script_kind == "low_yield_first":
        path = SCRIPT_DIR / "low_yield_first.json"
    else:
        raise ValueError(f"Unknown script_kind: {script_kind}")
    return json.loads(path.read_text()), path


def build_agent(profile: dict) -> DynamicPatientAgent:
    return DynamicPatientAgent(
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


def run(case: str, script_kind: str, output_dir: Path, dry_run: bool = False) -> dict | None:
    profile, profile_path = load_profile(case)
    script, script_path = load_script(case, script_kind)

    set_seed(config.RANDOM_SEED)
    agent = build_agent(profile)
    acuity = config.ACUITY[case]
    agent.configure_run(script_kind, acuity, len(script["questions"]))

    if dry_run:
        from phase3.dynamic_patient_agent import (
            target_state, clamp_step, render_state, phrasing_hints,
        )
        total = len(script["questions"])
        delayed = config.DELAYED_FOCAL_SCRIPTS[script_kind]
        arc = config.ARCS[(acuity, delayed)]
        print("=" * 80)
        print(f"DRY RUN — case={case}  script={script_kind}  acuity={acuity}  delayed_focal={delayed}")
        print(f"Arc: end={arc['end']}  onset={arc['onset']}")
        print("-" * 80)
        print("Planned target state trajectory + phrasing:")
        prev = dict(config.INITIAL_STATE)
        for i in range(1, total + 1):
            raw = target_state(i, total, config.INITIAL_STATE, arc["end"], arc["onset"])
            st = clamp_step(raw, prev)
            print(f"  T{i}: {render_state(st)}  -> {phrasing_hints(st)}")
            prev = st
        print("=" * 80)
        return None

    print(f"=== {case} / {script_kind} ===", flush=True)
    start = time.time()
    turns = []
    for i, q in enumerate(script["questions"], start=1):
        t0 = time.time()
        response = agent.inference(q)
        dt = time.time() - t0
        st = agent.state
        print(f"  turn {i}/{len(script['questions'])}  ({dt:.1f}s)  "
              f"state=[sev{st['clinical_severity']} eng{st['interactional_engagement']} "
              f"care{st['care_seeking_pressure']}]  patient: "
              f"{response[:70]}{'...' if len(response) > 70 else ''}", flush=True)
        turns.append({"turn": i, "doctor": q, "patient": response})
        time.sleep(0.3)
    elapsed = time.time() - start

    trajectory = {
        "trajectory_id": f"{case}_phase3_{script_kind}",
        "case_id": case,
        "method": "phase3_dynamic_state",
        "patient_profile_id": str(profile.get("hadm_id", "unknown")),
        "patient_profile_summary": {k: profile.get(k) for k in PROFILE_SUMMARY_FIELDS},
        "persona": dict(config.PERSONA),
        "backend": config.BACKEND_MODEL,
        "script_id": script["script_id"],
        "turns": turns,
        "state_trace": agent.state_trace,
        "initial_state": dict(config.INITIAL_STATE),
        "metadata": {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "elapsed_sec": round(elapsed, 2),
            "patient_token_log": agent.token_log,
            "profile_path": str(profile_path),
            "script_path": str(script_path),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{case}_phase3_{script_kind}.json"
    out_path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False))
    print(f"  -> {out_path}  ({len(turns)} turns, {elapsed:.1f}s)")
    return trajectory


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--case", required=True, choices=config.CASES)
    p.add_argument("--script", default="low_yield_first", choices=config.SCRIPTS)
    p.add_argument("--output_dir", default=str(REPO_ROOT / "phase3" / "trajectories"))
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    run(args.case, args.script, Path(args.output_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
