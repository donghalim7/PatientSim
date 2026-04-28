"""Phase 2 (Deliverable a): Scripted-doctor trajectory generator.

Drives PatientAgent with a fixed sequence of doctor questions (no LLM doctor) so
that variation between trajectories is attributable solely to the patient model
+ persona + optional dynamic-state prefix.

Usage
-----
Single condition:
    python -m src.phase2.run_scripted_dialogue \
        --profile phase2/patient_profiles/cerebral_infarction.json \
        --condition B_low_yield_dyn_prefix \
        --output_dir phase2/trajectories

All 3 conditions for one case:
    python -m src.phase2.run_scripted_dialogue \
        --profile phase2/patient_profiles/cerebral_infarction.json \
        --case_id cerebral_infarction \
        --run_all_conditions \
        --output_dir phase2/trajectories
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root BEFORE importing models.py (which reads env at import time)
REPO_ROOT_FOR_ENV = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT_FOR_ENV / ".env", override=False)

# Make src/ importable so the existing PatientAgent / utils resolve
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from agent.patient_agent import PatientAgent  # noqa: E402
from utils import set_seed  # noqa: E402
from phase2 import config  # noqa: E402

REPO_ROOT = SRC_DIR.parent
PROMPT_DIR = SRC_DIR / "prompts" / "simulation"
PREFIX_PATH = PROMPT_DIR / "dynamic_state_prefix.txt"

# Fields the PatientAgent prompt template requires (initial_system_patient_w_persona.txt)
REQUIRED_PROFILE_FIELDS = [
    "hadm_id", "age", "gender", "race",
    "tobacco", "alcohol", "illicit_drug", "exercise",
    "marital_status", "children", "living_situation", "occupation", "insurance",
    "allergies", "family_medical_history", "medical_device", "medical_history",
    "present_illness_positive", "present_illness_negative",
    "chiefcomplaint", "pain", "medication", "arrival_transport", "disposition",
    "diagnosis",
    "cefr_A1", "cefr_A2", "cefr_B1", "cefr_B2", "cefr_C1", "cefr_C2",
    "med_A", "med_B", "med_C",
]


def validate_profile(profile: dict) -> None:
    missing = [f for f in REQUIRED_PROFILE_FIELDS if f not in profile]
    if missing:
        raise ValueError(
            f"Patient profile is missing {len(missing)} required fields: {missing}"
        )


def load_profile(profile_path: Path) -> dict:
    with open(profile_path) as f:
        profile = json.load(f)
    if isinstance(profile, list):
        if len(profile) != 1:
            raise ValueError(
                "If profile JSON is a list, it must contain exactly one patient. "
                f"Got {len(profile)}."
            )
        profile = profile[0]
    return profile


def load_script(case_id: str, script_kind: str) -> tuple[dict, Path]:
    if script_kind == "high_yield_first":
        script_path = REPO_ROOT / "phase2" / "scripts" / "high_yield_first" / f"{case_id}.json"
    elif script_kind == "low_yield_first":
        script_path = REPO_ROOT / "phase2" / "scripts" / "low_yield_first.json"
    else:
        raise ValueError(f"Unknown script_kind: {script_kind}")
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    with open(script_path) as f:
        return json.load(f), script_path


def build_patient_agent(profile: dict, use_dynamic_prefix: bool) -> PatientAgent:
    agent = PatientAgent(
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
    if use_dynamic_prefix:
        prefix = PREFIX_PATH.read_text()
        agent.system_prompt_text = prefix + "\n\n" + agent.system_prompt_text
        agent.reset()
    return agent


def profile_summary(profile: dict) -> dict:
    keys = [
        "age", "gender", "diagnosis", "chiefcomplaint",
        "present_illness_positive", "present_illness_negative",
        "pain", "arrival_transport", "medical_history",
    ]
    return {k: profile.get(k) for k in keys}


def run_one_condition(
    profile: dict,
    profile_path: Path,
    case_id: str,
    condition_name: str,
    output_dir: Path,
    dry_run: bool = False,
) -> dict | None:
    cond = config.CONDITIONS[condition_name]
    script, script_path = load_script(case_id, cond["script_kind"])

    set_seed(config.RANDOM_SEED)
    agent = build_patient_agent(profile, cond["use_dynamic_prefix"])

    if dry_run:
        print("=" * 80)
        print(f"DRY RUN — case={case_id}  condition={condition_name}  "
              f"prefix={cond['use_dynamic_prefix']}")
        print(f"Script: {script['script_id']}  ({len(script['questions'])} questions)")
        print("-" * 80)
        print("System prompt (truncated to 3000 chars):")
        print(agent.system_prompt[:3000])
        print("=" * 80)
        return None

    print(f"=== {case_id} / {condition_name} (prefix={cond['use_dynamic_prefix']}) ===",
          flush=True)
    start = time.time()
    turns = []
    for i, q in enumerate(script["questions"], start=1):
        t0 = time.time()
        response = agent.inference(q)
        dt = time.time() - t0
        print(f"  turn {i}/{len(script['questions'])}  ({dt:.1f}s)  "
              f"patient: {response[:80]}{'...' if len(response) > 80 else ''}",
              flush=True)
        turns.append({"turn": i, "doctor": q, "patient": response})
        time.sleep(0.3)  # gentle rate limit
    elapsed = time.time() - start

    trajectory = {
        "trajectory_id": f"{case_id}_{condition_name}",
        "case_id": case_id,
        "patient_profile_id": str(profile.get("hadm_id", "unknown")),
        "patient_profile_summary": profile_summary(profile),
        "persona": dict(config.PERSONA),
        "backend": config.BACKEND_MODEL,
        "backend_api_type": config.BACKEND_API_TYPE,
        "use_dynamic_prefix": cond["use_dynamic_prefix"],
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
    output_path = output_dir / f"{case_id}_{condition_name}.json"
    with open(output_path, "w") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)
    print(f"  -> {output_path}  ({len(turns)} turns, {elapsed:.1f}s)")
    return trajectory


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--profile", required=True, help="Path to patient profile JSON")
    p.add_argument("--case_id", help="If omitted, derived from profile filename stem")
    p.add_argument("--condition", help=f"One of: {list(config.CONDITIONS)}")
    p.add_argument("--run_all_conditions", action="store_true",
                   help="Run all 3 conditions for the case in one go")
    p.add_argument("--output_dir", default="phase2/trajectories")
    p.add_argument("--dry_run", action="store_true",
                   help="Print system prompt and exit without LLM calls")
    p.add_argument("--validate_only", action="store_true",
                   help="Validate profile fields and exit")
    args = p.parse_args()

    profile_path = Path(args.profile)
    profile = load_profile(profile_path)
    validate_profile(profile)

    if args.validate_only:
        print(f"OK: {profile_path} has all {len(REQUIRED_PROFILE_FIELDS)} required fields.")
        print(f"     diagnosis={profile.get('diagnosis')}  age={profile.get('age')}  "
              f"gender={profile.get('gender')}")
        return

    case_id = args.case_id or profile_path.stem
    output_dir = Path(args.output_dir)

    if args.run_all_conditions:
        for cond_name in config.CONDITIONS:
            print(f"\n=== {case_id} / {cond_name} ===")
            run_one_condition(profile, profile_path, case_id, cond_name,
                              output_dir, dry_run=args.dry_run)
    else:
        if not args.condition:
            raise SystemExit("Specify --condition or --run_all_conditions")
        if args.condition not in config.CONDITIONS:
            raise SystemExit(
                f"Unknown condition '{args.condition}'. Choices: {list(config.CONDITIONS)}"
            )
        run_one_condition(profile, profile_path, case_id, args.condition,
                          output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
