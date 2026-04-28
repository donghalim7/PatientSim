"""Phase 2 (Deliverable b): LLM-as-Judge evaluator.

Scores a trajectory on 5 dimensions using one LLM call per dimension. This
is the central Phase 2 evaluator -- human ratings (Deliverable d) only exist
to validate it.

Usage
-----
Single trajectory:
    python -m src.phase2.dynamic_state_eval \
        --trajectory phase2/trajectories/cerebral_infarction_B_low_yield_dyn_prefix.json \
        --output phase2/ratings/results/llm_judge/cerebral_infarction_B_low_yield_dyn_prefix.json

Batch:
    python -m src.phase2.dynamic_state_eval \
        --trajectories "phase2/trajectories/*.json" \
        --output_dir phase2/ratings/results/llm_judge/

Dry run (print prompts, no LLM call):
    python -m src.phase2.dynamic_state_eval \
        --trajectory phase2/trajectories/X.json --dry_run
"""
from __future__ import annotations

import argparse
import ast
import datetime
import glob
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root BEFORE importing models.py (which reads env at import time)
REPO_ROOT_FOR_ENV = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT_FOR_ENV / ".env", override=False)

# Make src/ importable so models.py resolves
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from models import get_response_method, get_answer  # noqa: E402
from phase2 import config  # noqa: E402
from phase2.aggregate import (  # noqa: E402
    weighted_overall, guarded_overall, simple_sum, simple_mean
)

PROMPT_TEMPLATE_DIR = SRC_DIR / "prompts" / "eval" / "phase2_judge"
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"
DEFAULT_JUDGE_API_TYPE = "genai"
JUDGE_PARAMS = {"temperature": 0, "seed": 42}
MAX_RETRIES = 5

# Profile fields surfaced to the judge (kept compact to avoid prompt bloat)
PROFILE_FIELDS_FOR_JUDGE = [
    "age", "gender", "diagnosis", "chiefcomplaint",
    "present_illness_positive", "present_illness_negative",
    "pain", "arrival_transport", "medical_history",
]


def format_profile(profile_summary: dict) -> str:
    lines = []
    for key in PROFILE_FIELDS_FOR_JUDGE:
        val = profile_summary.get(key)
        if val in (None, ""):
            continue
        lines.append(f"  - {key}: {val}")
    return "\n".join(lines) if lines else "  (no profile summary available)"


def format_dialogue(turns: list[dict]) -> str:
    parts = []
    for t in turns:
        parts.append(f"Turn {t['turn']}")
        parts.append(f"  Doctor:  {t['doctor']}")
        parts.append(f"  Patient: {t['patient']}")
    return "\n".join(parts)


def load_prompt_template(dimension: str) -> str:
    path = PROMPT_TEMPLATE_DIR / f"{dimension}.txt"
    return path.read_text()


def render_prompt(dimension: str, profile_text: str, dialogue_text: str) -> str:
    template = load_prompt_template(dimension)
    return template.replace("{patient_profile}", profile_text) \
                   .replace("{dialogue}", dialogue_text)


def extract_json_object(text: str) -> dict | None:
    """Pull the first {...} JSON object from raw LLM text. Tolerant of fences."""
    # Strip code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")
    # Find first {...} balanced object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match is None:
        # Fall back to greedy match (handles nested but rare here)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            return None
    blob = match.group(0)
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(blob)
            if isinstance(obj, dict):
                return obj
        except (ValueError, SyntaxError):
            continue
    return None


def call_judge(client, model: str, user_prompt: str) -> dict:
    """Call the LLM with retries until a valid {score, rationale} dict is returned."""
    messages = [
        {"role": "system",
         "content": "You are an expert evaluator of doctor-patient simulator dialogues. "
                    "Always respond with the exact JSON shape the user requests, nothing else."},
        {"role": "user", "content": user_prompt},
    ]
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client(messages, model=model, **JUDGE_PARAMS)
            raw = get_answer(resp)
            obj = extract_json_object(raw)
            if obj is None:
                last_err = f"Could not parse JSON from: {raw[:200]}"
                continue
            score = obj.get("score")
            if not isinstance(score, int) or not (1 <= score <= 5):
                last_err = f"Invalid score field: {score}"
                continue
            return {"score": score, "rationale": str(obj.get("rationale", "")).strip()}
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
        time.sleep(0.5 * attempt)
    raise RuntimeError(f"Judge failed after {MAX_RETRIES} attempts. Last error: {last_err}")


def score_trajectory(trajectory: dict, judge_model: str, judge_api_type: str,
                     dry_run: bool = False) -> dict:
    profile_text = format_profile(trajectory.get("patient_profile_summary", {}))
    dialogue_text = format_dialogue(trajectory["turns"])

    if dry_run:
        for dim in config.DIMENSIONS:
            prompt = render_prompt(dim, profile_text, dialogue_text)
            print("=" * 80)
            print(f"DRY RUN — dimension: {dim}")
            print("-" * 80)
            print(prompt)
        return {}

    client = get_response_method(judge_api_type)
    if client is None:
        raise SystemExit(f"Unknown judge_api_type: {judge_api_type}")

    scores = {}
    for dim in config.DIMENSIONS:
        prompt = render_prompt(dim, profile_text, dialogue_text)
        result = call_judge(client, judge_model, prompt)
        scores[dim] = result
        print(f"  {dim}: {result['score']}  ({result['rationale'][:60]}...)")

    per_dim_means = {dim: float(scores[dim]["score"]) for dim in config.DIMENSIONS}
    overall = weighted_overall(per_dim_means)
    overall_guarded = guarded_overall(per_dim_means)

    return {
        "trajectory_id": trajectory["trajectory_id"],
        "case_id": trajectory.get("case_id"),
        "use_dynamic_prefix": trajectory.get("use_dynamic_prefix"),
        "rater_id": f"llm_judge:{judge_model}",
        "rater_type": "llm",
        "scores": scores,
        "overall_dynamic_score": overall,
        "overall_dynamic_score_guarded": overall_guarded,
        "simple_sum": simple_sum(per_dim_means),
        "simple_mean": simple_mean(per_dim_means),
        "metadata": {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "judge_model": judge_model,
            "judge_api_type": judge_api_type,
            "judge_params": JUDGE_PARAMS,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectory", help="Path to a single trajectory JSON")
    p.add_argument("--trajectories", help="Glob pattern, e.g. 'phase2/trajectories/*.json'")
    p.add_argument("--output", help="Output path (single trajectory mode)")
    p.add_argument("--output_dir", help="Output directory (batch mode)")
    p.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--judge_api_type", default=DEFAULT_JUDGE_API_TYPE)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if args.trajectory:
        paths = [Path(args.trajectory)]
    elif args.trajectories:
        paths = sorted(Path(p) for p in glob.glob(args.trajectories))
        if not paths:
            raise SystemExit(f"No files matched: {args.trajectories}")
    else:
        raise SystemExit("Specify --trajectory or --trajectories")

    if not args.dry_run:
        if args.trajectory and not args.output:
            raise SystemExit("--output is required with --trajectory")
        if args.trajectories and not args.output_dir:
            raise SystemExit("--output_dir is required with --trajectories")

    for path in paths:
        with open(path) as f:
            trajectory = json.load(f)
        print(f"\n=== {trajectory['trajectory_id']} ===")
        result = score_trajectory(trajectory, args.judge_model, args.judge_api_type,
                                  dry_run=args.dry_run)

        if args.dry_run:
            continue

        if args.trajectory:
            out_path = Path(args.output)
        else:
            out_path = Path(args.output_dir) / f"{trajectory['trajectory_id']}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  -> {out_path}  overall={result['overall_dynamic_score']}")


if __name__ == "__main__":
    main()
