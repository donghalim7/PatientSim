"""Phase 2 (shared utility): score aggregation for human + LLM ratings.

Input formats accepted (auto-detected per file):

1. Single per-rater per-trajectory rating:
   {"rater_id": "alice", "trajectory_id": "X", "scores": {dim: int, ...}}

2. Multi-trajectory rater bundle:
   {"rater_id": "alice",
    "ratings": [{"trajectory_id": "X", "scores": {dim: int}}, ...]}

3. LLM-judge output (from dynamic_state_eval.py):
   {"trajectory_id": "X", "rater_id": "llm_judge:...",
    "scores": {dim: {"score": int, "rationale": str}, ...},
    "overall_dynamic_score": float}

Output: per-trajectory aggregate with mean / std / overall_dynamic_score.

Run as a CLI to aggregate a glob of files into one JSON, or import the
functions from other modules.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Local import without requiring src/ on sys.path: relative import works because
# this module lives inside the src/phase2 package.
from . import config


def _normalize_score(value):
    """Accept int, float, or {'score': int, ...} and return int (or None)."""
    if isinstance(value, dict):
        return value.get("score")
    if isinstance(value, (int, float)):
        return value
    return None


def normalize_rating_record(record: dict) -> list[dict]:
    """Return a list of {rater_id, trajectory_id, scores} dicts."""
    if "ratings" in record and isinstance(record["ratings"], list):
        rater_id = record.get("rater_id", "unknown")
        out = []
        for r in record["ratings"]:
            scores = {k: _normalize_score(v) for k, v in r.get("scores", {}).items()}
            out.append({
                "rater_id": rater_id,
                "trajectory_id": r["trajectory_id"],
                "scores": scores,
            })
        return out
    # Single record path (covers both human single + LLM judge output)
    scores = {k: _normalize_score(v) for k, v in record.get("scores", {}).items()}
    return [{
        "rater_id": record.get("rater_id", "unknown"),
        "trajectory_id": record["trajectory_id"],
        "scores": scores,
    }]


def weighted_overall(per_dim_means: dict[str, float]) -> float | None:
    """Additive weighted score. Returns None if any dimension missing.

    overall = Σ (weight_i · score_i)  for i in 5 dimensions.
    Range: 1.0 (all 1s) .. 5.0 (all 5s).
    """
    total = 0.0
    for dim, weight in config.DIMENSION_WEIGHTS.items():
        if dim not in per_dim_means or per_dim_means[dim] is None:
            return None
        total += weight * per_dim_means[dim]
    return round(total, 4)


def simple_sum(per_dim_means: dict[str, float]) -> float | None:
    """Unweighted sum of all 5 dimension means. Range 5.0..25.0.
    For sanity-checking against the weighted scores."""
    if any(per_dim_means.get(d) is None for d in config.DIMENSIONS):
        return None
    return round(sum(per_dim_means[d] for d in config.DIMENSIONS), 4)


def simple_mean(per_dim_means: dict[str, float]) -> float | None:
    """Unweighted arithmetic mean of all 5 dimensions. Range 1.0..5.0."""
    s = simple_sum(per_dim_means)
    return None if s is None else round(s / len(config.DIMENSIONS), 4)


def guarded_overall(per_dim_means: dict[str, float]) -> float | None:
    """Safety-guarded overall: weighted_overall · min(plaus, faith) / 5.

    Penalises trajectories that drift dynamically but violate plausibility or
    faithfulness. A trajectory with drift=5 / engagement=4 / care=4 but
    plaus=1 / faith=1 collapses from additive 3.4 to guarded ~0.68.
    A trajectory with all dims=4 stays at additive·0.8 ≈ guarded.

    Returns None if either safety dimension is missing.
    """
    additive = weighted_overall(per_dim_means)
    if additive is None:
        return None
    plaus = per_dim_means.get("state_drift_plausibility")
    faith = per_dim_means.get("clinical_fact_faithfulness")
    if plaus is None or faith is None:
        return None
    safety_factor = min(plaus, faith) / 5.0
    return round(additive * safety_factor, 4)


def aggregate(records: list[dict]) -> dict:
    """Aggregate normalized rating records into per-trajectory summaries."""
    by_traj: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_traj[r["trajectory_id"]].append(r)

    out = {}
    for traj_id, raters in by_traj.items():
        # Collect scores per dimension
        per_dim_scores: dict[str, list[float]] = defaultdict(list)
        for r in raters:
            for dim, score in r["scores"].items():
                if score is not None:
                    per_dim_scores[dim].append(float(score))

        per_dim_summary = {}
        per_dim_means = {}
        for dim, scores in per_dim_scores.items():
            mean = round(statistics.mean(scores), 4)
            std = round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0
            per_dim_summary[dim] = {
                "mean": mean,
                "std": std,
                "n": len(scores),
                "scores": scores,
            }
            per_dim_means[dim] = mean

        out[traj_id] = {
            "n_raters": len(raters),
            "rater_ids": sorted({r["rater_id"] for r in raters}),
            "per_dimension": per_dim_summary,
            "overall_dynamic_score": weighted_overall(per_dim_means),
            "overall_dynamic_score_guarded": guarded_overall(per_dim_means),
            "simple_sum": simple_sum(per_dim_means),
            "simple_mean": simple_mean(per_dim_means),
        }
    return out


def aggregate_from_paths(paths: list[Path]) -> dict:
    records: list[dict] = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                records.extend(normalize_rating_record(item))
        else:
            records.extend(normalize_rating_record(data))
    return aggregate(records)


def main():
    p = argparse.ArgumentParser(description="Aggregate Phase 2 rating JSONs.")
    p.add_argument("--input", required=True,
                   help="Glob pattern (e.g. 'phase2/ratings/results/human/*.json')")
    p.add_argument("--output", required=True, help="Output JSON path")
    args = p.parse_args()

    paths = [Path(p) for p in glob.glob(args.input)]
    if not paths:
        raise SystemExit(f"No files matched: {args.input}")
    print(f"Aggregating {len(paths)} file(s)")
    result = aggregate_from_paths(paths)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}  ({len(result)} trajectories)")


if __name__ == "__main__":
    main()
