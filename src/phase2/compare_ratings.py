"""Phase 2: Human-aggregated ratings vs LLM judge — agreement analysis.

Inputs
------
--human   Path to one JSON produced by aggregate.py (human side)
--llm     Glob (or comma-separated globs) for per-trajectory LLM judge JSONs

Output
------
A Markdown report with:
  * per-trajectory table (human μ, LLM, |Δ|) for every dimension
  * Pearson r per dimension across trajectories
  * Spearman ρ on overall_dynamic_score
  * Negative-control check (manual_implausible plausibility/faithfulness ≤ 2)
  * Positive-control check (manual_plausible every dim ≥ 3)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

from . import config


# ---- correlation primitives (stdlib only) ----------------------------------

def _mean(xs):
    return sum(xs) / len(xs)


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(dx2 * dy2)
    if denom == 0:
        return None
    return num / denom


def _ranks(xs):
    """Average ranks for ties."""
    indexed = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    return pearson(_ranks(xs), _ranks(ys))


# ---- IO --------------------------------------------------------------------

def load_human(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_llm(globs: list[str]) -> dict[str, dict]:
    out = {}
    for g in globs:
        for p in glob.glob(g):
            with open(p) as f:
                data = json.load(f)
            out[data["trajectory_id"]] = data
    return out


# ---- analysis --------------------------------------------------------------

def per_traj_dim_pairs(human: dict, llm: dict[str, dict]):
    """Yield (trajectory_id, dimension, human_mean, llm_score)."""
    for traj_id, h in human.items():
        if traj_id not in llm:
            continue
        l = llm[traj_id]
        for dim in config.DIMENSIONS:
            h_mean = h.get("per_dimension", {}).get(dim, {}).get("mean")
            l_score = l.get("scores", {}).get(dim, {}).get("score")
            if h_mean is None or l_score is None:
                continue
            yield traj_id, dim, h_mean, l_score


def per_dim_pearson(pairs):
    by_dim: dict[str, tuple[list, list]] = {d: ([], []) for d in config.DIMENSIONS}
    for _, dim, h, l in pairs:
        by_dim[dim][0].append(h)
        by_dim[dim][1].append(l)
    return {dim: pearson(xs, ys) for dim, (xs, ys) in by_dim.items()}


def overall_pairs(human: dict, llm: dict[str, dict], key: str = "overall_dynamic_score"):
    xs, ys, traj_ids = [], [], []
    for traj_id, h in human.items():
        if traj_id not in llm:
            continue
        h_overall = h.get(key)
        l_overall = llm[traj_id].get(key)
        if h_overall is None or l_overall is None:
            continue
        xs.append(h_overall)
        ys.append(l_overall)
        traj_ids.append(traj_id)
    return traj_ids, xs, ys


def control_check(human: dict, llm: dict[str, dict], substr: str,
                  dims: list[str], threshold: float, op: str) -> dict | None:
    """Find first trajectory whose id contains `substr` and report dim scores."""
    for traj_id in human:
        if substr not in traj_id:
            continue
        if traj_id not in llm:
            continue
        h_dims = {d: human[traj_id]["per_dimension"][d]["mean"]
                  for d in dims if d in human[traj_id].get("per_dimension", {})}
        l_dims = {d: llm[traj_id]["scores"][d]["score"]
                  for d in dims if d in llm[traj_id].get("scores", {})}
        passed_h = all((v <= threshold if op == "le" else v >= threshold)
                       for v in h_dims.values()) if h_dims else None
        passed_l = all((v <= threshold if op == "le" else v >= threshold)
                       for v in l_dims.values()) if l_dims else None
        return {
            "trajectory_id": traj_id,
            "human": h_dims,
            "llm": l_dims,
            "human_pass": passed_h,
            "llm_pass": passed_l,
        }
    return None


# ---- report ----------------------------------------------------------------

def fmt(x):
    return "—" if x is None else f"{x:.2f}"


def render_markdown(human: dict, llm: dict[str, dict]) -> str:
    lines = ["# Human vs LLM Judge — Phase 2 Evaluator Validation", ""]

    # Coverage
    common = sorted(set(human) & set(llm))
    lines += [f"**Trajectories covered:** {len(common)} / "
              f"human={len(human)}, llm={len(llm)}"]
    if not common:
        lines += ["", "_No overlapping trajectories. Cannot compute agreement._"]
        return "\n".join(lines)

    # Per-trajectory table
    lines += ["", "## Per-trajectory dimension scores", "",
              "| Trajectory | Dimension | Human μ | LLM | |Δ| |",
              "|---|---|---|---|---|"]
    for traj_id, dim, h, l in per_traj_dim_pairs(human, llm):
        delta = abs(h - l)
        lines.append(f"| {traj_id} | {dim} | {fmt(h)} | {l} | {fmt(delta)} |")

    # Pearson per dim
    pairs = list(per_traj_dim_pairs(human, llm))
    pdim = per_dim_pearson(pairs)
    lines += ["", "## Pearson correlation per dimension (across trajectories)", ""]
    for dim in config.DIMENSIONS:
        lines.append(f"- **{dim}**: r = {fmt(pdim.get(dim))}")

    # Overall ranking agreement (additive)
    _, hs, ls = overall_pairs(human, llm, "overall_dynamic_score")
    rho = spearman(hs, ls)
    r = pearson(hs, ls)
    mae = (sum(abs(h - l) for h, l in zip(hs, ls)) / len(hs)) if hs else None
    lines += ["", "## Overall dynamic score (additive) agreement", "",
              f"- Pearson r:  {fmt(r)}",
              f"- Spearman ρ: {fmt(rho)}",
              f"- MAE:        {fmt(mae)}"]

    # Overall ranking agreement (guarded)
    _, hs_g, ls_g = overall_pairs(human, llm, "overall_dynamic_score_guarded")
    if hs_g:
        rho_g = spearman(hs_g, ls_g)
        r_g = pearson(hs_g, ls_g)
        mae_g = sum(abs(h - l) for h, l in zip(hs_g, ls_g)) / len(hs_g)
        lines += ["", "## Overall dynamic score (guarded by min(plaus,faith)) agreement", "",
                  "_Guarded score = additive · min(plaus, faith) / 5. Penalises high-drift "
                  "trajectories that violate plausibility or faithfulness._", "",
                  f"- Pearson r:  {fmt(r_g)}",
                  f"- Spearman ρ: {fmt(rho_g)}",
                  f"- MAE:        {fmt(mae_g)}"]

    # Control checks
    lines += ["", "## Negative control (manual_implausible)", ""]
    neg = control_check(
        human, llm, "manual_implausible",
        ["state_drift_plausibility", "clinical_fact_faithfulness"],
        2.0, "le")
    if neg is None:
        lines.append("_No `manual_implausible` trajectory found in the inputs._")
    else:
        lines.append(f"- Trajectory: `{neg['trajectory_id']}`")
        lines.append(f"- Human dims: {neg['human']} → "
                     f"{'PASS' if neg['human_pass'] else 'FAIL'}")
        lines.append(f"- LLM   dims: {neg['llm']} → "
                     f"{'PASS' if neg['llm_pass'] else 'FAIL'}")

    lines += ["", "## Positive control (manual_plausible)", ""]
    pos = control_check(
        human, llm, "manual_plausible",
        list(config.DIMENSIONS),
        3.0, "ge")
    if pos is None:
        lines.append("_No `manual_plausible` trajectory found in the inputs._")
    else:
        lines.append(f"- Trajectory: `{pos['trajectory_id']}`")
        lines.append(f"- Human dims: {pos['human']} → "
                     f"{'PASS' if pos['human_pass'] else 'FAIL'}")
        lines.append(f"- LLM   dims: {pos['llm']} → "
                     f"{'PASS' if pos['llm_pass'] else 'FAIL'}")

    lines += ["", "## Interpretation guide", "",
              "- **High r per dimension + high overall ρ**: LLM judge is a viable "
              "stand-in for human raters in Phase 3.",
              "- **High r but failed negative control**: LLM tracks rough quality "
              "but cannot catch implausible drift — keep human raters for safety.",
              "- **Low r overall**: prompt redesign needed; do not rely on LLM "
              "judge alone."]

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--human", required=True,
                   help="Path to aggregated human ratings JSON (from aggregate.py)")
    p.add_argument("--llm", required=True,
                   help="Glob, or comma-separated globs, for LLM judge JSONs")
    p.add_argument("--output", required=True, help="Output Markdown path")
    args = p.parse_args()

    human = load_human(Path(args.human))
    globs = [g.strip() for g in args.llm.split(",")]
    llm = load_llm(globs)
    if not llm:
        raise SystemExit(f"No LLM judge files matched: {args.llm}")

    md = render_markdown(human, llm)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"Wrote {out_path}")
    print(md)


if __name__ == "__main__":
    main()
