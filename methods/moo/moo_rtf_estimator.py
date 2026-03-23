"""
Multi-Objective Optimization for RTF Estimation
================================================
Estimates residual thyroid function (RTF) using a multi-objective approach
that balances two competing goals:
  1. Matching observed clinical hormone levels (estimation accuracy)
  2. Minimizing deviation from healthy euthyroid reference values (health optimality)

Uses the Thyrosim simulation grid and sampled clinical measurements.
"""

import numpy as np
import pandas as pd
from pathlib import Path


BIOMARKERS_SIM = ["FT4_mean", "FT3_mean", "TT3_mean", "TSH_mean"]
BIOMARKERS_SAMPLE = ["FT4_sample", "FT3_sample", "TT3_sample", "TSH_sample"]


def load_data():
    sim_candidates = [
        "simulation/thyrosim_cut_dataset_v2.csv",
        "simulation/thyrosim_cut_dataset.csv",
        "thyrosim_cut_dataset_v2.csv",
        "thyrosim_cut_dataset.csv",
    ]
    sample_candidates = [
        "simulation/thyrosim_sample_data.csv",
        "thyrosim_sample_data.csv",
    ]
    sim, sample = None, None
    for p in sim_candidates:
        if Path(p).exists():
            sim = pd.read_csv(p)
            print(f"Loaded sim grid: {p} ({len(sim)} rows)")
            break
    for p in sample_candidates:
        if Path(p).exists():
            sample = pd.read_csv(p)
            print(f"Loaded clinical samples: {p} ({len(sample)} rows)")
            break
    if sim is None or sample is None:
        raise FileNotFoundError("Missing dataset files. Need simulation grid + sample data.")
    return sim, sample


def compute_healthy_reference(sim):
    healthy = sim[np.isclose(sim["RTF"], 1.0, atol=0.005)]
    ref = {}
    for col in BIOMARKERS_SIM:
        ref[col] = {"median": healthy[col].median(), "std": healthy[col].std()}
    return ref


def snap_to_grid(sample, sim):
    sim_lt4 = sorted(sim["lt4"].unique())
    sample = sample.copy()
    sample["lt4_grid"] = sample["lt4"].apply(lambda x: min(sim_lt4, key=lambda g: abs(g - x)))
    return sample


def run_moo_estimation(sim, sample, ref):
    results = []
    n_matched = 0
    n_unmatched = 0

    for i, patient in sample.iterrows():
        mask = (
            (sim["height"] == patient["height"])
            & (sim["weight"] == patient["weight"])
            & (sim["sex"] == patient["sex"])
            & (sim["lt4"] == patient["lt4_grid"])
            & (sim["lt3"] == patient["lt3"])
        )
        candidates = sim[mask]

        if len(candidates) == 0:
            n_unmatched += 1
            continue
        n_matched += 1

        # Objective 1: matching error (normalized squared diff, sim vs observed)
        match_error = sum(
            ((candidates[sc] - patient[sa]) / ref[sc]["std"]) ** 2
            for sc, sa in zip(BIOMARKERS_SIM, BIOMARKERS_SAMPLE)
        )

        # Objective 2: health deviation (normalized squared diff, sim vs healthy median)
        health_dev = sum(
            ((candidates[sc] - ref[sc]["median"]) / ref[sc]["std"]) ** 2
            for sc in BIOMARKERS_SIM
        )

        # Best RTF by each objective
        best_match_idx = match_error.idxmin()
        best_match = candidates.loc[best_match_idx]
        best_health_idx = health_dev.idxmin()

        # Pareto front over the two objectives
        errs = pd.DataFrame({
            "RTF": candidates["RTF"].values,
            "match_error": match_error.values,
            "health_dev": health_dev.values,
        })
        vals = errs[["match_error", "health_dev"]].values
        n = len(vals)
        is_pareto = np.ones(n, dtype=bool)
        for a in range(n):
            if not is_pareto[a]:
                continue
            for b in range(n):
                if a != b and is_pareto[b] and np.all(vals[b] <= vals[a]) and np.any(vals[b] < vals[a]):
                    is_pareto[a] = False
                    break
        pareto_rtfs = errs[is_pareto]

        results.append({
            "height": patient["height"],
            "weight": patient["weight"],
            "sex": patient["sex"],
            "lt4": patient["lt4"],
            "lt3": patient["lt3"],
            "true_RTF": patient["RTF"],
            "est_RTF_match": best_match["RTF"],
            "est_RTF_health": candidates.loc[best_health_idx, "RTF"],
            "match_error": match_error.min(),
            "health_dev_at_match": health_dev.loc[best_match_idx],
            "health_dev_at_best": health_dev.min(),
            "n_pareto": len(pareto_rtfs),
            "pareto_RTF_min": pareto_rtfs["RTF"].min(),
            "pareto_RTF_max": pareto_rtfs["RTF"].max(),
            "FT4_pred": best_match["FT4_mean"],
            "FT3_pred": best_match["FT3_mean"],
            "TT3_pred": best_match["TT3_mean"],
            "TSH_pred": best_match["TSH_mean"],
            "FT4_sample": patient["FT4_sample"],
            "FT3_sample": patient["FT3_sample"],
            "TT3_sample": patient["TT3_sample"],
            "TSH_sample": patient["TSH_sample"],
        })

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(sample)} processed")

    print(f"Matched: {n_matched}, Unmatched: {n_unmatched}")
    return pd.DataFrame(results)


def evaluate(res):
    res["abs_error_match"] = np.abs(res["true_RTF"] - res["est_RTF_match"])
    res["abs_error_health"] = np.abs(res["true_RTF"] - res["est_RTF_health"])

    print("\n=== RTF Estimation via Matching Objective ===")
    print(f"  MAE:    {res['abs_error_match'].mean():.4f}")
    print(f"  Median: {res['abs_error_match'].median():.4f}")
    print(f"  Std:    {res['abs_error_match'].std():.4f}")

    print("\n=== RTF Estimation via Health Objective ===")
    print(f"  MAE:    {res['abs_error_health'].mean():.4f}")
    print(f"  Median: {res['abs_error_health'].median():.4f}")

    print(f"\n=== Pareto Front Stats ===")
    print(f"  Avg Pareto size:       {res['n_pareto'].mean():.1f}")
    print(f"  Avg Pareto RTF spread: {(res['pareto_RTF_max'] - res['pareto_RTF_min']).mean():.4f}")

    print(f"\n=== Health Deviation (lower = closer to healthy) ===")
    hm = res["health_dev_at_match"].mean()
    hb = res["health_dev_at_best"].mean()
    print(f"  At matching RTF: {hm:.4f}")
    print(f"  At optimal RTF:  {hb:.4f}")
    print(f"  Improvement:     {(1 - hb / hm) * 100:.1f}%")

    return res


def main():
    sim, sample = load_data()

    print("\nComputing healthy euthyroid reference from RTF=1.0...")
    ref = compute_healthy_reference(sim)
    for col in BIOMARKERS_SIM:
        r = ref[col]
        print(f"  {col}: median={r['median']:.4f}, std={r['std']:.4f}")

    print("\nSnapping sample LT4 values to simulation grid...")
    sample = snap_to_grid(sample, sim)

    print("\nRunning multi-objective RTF estimation...")
    res = run_moo_estimation(sim, sample, ref)

    print("\nEvaluating...")
    res = evaluate(res)

    out_path = "moo_rtf_results.csv"
    res.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(res)} rows)")


if __name__ == "__main__":
    main()
