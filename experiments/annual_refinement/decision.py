"""Choose the primary research model once, from direct validation of portable 12-month weights."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from freeze import freeze
from records import WORK
from scoreboard import load_set, ocean_stats, window_stats
from submit_frozen import run as submit_holdouts

from annual_lim.layout import read
from tc_emulator.common import atomic, save_json


def metrics_for(tag):
    role = WORK / "verification" / tag / "val"
    complete = sorted(role.glob("*/complete.json"))
    if len(complete) != 12 or any(read(p)["records"] != 42 for p in complete):
        raise ValueError(f"Missing validation months or members for {tag}")
    wind = load_set(complete, "monthly_wind.parquet")
    windows = load_set(complete, "windows.parquet")
    windows["abs_log_error"] = abs(np.log(windows.model / windows.truth))
    ocean = load_set(complete, "ocean.parquet")
    f = load_set(complete, "forecast.parquet")
    guard = load_set(complete, "physics.parquet")
    guard = guard[guard.group == "all"]
    return dict(
        conditional_wind_rmse=np.sqrt(wind.conditional_mse.mean()),
        forecast_day1=np.sqrt(f[f.lead == 1].lim_mse.mean()),
        forecast_day15=np.sqrt(f[f.lead == 15].lim_mse.mean()),
        shear_error=guard.shear_quantile_ms.mean(),
        lag_error=guard.lag_bias.mean(),
        coverage_error=(ocean.groupby(["region", "field"]).coverage90.mean() - 0.9).abs().mean(),
        min_coverage=ocean.groupby(["field", "region"]).coverage90.mean().min(),
        **window_stats(windows),
        **ocean_stats(ocean),
    )


def table():
    baseline = metrics_for("published_v1")
    fields = [k for k in baseline if k.endswith("_crps")]
    rows = []
    for file in sorted((WORK / "parents").glob("*.json")):
        parent = read(file)
        for refined in (False, True):
            tag = parent["tag"] + ("_refined" if refined else "")
            if tag == "published_v1":
                continue
            comp = WORK / "comparison" / tag / "val" / "paired.csv"
            if not comp.is_file():
                continue
            weights = (
                WORK / "exports/combined" / tag / "weights" if refined else Path(parent["source"])
            )
            scores = metrics_for(tag)
            oc = np.mean([scores[k] / baseline[k] for k in fields])
            ratios = [
                oc,
                scores["conditional_wind_rmse"] / baseline["conditional_wind_rmse"],
                scores["total_all_logerr"] / baseline["total_all_logerr"],
                scores["within_all_logerr"] / baseline["within_all_logerr"],
            ]
            guard = (
                ratios[0] < 1.0
                and ratios[1] <= 1.002
                and max(ratios[2:]) <= 1.1
                and scores["forecast_day1"] <= 1.03 * baseline["forecast_day1"]
                and scores["forecast_day15"] <= 1.03 * baseline["forecast_day15"]
                and scores["shear_error"] <= baseline["shear_error"] + 0.35
                and scores["lag_error"] <= baseline["lag_error"] + 0.025
                and all(
                    scores[f"{t}_{g}_logerr"] <= 1.30 * baseline[f"{t}_{g}_logerr"]
                    for t in ("within", "total")
                    for g in ("cold", "warm")
                )
            )
            fit = read(weights.parent / "fit/09.json")
            recipe = fit["background_model"]
            # Calibration parameters are folded into the numerical Gaussian weights before deployment.
            dim = recipe["sst"] + recipe["ocean"] + 42 + 2 * recipe["wind"]
            rows.append(
                dict(
                    tag=tag,
                    parent=parent["tag"],
                    cohort=parent["cohort"],
                    weights=str(weights),
                    recipe=parent["name"],
                    filter=parent["filter"],
                    refined=refined,
                    dimension=dim,
                    guardrails=guard,
                    relative_composite=np.mean(ratios),
                    ocean_ratio=ratios[0],
                    mean_wind_ratio=ratios[1],
                    total_ratio=ratios[2],
                    within_ratio=ratios[3],
                    **scores,
                )
            )
    return pd.DataFrame(rows)


def selection(t):
    if t.empty:
        raise ValueError("No fully verified research candidates")
    feasible = t[t.guardrails]
    if feasible.empty:
        feasible = t
    near = feasible[feasible.relative_composite <= feasible.relative_composite.min() + 0.006]
    return near.sort_values(["dimension", "relative_composite", "refined", "tag"]).iloc[0]


def compare(account, final):
    data = table()
    chosen = selection(data)
    stamp = "selection.csv" if final else "validation_decision.csv"
    with atomic(WORK / stamp) as tmp:
        data.sort_values("relative_composite").to_csv(tmp, index=False)
    if final:
        parent = read(WORK / "parents" / (chosen.parent + ".json"))
        recipe = {k: parent[k] for k in ("cohort", "name", "filter", "stage")}
        recipe.update(label=chosen.tag, parent=chosen.parent, combined=bool(chosen.refined))
        if "active_base" in parent:
            recipe.update(active_base=parent["active_base"], active_recipe=parent["active_recipe"])
        if chosen.refined:
            picked = read(WORK / "finalists" / chosen.parent / "jobs.json")["choice"]
            recipe.update({k: picked[k] for k in ("weather_family", "weather", "spread")})
        save_json(WORK / "recipe.json", recipe)
        rationale = (
            "Validation on 7 independent members x 6 years x 12 months. Four equally weighted ratios compare the "
            "six physical ocean CRPS scores, conditional steering RMSE, multiday total and within-month variance "
            "errors with v1. Check temperature, 1/15-day forecast, shear, and lag guardrails. Select the smallest "
            "state within 0.006 of the best guarded score. Selection uses no independent-test or scenario metrics. "
            f"Guardrails satisfied: {bool(chosen.guardrails)}. See the recorded selection.csv for all verified contenders."
        )
        freeze(chosen.tag, chosen.weights, chosen.cohort, rationale)
        print(submit_holdouts(account), flush=True)
    print(
        data.sort_values("relative_composite")[
            [
                "tag",
                "cohort",
                "ocean_ratio",
                "mean_wind_ratio",
                "total_ratio",
                "within_ratio",
                "guardrails",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("Selected", chosen.tag)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--account")
    a = p.parse_args()
    if a.freeze and not a.account:
        p.error("Supply the account to launch the independent verification")
    compare(a.account, a.freeze)
