"""Rank training-only corrections by their matched validation metrics and verify the exported fit."""
import argparse

import numpy as np
import pandas as pd
from records import WORK, record
from scoreboard import load_set, ocean_stats, window_stats
from submit_frozen import submit

from annual_lim.layout import read
from tc_emulator.common import atomic, lock, save_json


def corrections(tag, baseline):
    parent = read(WORK / "parents" / (tag + ".json"))
    root = WORK / "verification" / tag / "val"
    complete = sorted(root.glob("*/complete.json"))
    if len(complete) != 12:
        raise ValueError(f"Validation is pending for {tag}")
    raw = load_set(complete, "windows.parquet")
    raw["abs_log_error"] = abs(np.log(raw.model / raw.truth))
    d = load_set(complete, "monthly_wind.parquet", ["conditional_mse"])
    wind = np.sqrt(d.conditional_mse.mean())
    parent_ocean = load_set(complete, "ocean.parquet")
    ocean = [("own", parent_ocean)]
    sp = sorted((WORK / "ocean_spread" / tag).glob("*/fits.json"))
    if len(sp) == 12:
        o = load_set(sp, "ocean.parquet")
        ocean += [(key, s) for key, s in o.groupby("filter") if key != "own"]
    weather = [("parent", parent["filter"], window_stats(raw))]
    for family, folder in (("modal", "modal_weather"), ("seasonal", "seasonal")):
        pieces = sorted((WORK / folder / tag).glob("*/windows.parquet"))
        if len(pieces) == 12:
            frame = load_set(pieces, "windows.parquet")
            weather.extend((family, n, window_stats(df)) for n, df in frame.groupby("filter"))
    out = []
    fields = [k for k in baseline if k.endswith("_crps")]
    for label, oc in ocean:
        o = ocean_stats(oc)
        coverage = oc.groupby(["field", "region"]).coverage90.mean()
        for family, w, stats in weather:
            # Domain-weighted proper score, conditional wind, total variance, within-month variance.
            ratios = [
                np.mean([o[k] / baseline[k] for k in fields]),
                wind / baseline["conditional_wind_rmse"],
                stats["total_all_logerr"] / baseline["total_all_logerr"],
                stats["within_all_logerr"] / baseline["within_all_logerr"],
            ]
            out.append(
                dict(
                    parent=tag,
                    cohort=parent["cohort"],
                    weather_family=family,
                    weather=w,
                    spread=label,
                    relative_composite=np.mean(ratios),
                    ocean_ratio=ratios[0],
                    mean_wind_ratio=ratios[1],
                    total_ratio=ratios[2],
                    within_ratio=ratios[3],
                    conditional_wind_rmse=wind,
                    coverage_mean_abs90=float(abs(coverage - 0.9).mean()),
                    minimum_coverage=coverage.min(),
                    **o,
                    **stats,
                )
            )
    return pd.DataFrame(out)


def baseline_stats():
    root = WORK / "verification/published_v1/val"
    complete = sorted(root.glob("*/complete.json"))
    raw = load_set(complete, "windows.parquet")
    raw["abs_log_error"] = abs(np.log(raw.model / raw.truth))
    f = load_set(complete, "monthly_wind.parquet", ["conditional_mse"])
    return dict(
        conditional_wind_rmse=np.sqrt(f.conditional_mse.mean()),
        **window_stats(raw),
        **ocean_stats(load_set(complete, "ocean.parquet")),
    )


def select(table):
    # Avoid modest aggregate gains obtained from pronounced temperature-regime regressions.
    base = baseline_stats()
    eligible = table[
        (table.total_cold_logerr <= 1.25 * base["total_cold_logerr"])
        & (table.total_warm_logerr <= 1.25 * base["total_warm_logerr"])
        & (table.within_cold_logerr <= 1.30 * base["within_cold_logerr"])
        & (table.within_warm_logerr <= 1.30 * base["within_warm_logerr"])
    ]
    if eligible.empty:
        eligible = table
    # Favour a correctly calibrated 90% ocean envelope when CRPS differs by <0.5%.
    score = eligible.relative_composite.min()
    pool = eligible[eligible.relative_composite <= score + 0.003].copy()
    pool["weather_cost"] = pool.weather_family.map({"parent": 0, "seasonal": 1, "modal": 2})
    pool["spread_cost"] = pool.spread.map(
        {"own": 0, "scalar": 1, "local": 2, "spatial": 3, "spatial_strong": 3, "spatial_first": 3}
    )
    best_cov = pool.coverage_mean_abs90.min()
    pool = pool[pool.coverage_mean_abs90 <= best_cov + 0.006]
    return pool.sort_values(["weather_cost", "spread_cost", "relative_composite"]).iloc[0]


def run(tag, account, schedule):
    base = baseline_stats()
    table = corrections(tag, base)
    destination = WORK / "finalists" / tag
    with atomic(destination / "candidates.csv") as tmp:
        table.to_csv(tmp, index=False)
    result = select(table)
    if not schedule:
        return result
    if len(table) < 30:
        raise ValueError(
            "The complete weather and ocean correction panels are required before scheduling"
        )
    parent = read(WORK / "parents" / (tag + ".json"))
    args = dict(MODEL_TAG=tag, COHORT=parent["cohort"], MODEL_WEIGHTS=parent["source"])
    name = tag + "_refined"
    receipt = destination / "jobs.json"
    with lock(receipt.with_suffix(".lock")):
        if receipt.is_file():
            run = read(receipt)
            choice = run["choice"]
        else:
            choice = result.to_dict()
            run = dict(choice=choice, jobs={})
            save_json(receipt, run)
        family, w, sp = choice["weather_family"], choice["weather"], choice["spread"]
        jobs = run["jobs"]

        def add(key, script, values, deps=()):
            if key not in jobs:
                jobs[key] = submit(script, values, deps, account)
                save_json(receipt, run)

        needs = []
        if family != "parent":
            add("weather", family, dict(args, **{family.upper() + "_EXPORT": w}))
            needs.append(jobs["weather"])
        if sp != "own":
            add("ocean", "spread", dict(args, SPREAD_EXPORT=sp))
            needs.append(jobs["ocean"])
        add(
            "compose",
            "combine",
            dict(
                PARENT_TAG=tag, WEATHER_FAMILY=family, WEATHER_FIT=w, SPREAD_FIT=sp, MODEL_TAG=name
            ),
            needs,
        )
        validate = dict(
            MODEL_TAG=name,
            MODEL_WEIGHTS=str(WORK / "exports/combined" / name / "weights"),
            COHORT=parent["cohort"],
        )
        if tag == "extended_balanced":
            add("directional_training_ablation", "lag_calibration", validate, [jobs["compose"]])
            add("ocean_residual_ablation", "ocean_residual", validate, [jobs["compose"]])
        add("validation", "verify", validate, [jobs["compose"]])
        add("intervals", "intervals", validate, [jobs["validation"]])
        add("spatial", "spatial", validate, [jobs["compose"]])
        add("marine", "marine", validate, [jobs["compose"]])
        add("controls", "controls", validate, [jobs["compose"]])
        add("budget", "attribution", validate, [jobs["compose"]])
    record(
        "validation_finalist", parent=tag, cohort=parent["cohort"], choice=[family, w, sp], tag=name
    )
    return choice


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--account")
    p.add_argument("--schedule", action="store_true")
    a = p.parse_args()
    if a.schedule and not a.account:
        p.error("An allocation is required for scheduled exports")
    print(run(a.tag, a.account, a.schedule))
