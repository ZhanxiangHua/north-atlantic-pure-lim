"""Register full-year validation finalists, then schedule compatible train-only postfits."""
import argparse
from pathlib import Path

import numpy as np
from fullfit import CASES
from records import WORK, record
from scoreboard import acquire
from submit_frozen import submit

from annual_lim.layout import DEFAULT_WORK, read
from tc_emulator.common import lock, save_json


def cohort_bases(cohort):
    frame = acquire()
    frame = frame[frame.cohort == cohort].copy()
    frame = frame[~frame.name.str.contains("_block") & ~frame.name.str.startswith("calendar_")]
    frame = frame[frame.ocean_ratio <= 0.995]
    # Two different tradeoffs are examined before freezing. Existing windows are never pooled across calendar months first.
    valid = frame[
        (frame.mean_wind_ratio <= 1.002)
        & (frame.total_ratio <= 1.25)
        & (frame.within_ratio <= 1.25)
    ]
    if valid.empty:
        valid = frame

    def pick(data, weights):
        data = data.copy()
        columns = ["ocean_ratio", "mean_wind_ratio", "total_ratio", "within_ratio"]
        data["score"] = data[columns].to_numpy() @ np.array(weights)
        data["rank_cost"] = [CASES[n].sst + CASES[n].ocean + CASES[n].wind for n in data.name]
        minimum = data.score.min()
        # A smaller linear state breaks ties within 0.3% of the same validation criterion.
        return (
            data[data.score <= minimum + 0.003]
            .sort_values(["rank_cost", "score", "name", "filter"])
            .iloc[0]
        )

    joint = pick(valid, [0.25, 0.25, 0.25, 0.25])
    gain = valid[valid.mean_wind_ratio <= 0.99]
    if gain.empty:
        gain = valid
    predictive = pick(gain, [0.30, 0.40, 0.15, 0.15])
    choices = [joint]
    if (predictive["name"], predictive["filter"]) != (joint["name"], joint["filter"]):
        choices.append(predictive)
    return choices


def schedule(tag, cohort, name, filt, stage, account, dependencies=()):
    catalog = WORK / "parents" / (tag + ".json")
    with lock(catalog.with_suffix(".lock")):
        if catalog.is_file():
            row = read(catalog)
            if (row["cohort"], row["name"], row["filter"], row["stage"]) != (
                cohort,
                name,
                filt,
                stage,
            ):
                raise ValueError(f"Conflicting registration {tag}")
        else:
            source = (
                DEFAULT_WORK / "weights"
                if stage == "frozen"
                else WORK / "exports" / cohort / (name + "__" + filt) / "weights"
            )
            row = dict(
                tag=tag,
                cohort=cohort,
                name=name,
                filter=filt,
                stage=stage,
                source=str(source),
                jobs={},
            )
            save_json(catalog, row)
        jobs = row["jobs"]

        def add(key, script, settings, deps=()):
            if key not in jobs:
                jobs[key] = submit(script, settings, deps, account)
                save_json(catalog, row)

        source = Path(row["source"])
        ready = all((source / f"{m:02}/model.json").is_file() for m in range(1, 13))
        required = list(dependencies)
        if not ready:
            if stage not in ("coupled", "budget"):
                raise ValueError(f"Cannot train parent stage {stage}")
            add(
                "export",
                "fullfit" if stage == "coupled" else "between",
                dict(COHORT=cohort, EXPORT_MODEL=name, EXPORT_FILTER=filt),
                required,
            )
            required = [jobs["export"]]
        args = dict(MODEL_TAG=tag, MODEL_WEIGHTS=str(source), COHORT=cohort)
        for arm, script, root, marker in (
            ("seasonal", "seasonal", "seasonal", "windows.parquet"),
            ("modal", "modal", "modal_weather", "fits.json"),
            ("spread", "spread", "ocean_spread", "fits.json"),
        ):
            if len(list((WORK / root / tag).glob("*/" + marker))) < 12:
                add(arm, script, args, required)
        if len(list((WORK / "verification" / tag / "val").glob("*/complete.json"))) < 12:
            add("validation", "verify", args, required)
        if "active_base" not in row and cohort != "original":
            add("active", "active", args, required)
            complete = [jobs["active"]] + ([jobs["validation"]] if "validation" in jobs else [])
            add("active_selection", "active_parent", args, complete)
        if not (WORK / "comparison" / tag / "val/paired.csv").is_file() and tag != "published_v1":
            add(
                "intervals", "intervals", args, [jobs["validation"]] if "validation" in jobs else []
            )
        if tag != "published_v1" and not (WORK / "finalists" / tag / "jobs.json").is_file():
            dependencies = [
                jobs[k] for k in ("seasonal", "modal", "spread", "validation") if k in jobs
            ]
            add("finalist", "finalists", args, dependencies)
    record(
        "validation_parent_registered",
        cohort=cohort,
        tag=tag,
        recipe=name,
        filter=filt,
        stage=stage,
    )
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--account", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--tag")
    p.add_argument("--name")
    p.add_argument("--filter")
    p.add_argument("--stage", choices=["frozen", "budget", "coupled"])
    a = p.parse_args()
    if a.tag:
        print(schedule(a.tag, a.cohort, a.name, a.filter, a.stage, a.account))
    else:
        target = WORK / "experiment_records" / ("shortlist_" + a.cohort + ".json")
        with lock(target.with_suffix(".lock")):
            if target.is_file():
                candidates = read(target)["rows"]
            else:
                candidates = [r.to_dict() for r in cohort_bases(a.cohort)]
                save_json(target, dict(cohort=a.cohort, rows=candidates))
        for i, r in enumerate(candidates):
            tag = f"{a.cohort}_{'balanced' if i==0 else 'prediction'}"
            print(schedule(tag, a.cohort, r["name"], r["filter"], r["stage"], a.account))
