"""Select at most one wind-active covariance correction per validated parent."""
import argparse

import numpy as np
import pandas as pd
from decision import metrics_for
from records import WORK, record
from scoreboard import load_set, ocean_stats, window_stats
from shortlist import schedule
from submit_frozen import submit

from annual_lim.layout import read
from tc_emulator.common import atomic, lock, save_json


def pick(tag):
    base = metrics_for("published_v1")
    parent = metrics_for(tag)
    complete = sorted((WORK / "active_warming" / tag).glob("*/fits.json"))
    if len(complete) != 12:
        raise ValueError("Incomplete warming experiments")
    ocean = load_set(complete, "ocean.parquet")
    wind = load_set(complete, "windows.parquet")
    numeric = []
    oc = [k for k in base if k.endswith("_crps")]
    for name, g in ocean.groupby("filter"):
        scores = {**ocean_stats(g), **window_stats(wind[wind["filter"] == name])}
        rr = np.mean([scores[k] / base[k] for k in oc])
        numeric.append(
            dict(
                filter=name,
                ocean_ratio=rr,
                total_ratio=scores["total_all_logerr"] / base["total_all_logerr"],
                plausible=all(
                    scores[f"total_{x}_logerr"] <= 1.3 * base[f"total_{x}_logerr"]
                    for x in ("cold", "warm")
                ),
                **scores,
            )
        )
    result = pd.DataFrame(numeric)
    result["pair_loss"] = result.ocean_ratio + result.total_ratio
    original = (
        np.mean([parent[k] / base[k] for k in oc])
        + parent["total_all_logerr"] / base["total_all_logerr"]
    )
    result["quarter_gain"] = (original - result.pair_loss) / 4
    with atomic(WORK / "active_warming" / tag / "validation.csv") as tmp:
        result.to_csv(tmp, index=False)
    eligible = result[result.plausible & (result.quarter_gain >= 0.002)]
    if eligible.empty:
        return None
    pool = eligible[eligible.pair_loss <= eligible.pair_loss.min() + 0.004].copy()
    pool["complexity"] = [
        (0 if "f80" in v else 1) + (0 if "_raw" in v else 1) for v in pool["filter"]
    ]
    return pool.sort_values(["complexity", "pair_loss"]).iloc[0].to_dict()


def run(tag, account):
    parent = read(WORK / "parents" / (tag + ".json"))
    choice = pick(tag)
    if choice is None:
        record("warming_branch_screened", tag=tag, result="no qualifying improvement")
        return
    derived = tag + "_active"
    label = choice["filter"]
    catalog = WORK / "parents" / (derived + ".json")
    with lock(catalog.with_suffix(".lock")):
        if not catalog.exists():
            job = submit("active", dict(MODEL_TAG=tag, ACTIVE_EXPORT=label), [], account)
            source = WORK / "exports/active" / (tag + "__" + label) / "weights"
            item = {k: parent[k] for k in ("cohort", "name", "filter", "stage")}
            item.update(
                tag=derived,
                source=str(source),
                active_base=tag,
                active_recipe=label,
                jobs={"export": job},
            )
            save_json(catalog, item)
    schedule(derived, parent["cohort"], parent["name"], parent["filter"], parent["stage"], account)
    record(
        "warming_branch_screened",
        tag=tag,
        result="validated derivative",
        label=label,
        validation_gain=choice["quarter_gain"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--account", required=True)
    a = p.parse_args()
    run(a.tag, a.account)
