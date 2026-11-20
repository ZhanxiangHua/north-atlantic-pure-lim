"""Admit one seasonally pooled ocean basis only when it improves the matched single-month fit."""
import argparse

from records import WORK, record
from scoreboard import acquire
from shortlist import schedule

from tc_emulator.common import atomic


def decide(cohort, account):
    frame = acquire()
    rows = frame[(frame.cohort == cohort) & (frame.stage == "coupled")]
    trials = rows[rows.name.isin(["calendar_wide", "calendar_partial"])].copy()
    trials["comparison"] = trials.name.map(
        {"calendar_wide": "nested_wide", "calendar_partial": "nested_partial"}
    )
    axes = ["ocean_ratio", "mean_wind_ratio", "total_ratio", "within_ratio"]
    original = rows[["name", "filter", *axes]].rename(
        columns={"name": "comparison", **{k: "base_" + k for k in axes}}
    )
    trials = trials.merge(original, on=["comparison", "filter"], validate="one_to_one")
    if len(trials) != 10:
        raise ValueError("The two same-rank comparisons need all five fixed temporal calibrations")
    for k in axes:
        trials["difference_" + k] = trials[k] - trials["base_" + k]
    eligible = trials.difference_ocean_ratio <= -0.002
    for k in axes[1:]:
        eligible &= trials["difference_" + k] <= 0.005
    trials["eligible"] = eligible
    with atomic(WORK / "calendar_ocean" / cohort / "validation.csv") as tmp:
        trials.to_csv(tmp, index=False)
    record(
        "calendar_ocean_gate",
        cohort=cohort,
        eligible=int(eligible.sum()),
        best_ocean_difference=float(trials.difference_ocean_ratio.min()),
    )
    if eligible.any():
        best = trials[eligible].sort_values(["relative_composite", "name", "filter"]).iloc[0]
        schedule(
            cohort + "_calendar_ocean", cohort, best["name"], best["filter"], "coupled", account
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--account", required=True)
    a = p.parse_args()
    decide(a.cohort, a.account)
