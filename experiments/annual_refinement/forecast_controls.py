"""Compare two-day-conditioned linear forecasts using LIM and the two Fourier covariance laws."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ocean import GROUP_LIMITS, data_paths, panel_cohort
from paired_intervals import panel_weights, record_estimate
from records import WORK, record
from tuned_fourier import calibrated
from weather_controls import daily_fourier_lags

from annual_lim.layout import read
from annual_lim.model import MonthlyModel
from annual_lim.train import months
from tc_emulator.common import atomic
from tc_emulator.diagnostics import region_area, steering

LEADS = (1, 5, 10, 15)


def linear_weights(lags, lead):
    """Regress y[t+h] on (y[t],y[t-1]); a common spatial covariance cancels from the equation."""
    c0, c1 = lags[:2]
    return np.linalg.solve([[c0, c1], [c1, c0]], lags[lead : lead + 2])


def run(tag, weights, cohort, month, role):
    model = MonthlyModel(weights, month)
    if role != "val":
        receipt = read(WORK / "FROZEN.json")
        if tag != receipt["stage"] or model.meta["id"] != receipt["months"][f"{month:02}"]:
            raise ValueError("A forecast control requires the frozen monthly model")
    dest = WORK / "forecast_controls" / tag / role / f"{month:02}.parquet"
    if dest.is_file():
        return
    panel, split = panel_cohort(role)
    cfg, raw = data_paths(panel)
    fields, records, _ = months(cfg, raw, month, split)
    eof = model.rep.blocks["wind"]
    area = region_area(model.rep.lat, model.rep.lon)
    source = WORK / "verification" / tag / role / f"{month:02}"
    if read(source / "complete.json")["records"] != len(records):
        raise ValueError("The paired LIM forecast panel has different cases")
    base = pd.read_parquet(source / "forecast.parquet")
    spectra = {
        "native_fourier": daily_fourier_lags(),
        "trained_fourier": calibrated(model, tag, cohort, month),
    }
    coef = {name: {h: linear_weights(lag, h) for h in LEADS} for name, lag in spectra.items()}
    rows = []
    for maps, case in zip(fields, records, strict=True):
        w = maps["wind"]
        mean = model.wind_mean(case["g"], model.rep.encode(maps))
        x = eof.transform(w) - mean
        for h in LEADS:
            t = np.arange(1, len(w) - h)
            row = dict(member=case["member"], year=case["year"], g=case["g"], lead=h)
            for name, coeff in coef.items():
                a, b = coeff[h]
                forecast = eof.inverse(mean + a * x[t] + b * x[t - 1])
                difference = steering(forecast - w[t + h])
                row[name] = np.mean((difference**2 * area).sum(axis=(-2, -1)))
            rows.append(row)
    keys = ["member", "year", "g", "lead"]
    table = base.merge(pd.DataFrame(rows), on=keys, how="inner", validate="one_to_one")
    if len(table) != len(records) * len(LEADS):
        raise ValueError("The Fourier and LIM forecasts do not use the same MPI observations")
    with atomic(dest) as tmp:
        table.to_parquet(tmp, index=False, compression="zstd")
    record("two_day_forecast_control", tag=tag, role=role, month=month, member_years=len(records))


def intervals(tag, role, draws=2000):
    files = [
        WORK / "forecast_controls" / tag / role / f"{month:02}.parquet" for month in range(1, 13)
    ]
    cases = pd.concat(
        [pd.read_parquet(f).assign(month=i) for i, f in enumerate(files, 1)], ignore_index=True
    )
    cluster = ["member", "year", "g", "lead"]
    if not cases.groupby(cluster).month.nunique().eq(12).all():
        raise ValueError("Every forecast member-year requires all calendar months")
    methods = ("lim_mse", "native_fourier", "trained_fourier", "persistence_mse", "mean_mse")
    case = cases.groupby(cluster, as_index=False)[list(methods)].mean()
    comparisons = []
    for h in LEADS:
        block = case[case.lead.eq(h)].sort_values(["member", "year"])
        g = block.g.to_numpy()
        lo, hi = GROUP_LIMITS
        available = set(np.where(g < lo, "cold", np.where(g > hi, "warm", "inside"))) | {"all"}
        for i, regime in enumerate(("all", "cold", "inside", "warm")):
            if regime not in available:
                continue
            w, nm, ny = panel_weights(
                block.member.to_numpy(), block.year.to_numpy(), g, regime, draws, 20261120 + i
            )
            lim = np.sqrt(w @ block.lim_mse.to_numpy())
            for other in methods[1:]:
                old = np.sqrt(w @ block[other].to_numpy())
                record_estimate(
                    comparisons, regime, "steering_rmse", [str(h)], old, lim, nm, ny, "m/s"
                )
                comparisons[-1]["comparator"] = other
    return pd.DataFrame(comparisons)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--role", choices=["val", "fresh", "test", "ssp245_year"], required=True)
    a = p.parse_args()
    run(a.tag, a.weights, a.cohort, a.month, a.role)
