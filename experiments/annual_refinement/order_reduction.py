"""Compare one and two latent wind lags on the existing development-validation members."""
import argparse
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from fullfit import FILTERS
from ocean import GROUP_LIMITS, data_paths
from paired_intervals import panel_weights, record_estimate
from records import WORK, record
from verify_model import forecast_coefficients
from weather import cache_read, model_terms, projected, train_cal, year_table

from annual_lim.layout import read
from annual_lim.model import MonthlyModel
from annual_lim.statistics import metrics
from annual_lim.train import months
from tc_emulator.common import atomic
from tc_emulator.diagnostics import steering
from tc_emulator.wind_lim import Dynamics


METHODS = ("selected_VAR2", "matched_VAR1", "calibrated_VAR1")
LEADS = (1, 5, 10, 15)


def matched_dynamics(base):
    """A1 = C1 C0^{-1} and Q1 = C0 - A1 C0 A1^T preserve latent lag 0 and lag 1."""
    k = base.observed_dim
    c0 = base.sigma[:k, :k].copy()
    c1 = (base.a @ base.sigma[:, :k])[:k]
    a = np.linalg.solve(c0, c1.T).T
    q = c0 - a @ c0 @ a.T
    short = Dynamics(a, (q + q.T) / 2, c0, 1, k)
    error = float(np.linalg.norm(a @ c0 - c1) / np.linalg.norm(c1))
    if (
        short.rho >= 1
        or short.residual > 1e-10
        or error > 1e-10
        or np.linalg.eigvalsh(short.q).min() < -1e-10 * np.linalg.norm(c0)
    ):
        raise ValueError(
            "Reduced wind transition failed stability, covariance or innovation checks"
        )
    return short, error


def forecast_inputs(baseline, month):
    fields, cases, _ = months(*data_paths("original"), month, "val")
    meta = pd.DataFrame(cases)
    area = metrics(baseline.rep)[0]
    winds = [r["wind"] for r in fields]
    block = baseline.rep.blocks["wind"]
    pcs = [block.transform(w) for w in winds]
    means = [
        baseline.wind_mean(gg, baseline.rep.encode(d)) for gg, d in zip(meta.g, fields)
    ]
    return meta, block, area, winds, pcs, means


def steering_error(w, anomaly, mu, block, area, lead, coef):
    start = np.arange(1, len(w) - lead)
    context = np.column_stack((anomaly[start], anomaly[start - 1]))
    pred = block.inverse(mu + context @ coef.T)
    err = steering(pred - w[start + lead])
    mse = np.mean((err * err * area).sum(axis=(-2, -1)))
    return mse


def evaluate(month):
    destination = WORK / "order_reduction" / f"{month:02}"
    receipt = read(WORK / "FROZEN.json")
    baseline = MonthlyModel(receipt["weights"], month)
    matched = copy(baseline)
    matched.base, error = matched_dynamics(baseline.base)
    calibrated = copy(matched)
    train = cache_read("extended", "train", month)
    val = cache_read("original", "val", month)
    _, train_g, _ = year_table(train)
    years, g, lengths = year_table(val)
    lags, bg, _ = model_terms(matched, train_g)
    calibrated.cal, _ = train_cal(matched, train, lags, bg, *FILTERS["month_weighted"])
    meta, block, area, winds, pcs, means = forecast_inputs(baseline, month)
    forecasts, windows, diagnostics = [], [], []
    for name, model in zip(METHODS, (baseline, matched, calibrated)):
        lags, bg, _ = model_terms(model, g)
        raw, within, _ = projected(model.cal, lags, g, lengths, bg)
        for j, year in enumerate(years):
            for scope, pred, truth in (
                ("total", raw, val["by_year_cov"]),
                ("within", within, val["by_year_within"]),
            ):
                ratio = pred[j, 3] / truth[j, 3]
                windows.append(
                    dict(
                        month=month,
                        year=int(year),
                        g=g[j],
                        method=name,
                        target=scope,
                        ratio=ratio,
                        abs_log_error=abs(np.log(ratio)),
                    )
                )
        coefs = forecast_coefficients(model, LEADS)
        for i, (w, pc, mu) in enumerate(zip(winds, pcs, means)):
            anomaly = pc - mu
            for lead, coef in coefs.items():
                mse = steering_error(w, anomaly, mu, block, area, lead, coef)
                forecasts.append(
                    dict(
                        month=month,
                        member=meta.member[i],
                        year=meta.year[i],
                        g=meta.g[i],
                        method=name,
                        lead=lead,
                        mse=mse,
                    )
                )
        diagnostics.append(
            dict(
                month=month,
                method=name,
                state_dimension=len(model.base.a),
                rho=model.base.rho,
                stationary_residual=model.base.residual,
                latent_lag1_relative_difference=0.0 if name == METHODS[0] else error,
                innovation_min_eigenvalue=np.linalg.eigvalsh(model.base.q).min(),
                retention=model.cal.retention,
                feedback=model.cal.feedback,
                amplitude_intercept=model.cal.intercept,
                amplitude_slope=model.cal.slope,
                normalization=model.cal.normalization,
            )
        )
    case_scores = pd.DataFrame(forecasts)
    archived = pd.read_parquet(
        WORK
        / "verification"
        / receipt["stage"]
        / "val"
        / f"{month:02}/forecast.parquet"
    )
    keys = ["member", "year", "lead"]
    original = (
        case_scores[case_scores.method == METHODS[0]].set_index(keys).sort_index().mse
    )
    reference = archived.set_index(keys).sort_index().lim_mse
    if not original.index.equals(reference.index):
        raise ValueError("Forecast verification member/year order differs")
    np.testing.assert_allclose(original, reference, rtol=2e-8, atol=1e-10)
    for file, frame in (
        ("cases", case_scores),
        ("windows", pd.DataFrame(windows)),
        ("diagnostics", pd.DataFrame(diagnostics)),
    ):
        with atomic(destination / (file + ".parquet")) as temp:
            frame.to_parquet(temp, index=False)
    record(
        "lim_order_validation_month",
        month=month,
        cases=len(meta),
        methods=list(METHODS),
        latent_lag1_relative_difference=error,
    )


def summarize(out):
    parts = []
    for name in ("cases", "windows", "diagnostics"):
        parts.append(
            pd.concat(
                [
                    pd.read_parquet(WORK / "order_reduction" / f"{m:02}/{name}.parquet")
                    for m in range(1, 13)
                ],
                ignore_index=True,
            )
        )
    cases, windows, diagnostics = parts
    agg = (
        cases.groupby(["member", "year", "g", "method", "lead"])
        .mse.mean()
        .unstack(["method", "lead"])
    )
    mi, yr, gs = [agg.index.get_level_values(i).to_numpy() for i in range(3)]
    intervals = []
    for i, group in enumerate(("all", "cold", "inside", "warm")):
        weights, nm, ny = panel_weights(mi, yr, gs, group, 2000, 120748 + i)
        for lead in LEADS:
            ref = np.sqrt(weights @ agg[METHODS[0], lead].to_numpy())
            for name in METHODS[1:]:
                cand = np.sqrt(weights @ agg[name, lead].to_numpy())
                record_estimate(
                    intervals,
                    group,
                    "steering_rmse_ms",
                    [str(lead), name],
                    ref,
                    cand,
                    nm,
                    ny,
                    "raw",
                )
    windows["group"] = np.select(
        [windows.g < GROUP_LIMITS[0], windows.g > GROUP_LIMITS[1]],
        ["cold", "warm"],
        default="inside",
    )
    window_summary = (
        pd.concat([windows, windows.assign(group="all")])
        .groupby(["group", "method", "target"], as_index=False)
        .agg(mean_abs_log=("abs_log_error", "mean"), years=("year", "nunique"))
    )
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("forecasts", pd.DataFrame(intervals)),
        ("windows", window_summary),
        ("diagnostics", diagnostics),
    ):
        frame.to_csv(out / ("lim_order_validation_" + name + ".csv"), index=False)
    cases.to_parquet(
        out / "lim_order_validation_case_scores.parquet",
        index=False,
        compression="zstd",
    )
    windows.to_parquet(
        out / "lim_order_validation_year_scores.parquet",
        index=False,
        compression="zstd",
    )
    record(
        "lim_order_validation_summary",
        forecast_case_scores=len(cases),
        fixed_g_months=len(windows),
        members=len(set(mi)),
        years=len(set(yr)),
        bootstrap_replicates=2000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--month", type=int, choices=range(1, 13))
    task.add_argument("--summarize", type=Path, metavar="OUTPUT_DATA_DIR")
    args = parser.parse_args()
    if args.month:
        evaluate(args.month)
    else:
        summarize(args.summarize)
