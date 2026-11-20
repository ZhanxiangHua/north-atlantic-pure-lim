"""Compress the second daily wind lag using correlations from the frozen training model."""
import argparse
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from order_reduction import forecast_inputs, steering_error
from paired_intervals import panel_weights, record_estimate
from records import WORK, record
from verify_model import forecast_coefficients
from weather import cache_read, model_terms, projected, year_table

from annual_lim.layout import read
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic
from tc_emulator.wind_lim import Dynamics


RANKS = (0, 12, 36, 72, 96, 120)


def inverse_root(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    if values[0] <= np.finfo(float).eps * values[-1]:
        raise ValueError("Conditional training covariance is singular")
    return (vectors * (1 / np.sqrt(values))) @ vectors.T


def past_directions(base):
    """Condition both past and future on the current day, whiten, then rank by correlation."""
    k = base.observed_dim
    sigma = base.sigma
    if base.order != 2 or len(sigma) != 2 * k:
        raise ValueError("The input must carry two complete wind days")
    next_cov = (base.a @ sigma)[:k]
    c0 = sigma[:k, :k]
    back = np.linalg.solve(c0, sigma[:k, k:]).T
    forward = np.linalg.solve(c0, next_cov[:, :k].T).T
    past_inv = inverse_root(sigma[k:, k:] - back @ sigma[:k, k:])
    future_inv = inverse_root(c0 - forward @ next_cov[:, :k].T)
    conditional_cross = next_cov[:, k:] - next_cov[:, :k] @ back.T
    _, correlations, vh = np.linalg.svd(future_inv @ conditional_cross @ past_inv)
    if correlations[0] > 1 + 1e-9:
        raise ValueError("A conditional canonical correlation is outside its range")
    return back, vh @ past_inv, correlations


def project_state(base, back, axes, rank):
    """Moment-match the one-step transition in the smaller state y = T s."""
    k = base.observed_dim
    t = np.zeros((k + rank, 2 * k))
    t[:k, :k] = np.eye(k)
    t[k:, :k] = -axes[:rank] @ back
    t[k:, k:] = axes[:rank]
    s0 = t @ base.sigma @ t.T
    s1 = t @ base.a @ base.sigma @ t.T
    a = np.linalg.solve(s0, s1.T).T
    q = s0 - a @ s0 @ a.T
    compressed = Dynamics(a, (q + q.T) / 2, (s0 + s0.T) / 2, 1, k)
    # A compact state keeps the visible wind first. Extra coordinates contain past-wind contrasts.
    if (
        compressed.rho >= 1
        or compressed.residual > 1e-9
        or np.linalg.eigvalsh(compressed.q)[0] < -1e-10 * np.linalg.norm(s0)
    ):
        raise ValueError("The compressed model fails stability or covariance checks")
    return compressed


def run(month):
    root = WORK / "partial_wind_memory" / f"{month:02}"
    baseline = MonthlyModel(read(WORK / "FROZEN.json")["weights"], month)
    back, axes, correlations = past_directions(baseline.base)
    meta, block, area, winds, pcs, means = forecast_inputs(baseline, month)
    val = cache_read("original", "val", month)
    years, g, lengths = year_table(val)
    measured = pd.read_parquet(WORK / "order_reduction" / f"{month:02}/cases.parquet")
    old_windows = pd.read_parquet(
        WORK / "order_reduction" / f"{month:02}/windows.parquet"
    )
    cases, windows, diagnostics = [], [], []
    for rank in RANKS:
        model = copy(baseline)
        model.base = project_state(baseline.base, back, axes, rank)
        coef = forecast_coefficients(model, [1])[1]
        mse = np.array(
            [
                steering_error(w, pc - mu, mu, block, area, 1, coef)
                for w, pc, mu in zip(winds, pcs, means)
            ]
        )
        cases.append(
            meta[["member", "year", "g"]].assign(month=month, past_rank=rank, mse=mse)
        )
        lags, bg, _ = model_terms(model, g)
        raw, within, _ = projected(model.cal, lags, g, lengths, bg)
        table = pd.DataFrame(
            dict(
                year=years,
                g=g,
                month=month,
                past_rank=rank,
                total_abs_log=np.abs(np.log(raw[:, 3] / val["by_year_cov"][:, 3])),
                within_abs_log=np.abs(
                    np.log(within[:, 3] / val["by_year_within"][:, 3])
                ),
            )
        )
        windows.append(table)
        diagnostics.append(
            dict(
                month=month,
                past_rank=rank,
                state=len(model.base.a),
                rho=model.base.rho,
                stationary_residual=model.base.residual,
                fraction_squared_correlations=np.sum(correlations[:rank] ** 2)
                / np.sum(correlations**2),
            )
        )
        if rank in (0, baseline.base.observed_dim):
            method = "matched_VAR1" if rank == 0 else "selected_VAR2"
            known = measured[
                (measured.method == method) & (measured.lead == 1)
            ].set_index(["member", "year"])
            keys = pd.MultiIndex.from_frame(meta[["member", "year"]])
            np.testing.assert_allclose(mse, known.reindex(keys).mse, rtol=1e-8)
            ref = old_windows[old_windows.method == method].pivot(
                index="year", columns="target", values="abs_log_error"
            )
            np.testing.assert_allclose(
                table[["total_abs_log", "within_abs_log"]],
                ref.loc[years, ["total", "within"]],
                atol=1e-8,
            )
    for name, frame in (
        ("cases", pd.concat(cases)),
        ("windows", pd.concat(windows)),
        ("fit", pd.DataFrame(diagnostics)),
    ):
        with atomic(root / (name + ".parquet")) as temp:
            frame.to_parquet(temp, index=False)
    record(
        "partial_wind_memory_month",
        month=month,
        ranks=list(RANKS),
        endpoints_checked=True,
    )


def summarize(out):
    tables = []
    for name in ("cases", "windows", "fit"):
        parts = [
            pd.read_parquet(WORK / "partial_wind_memory" / f"{mo:02}/{name}.parquet")
            for mo in range(1, 13)
        ]
        tables.append(pd.concat(parts, ignore_index=True))
    cases, windows, fit = tables
    agg = (
        cases.groupby(["member", "year", "g", "past_rank"])
        .mse.mean()
        .unstack("past_rank")
    )
    mi, yr, g = [agg.index.get_level_values(i).to_numpy() for i in range(3)]
    intervals = []
    for index, group in enumerate(("all", "cold", "inside", "warm")):
        w, nm, ny = panel_weights(mi, yr, g, group, 2000, 120748 + index)
        reference = np.sqrt(w @ agg[120])
        for rank in RANKS:
            cand = np.sqrt(w @ agg[rank])
            record_estimate(
                intervals,
                group,
                "steering_rmse_ms",
                [str(rank)],
                reference,
                cand,
                nm,
                ny,
                "raw",
            )
    forecast = pd.DataFrame(intervals).rename(columns={"dimension": "past_rank"})
    forecast["past_rank"] = forecast.past_rank.astype(int)
    variances = windows.groupby("past_rank", as_index=False)[
        ["total_abs_log", "within_abs_log"]
    ].mean()
    summary = forecast[forecast.group == "all"].merge(variances, on="past_rank")
    for target in ("total_abs_log", "within_abs_log"):
        summary[target + "_difference"] = summary[target] - float(
            variances.loc[variances.past_rank == 120, target].iloc[0]
        )
    summary["meets_rule"] = (
        (summary.ci_high < 0.025)
        & (summary.total_abs_log_difference <= 0.005)
        & (summary.within_abs_log_difference <= 0.005)
    )
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("summary", summary),
        ("forecasts", forecast),
        ("diagnostics", fit),
    ):
        frame.to_csv(out / ("partial_wind_memory_" + name + ".csv"), index=False)
    for name, frame in (("cases", cases), ("windows", windows)):
        frame.to_parquet(
            out / ("partial_wind_memory_" + name + ".parquet"),
            index=False,
            compression="zstd",
        )
    record(
        "partial_wind_memory_summary",
        qualifying_past_ranks=summary.loc[summary.meets_rule, "past_rank"].tolist(),
        member_months=int(len(cases) / len(RANKS)),
        registered_day_one_margin_ms=0.025,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--month", type=int, choices=range(1, 13))
    task.add_argument("--summarize", type=Path, metavar="OUTPUT_DATA_DIR")
    args = parser.parse_args()
    if args.month:
        run(args.month)
    else:
        summarize(args.summarize)
