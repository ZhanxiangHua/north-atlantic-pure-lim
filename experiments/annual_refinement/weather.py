"""Fit reduced Gaussian weather to raw and within-month wind statistics on training members."""
import argparse
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from records import WORK, record
from scipy.optimize import least_squares

from annual_lim.layout import DEFAULT_WORK
from annual_lim.model import MonthlyModel
from annual_lim.statistics import metrics, trace_lags, variance
from tc_emulator.common import atomic, save_json
from tc_emulator.pure_lim_memory import MemoryCalibration, response
from tc_emulator.warming_circulation import transfer_matrix

LABELS = ("1", "5", "10", "15", "month")


@lru_cache(maxsize=16)
def kernels(n):
    """For a symmetric stationary lag covariance c, each row times c is a wind variance."""
    raw, within = [], []
    for d in (1, 5, 10, 15, n):
        windows = np.zeros((n - d + 1, n))
        for s in range(len(windows)):
            windows[s, s : s + d] = 1 / d
        for op, out in ((windows, raw), (windows - 1 / n, within)):
            gram = op.T @ op / len(op)
            row = np.zeros(31)
            row[0] = np.trace(gram)
            for lag in range(1, n):
                row[lag] = 2 * np.diag(gram, lag).sum()
            out.append(row)
    return np.stack(raw), np.stack(within)


def year_table(data):
    years, idx = np.unique(data["year"], return_index=True)
    assert np.array_equal(years, data["unique"])
    return years, data["g"][idx], data["length"][idx]


def model_terms(model, gs):
    area, ocean, G = metrics(model.rep)
    lags = trace_lags(model.base, G, 51)
    transfer = transfer_matrix(model.mean)
    metric = np.einsum("ic,mcd,jd->mij", transfer, G, transfer, optimize=True)
    background = []
    for g in gs:
        F = model.factor(g)
        background.append(np.einsum("mij,ij->m", metric, F @ F.T))
    slope = model.mean.coefficients[1] / model.bg.scale
    forced = float(slope @ G[0] @ slope)
    return lags, np.array(background), forced


def projected(cal, lags, gs, lengths, bg):
    shape = response(lags, cal.retention, cal.delay, cal.last_lag, cal.feedback, 31)
    scale = (cal.amplitude(gs) * cal.normalization) ** 2
    raw = np.stack([kernels(int(n))[0] @ shape[0] for n in lengths])
    centered = np.stack([kernels(int(n))[1] @ shape[0] for n in lengths])
    total = bg[:, :1] + scale[:, None] * raw
    acvf = bg[:, 1:, None] + scale[:, None, None] * shape[1:, :16]
    return total, scale[:, None] * centered, acvf / acvf[:, :, :1]


def train_cal(
    base, train, lags, bg, first, last, centered_weight, unbiased, lag_weight=1.0, month_weight=1.0
):
    _, g, nd = year_table(train)
    obs = train["by_year_cov" if unbiased else "by_year_raw"]
    obs_center = train["by_year_within"][:, :4]
    observed_acf = train["by_year_acf"][:, :, 1:]
    original = base.cal

    def calibration(x):
        intercept, slope, r, b = map(float, x)
        var = response(lags[0], r, first, last, b, 1)[0]
        return MemoryCalibration(
            "distributed_memory",
            float(g.mean()),
            float(g.min()),
            float(g.max()),
            original.extension,
            intercept,
            slope,
            r,
            delay=first,
            last_lag=last,
            feedback=b,
            normalization=float(np.sqrt(lags[0, 0] / var)),
        )

    def residual(x):
        cal = calibration(x)
        total, center, acf = projected(cal, lags, g, nd, bg)
        err = np.log(total / obs)
        err[:, -1] *= np.sqrt(month_weight)
        parts = [
            err.ravel() / np.sqrt(4 + month_weight),
            np.sqrt(lag_weight / 60) * (acf[:, :, 1:] - observed_acf).ravel(),
        ]
        if centered_weight:
            parts.append(np.sqrt(centered_weight / 4) * np.log(center[:, :4] / obs_center).ravel())
        return np.concatenate(parts)

    opt = least_squares(
        residual,
        [original.intercept, original.slope, original.retention, original.feedback],
        bounds=([-0.7, -0.35, 0.65, 0.0], [0.7, 0.35, 1.0, 1.2]),
        max_nfev=180,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    if not opt.success:
        raise RuntimeError(opt.message)
    return calibration(opt.x), float(2 * opt.cost / len(g))


def validation(cal, data, lags, bg, forced, gmin, gmax):
    years, g, nd = year_table(data)
    ids = np.searchsorted(years, data["year"])
    pred, center, _ = projected(cal, lags, g, nd, bg)
    masks = {
        "all": np.ones(len(ids), bool),
        "cold": data["g"] < gmin,
        "inside": (data["g"] >= gmin) & (data["g"] <= gmax),
        "warm": data["g"] > gmax,
    }
    out = []
    for group, mask in masks.items():
        if not mask.any():
            continue
        actual = variance(data["mu"][mask], data["second"][mask])
        simulated = pred[ids[mask]].mean(axis=0) + np.var(data["g"][mask]) * forced
        actual_center = data["within"][mask, :4].mean(axis=0)
        sim_center = center[ids[mask], :4].mean(axis=0)
        for task, obs, sim in (("total", actual, simulated), ("within", actual_center, sim_center)):
            for k, (o, s) in enumerate(zip(obs, sim)):
                out.append(
                    dict(
                        group=group,
                        target=task,
                        window=LABELS[k],
                        truth=o,
                        model=s,
                        ratio=s / o,
                        abs_log_error=abs(np.log(s / o)),
                    )
                )
    return out


def cache_read(cohort, role, month):
    with np.load(
        WORK / "wind_moments" / cohort / role / f"{month:02}.npz", allow_pickle=False
    ) as f:
        return {k: f[k] for k in f.files}


def run(month, weights, cohort, tag, deep):
    out = WORK / "weather" / tag / ("extended" if deep else "screen") / f"{month:02}.parquet"
    if out.exists():
        return
    start = time.monotonic()
    record("weather_sweep_start", month=month, cohort=cohort, tag=tag, deep=deep)
    train = cache_read(cohort, "train", month)
    val = cache_read("original", "val", month)
    model = MonthlyModel(weights, month)
    _, gt, _ = year_table(train)
    _, gv, _ = year_table(val)
    lags, background, forced = model_terms(model, np.r_[gt, gv])
    b = background[: len(gt)]
    bv = background[len(gt) :]
    cases = {}
    rows = []

    def score(name, cal, loss):
        for r in validation(cal, val, lags, bv, forced, -0.0115220616351514, 3.419637285227324):
            rows.append(dict(month=month, recipe=name, **r))
        cases[name] = {"calibration": asdict(cal), "training_loss": loss}

    score("as_fitted", model.cal, None)
    if deep:
        configs = [
            (d, h, w, u, a, m)
            for d, h in [(2, 12), (3, 15), (4, 18), (5, 20), (7, 17), (9, 20)]
            for w in (0.35, 1.0, 2.0)
            for u in (True,)
            for a in (0.35, 1.0)
            for m in (0.5, 2.0)
        ]
    else:
        configs = [
            (d, h, w, u, 1.0, 1.0)
            for d, h in [(3, 12), (4, 16), (5, 20), (7, 19)]
            for w in (0.0, 0.5, 1.5)
            for u in (False, True)
        ]
    for d, h, w, u, a, m in configs:
        name = f"d{d}-{h}_within{w:g}_unbiased{int(u)}_acf{a:g}_month{m:g}"
        cal, loss = train_cal(model, train, lags, b, d, h, w, u, a, m)
        score(name, cal, loss)
    with atomic(out) as tmp:
        pd.DataFrame(rows).to_parquet(tmp, compression="zstd", index=False)
    save_json(
        out.with_suffix(".json"),
        {"model_id": model.meta["id"], "cohort": cohort, "month": month, "recipes": cases},
    )
    record(
        "weather_sweep_done",
        month=month,
        cohort=cohort,
        tag=tag,
        deep=deep,
        recipes=len(cases),
        elapsed_s=round(time.monotonic() - start, 2),
        file=str(out.relative_to(WORK)),
    )
    print(month, tag, len(cases), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--cohort", default="original")
    p.add_argument("--tag", default="published_backbone")
    p.add_argument("--weights", type=Path, default=DEFAULT_WORK / "weights")
    p.add_argument("--deep", action="store_true")
    args = p.parse_args()
    run(args.month, args.weights, args.cohort, args.tag, args.deep)
