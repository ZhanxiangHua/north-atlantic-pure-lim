"""Calibrate shear variance and daily changes on September training members, then score validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from annual_lim.layout import CONFIG, case_dir, design, read
from annual_lim.model import MonthlyModel
from annual_lim.statistics import trace_lags
from benchmark_mpi2050 import MONTH, REPORT, SEED, WORK, corridor_weights, weather_metrics
from tc_emulator.common import ROOT, atomic, save_json, sha
from tc_emulator.diagnostics import region_area, steering
from tc_emulator.pure_lim_memory import response


@lru_cache(maxsize=1)
def setup(weights):
    model = MonthlyModel(weights, MONTH)
    return model, region_area(model.rep.lat, model.rep.lon), corridor_weights(model.rep.lat, model.rep.lon)


def fields(work, member, year):
    directory = case_dir(work, member, year)
    file = directory / "09.npz"
    if sha(file) != read(directory / "complete.json")["hashes"][file.name]:
        raise ValueError(f"Training/validation field checksum mismatch for {file}")
    return np.load(file, allow_pickle=False)


def fit_gain(weights, work, cases):
    model, _, area = setup(weights)
    eof = model.rep.blocks["wind"]
    k = model.base.observed_dim
    maps = eof.inverse(np.eye(k)) - eof.inverse(np.zeros((1, k)))
    contrast = ((maps[:, :2] - maps[:, 2:]) * np.sqrt(area / 2)).reshape(k, -1)
    lags = trace_lags(model.base, (contrast @ contrast.T)[None], 53)[0]
    rows = []
    for _, member, year, g in cases:
        with fields(work, member, year) as sample:
            wind = sample["wind"].astype(float)
        d = wind[:, :2] - wind[:, 2:]
        observed = np.sum(d.var(axis=0).mean(axis=0) * area)
        # E[monthly sample variance] = one-day marginal variance minus variance of its 30-day mean.
        if len(wind) != 30:
            raise ValueError("Expected 30 September daily means")
        change = np.sum(np.square(np.diff(d, axis=0)).mean(axis=(0, 1)) * area)
        rows.append(dict(member=member, year=year, observed=observed, daily_change=change, g=g))
    frame = pd.DataFrame(rows)
    cal = model.cal
    cov = response(lags, cal.retention, cal.delay, cal.last_lag, cal.feedback, 31)
    cov *= cal.normalization**2 * np.mean(cal.amplitude(frame.g.to_numpy())**2)

    def moments(b):
        # Cov(d_t + b d_{t-1}, d_{t+h} + b d_{t+h-1}) for the two-component contrast.
        lag = np.arange(30)
        c = (1+b**2)*cov[lag] + b*(cov[abs(lag-1)] + cov[lag+1])
        monthly = (30*c[0] + 2*(np.arange(29, 0, -1) @ c[1:])) / 30**2
        return c[0] - monthly, 2*(c[0] - c[1])

    target = frame.observed.mean()
    ratio = frame.daily_change.mean() / target
    def residual(b):
        variance, change = moments(b)
        return change / variance - ratio
    memory = brentq(residual, -0.99, 0.99)
    return dict(gain=float(np.sqrt(target/moments(0)[0])), lag_weight=float(memory),
                lagged_gain=float(np.sqrt(target/moments(memory)[0])),
                training_reference=float(target), training_lim=float(moments(0)[0]),
                daily_change_reference=float(frame.daily_change.mean()), daily_change_lim=float(moments(0)[1]))


def score(task):
    work, weights, fit, member, year, g = task
    model, basin, area = setup(weights)
    with fields(work, member, year) as sample:
        z = model.rep.encode(sample)
        wind = sample["wind"]
    eof = model.rep.blocks["wind"]
    mean = eof.inverse(model.wind_mean(g, z)[None])[0]
    mean_d = mean[:2] - mean[2:]
    ident = f"shear-validation/{member}"
    outputs = []
    def add(method, raw):
        m = weather_metrics(raw, basin, area, model.rep.lon)
        d = raw[:, :2].astype(float) - raw[:, 2:]
        speed = np.hypot(d[:, 0], d[:, 1])
        outputs.append(dict(method=method, member=member, year=year, shear_variance=np.sum(d.var(axis=0).mean(axis=0)*area),
                            daily_change=np.sum(np.square(np.diff(d, axis=0)).mean(axis=(0, 1))*area),
                            weak_shear_1d_pct=100*np.sum((speed < 8).mean(axis=0)*area),
                            shear_q90_ms=m["shear_q90_ms"], weak_shear_3d_pct=m["weak_shear_3d_pct"]))
    add("MPI daily", wind)
    max_change = 0.0
    for draw in range(16):
        pc = model.pc_path(g, z, len(wind) + 1, year, SEED, ident, draw)
        padded = eof.inverse(pc)
        base = padded[1:]
        d = (padded[:, :2] - padded[:, 2:]) - mean_d
        add("Pure LIM", base)
        for name, scale, lag in [("Shear gain", fit["gain"], 0.),
                                 ("Shear gain and lag", fit["lagged_gain"], fit["lag_weight"])]:
            correction = (scale - 1)*d[1:] + scale*lag*d[:-1]
            candidate = base.copy()
            candidate[:, :2] += 0.8 * correction
            candidate[:, 2:] -= 0.2 * correction
            max_change = max(max_change, float(np.max(abs(steering(candidate)-steering(base)))))
            add(name, candidate)
    return outputs, max_change


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annual-work", type=Path, default=ROOT / "runs/annual_lim")
    p.add_argument("--weights", type=Path, default=ROOT / "weights")
    p.add_argument("--output", type=Path, default=WORK)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    _, cases = design(CONFIG)
    train = [c for c in cases if c[0] == "train"]
    val = [c for c in cases if c[0] == "val"]
    fit = fit_gain(a.weights, a.annual_work, train)
    args = [(a.annual_work, a.weights, fit, *c[1:]) for c in val]
    rows, maximum = [], 0.
    with ProcessPoolExecutor(a.workers) as pool:
        for scores, delta in pool.map(score, args):
            rows.extend(scores)
            maximum = max(maximum, delta)
    per_case = pd.DataFrame(rows).groupby(["member", "year", "method"], as_index=False).mean(numeric_only=True)
    with atomic(a.output / "gain_validation.csv") as d:
        per_case.to_csv(d, index=False)
    save_json(a.output / "gain_fit.json", dict(**fit, september_training_months=len(train),
              validation_months=len(val), validation_draws_per_month=16, max_steering_change_ms=maximum))


if __name__ == "__main__":
    main()
