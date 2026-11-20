"""Refit existing LIM memory parameters with a training loss for directional wind covariances."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from leadlag import (
    LAGS,
    REGIONS,
    areas,
    errors,
    finite_lags,
    observed_covariances,
    target,
    underlying_lags,
    weather_shape,
)
from ocean import GROUP_LIMITS, data_paths
from records import WORK, record
from scipy.optimize import minimize
from weather import cache_read, model_terms, projected, validation, year_table

from annual_lim.layout import case_dir, design, read
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic, lock, save_json, sha
from tc_emulator.pure_lim_memory import response


def training_target(cohort, month):
    cache = WORK / "leadlag/training" / cohort / f"{month:02}.npz"
    with lock(cache.with_suffix(".lock")):
        if not cache.is_file():
            cfg, work = data_paths(cohort)
            cases = [c for c in design(cfg)[1] if c[0] == "train"]
            years = np.unique([c[2] for c in cases])
            counts = np.zeros(len(years), int)
            sums = np.zeros((len(years), 2, len(LAGS), 4, 4))
            for _, member, year, _ in cases:
                folder = case_dir(work, member, year)
                name = f"{month:02}.npz"
                if sha(folder / name) != read(folder / "complete.json")["hashes"][name]:
                    raise ValueError(f"Changed directional training input {folder}")
                with np.load(folder / name, allow_pickle=False) as f:
                    i = np.searchsorted(years, year)
                    sums[i] += observed_covariances(f["wind"], areas(f["lat"], f["lon"]))
                    counts[i] += 1
            if not np.all(counts == 35):
                raise ValueError("Incomplete training ensemble")
            with atomic(cache) as tmp:
                np.savez(tmp, years=years, observed=sums / counts[:, None, None, None, None])
            record("wind_direction_training_target", cohort=cohort, month=month, cases=len(cases))
    with np.load(cache) as f:
        return f["years"], f["observed"]


def simulated(raw, cal, g, nd):
    filtered = weather_shape(raw, cal)
    known = {n: finite_lags(filtered, int(n)) for n in np.unique(nd)}
    return np.stack([known[n] for n in nd]) * cal.amplitude(g)[:, None, None, None, None] ** 2


def study(tag, weights, cohort, month):
    root = WORK / "lag_calibration" / tag / f"{month:02}"
    done = root / "fits.json"
    if done.exists():
        return
    model = MonthlyModel(weights, month)
    base = model.cal
    train = cache_read(cohort, "train", month)
    val = cache_read("original", "val", month)
    y, g, nd = year_table(train)
    ref_years, ref = training_target(cohort, month)
    np.testing.assert_array_equal(ref_years, y)
    lag, bg, _ = model_terms(model, g)
    raw = underlying_lags(model, 31 + base.last_lag)
    initial_var = (
        response(lag[0], base.retention, base.delay, base.last_lag, base.feedback, 1)[0]
        * base.normalization**2
    )

    def parameter(v):
        r, b = map(float, v)
        scale = response(lag[0], r, base.delay, base.last_lag, b, 1)[0]
        return replace(
            base, retention=r, feedback=b, normalization=float(np.sqrt(initial_var / scale))
        )

    def losses(cal):
        total, within, _ = projected(cal, lag, g, nd, bg)
        scalar = 0.5 * (
            np.mean(np.log(total / train["by_year_raw"]) ** 2)
            + np.mean(np.log(within[:, :4] / train["by_year_within"][:, :4]) ** 2)
        )
        err = errors(ref, simulated(raw, cal, g, nd))
        matrix = 0.5 * (
            np.mean(err["all_components"][:, :, 1:4] ** 2)
            + np.mean(err["directional"][:, :, 1:4] ** 2)
        )
        return np.array([scalar, matrix])

    baseline = losses(base)
    fits = {"parent": base}
    details = {}
    for fraction in (0.0, 0.25, 0.60):

        def objective(v):
            rel = losses(parameter(v)) / baseline
            penalty = (
                0.02 * ((v[0] - base.retention) / 0.1) ** 2
                + 0.02 * ((v[1] - base.feedback) / 0.5) ** 2
            )
            return (1 - fraction) * rel[0] + fraction * rel[1] + penalty

        v0 = [base.retention, base.feedback]
        answer = minimize(
            objective,
            v0,
            method="L-BFGS-B",
            bounds=[(0.65, 1.0), (0.0, 1.2)],
            options={"ftol": 1e-10},
        )
        optimum = answer.x if np.isfinite(answer.fun) and answer.fun <= objective(v0) else v0
        fits[f"direction_{fraction:g}"] = parameter(optimum)
    held = target(month, "val")
    _, gv, nv = year_table(val)
    _, bval, force = model_terms(model, gv)
    windows = []
    direction = []
    low, high = GROUP_LIMITS
    for name, cal in fits.items():
        predicted = simulated(raw, cal, held["g"], held["length"])
        for band, mask in (
            ("all", np.ones(len(predicted), bool)),
            ("cold", held["g"] < low),
            ("inside", (held["g"] >= low) & (held["g"] <= high)),
            ("warm", held["g"] > high),
        ):
            scores = errors(
                held["observed"][mask].mean(axis=0, keepdims=True),
                predicted[mask].mean(axis=0, keepdims=True),
            )
            for metric, values in scores.items():
                for j, region in enumerate(REGIONS):
                    for k, h in enumerate(LAGS[1:], start=1):
                        direction.append(
                            dict(
                                method=name,
                                group=band,
                                region=region,
                                lag=h,
                                metric=metric,
                                error=values[0, j, k],
                            )
                        )
        ws = validation(cal, val, lag, bval, force, low, high)
        windows.extend(dict(method=name, **row) for row in ws)
        loss = losses(cal)
        details[name] = dict(
            retention=cal.retention,
            feedback=cal.feedback,
            normalization=cal.normalization,
            training_window_loss=loss[0],
            training_matrix_loss=loss[1],
        )
    for filename, rows in (("windows", windows), ("direction", direction)):
        with atomic(root / (filename + ".parquet")) as tmp:
            pd.DataFrame(rows).to_parquet(tmp, index=False)
    save_json(done, details)
    record(
        "lag_calibration_validated",
        tag=tag,
        cohort=cohort,
        month=month,
        training_years=len(y),
        methods=len(fits),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--month", type=int, required=True)
    a = p.parse_args()
    study(a.tag, a.weights, a.cohort, a.month)
