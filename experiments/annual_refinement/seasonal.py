"""Partially pool month-specific weather coefficients around the 12-month calendar."""
import argparse
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from records import WORK, record
from scipy.optimize import least_squares
from weather import cache_read, model_terms, projected, validation, year_table

from annual_lim.layout import DEFAULT_WORK, read
from annual_lim.model import MonthlyModel
from annual_lim.statistics import variance
from tc_emulator.common import atomic, digest, save_json
from tc_emulator.pure_lim_memory import response

# The fitting observations in every objective come exclusively from training member-years.
OBJECTIVES = {
    "local": (0.0, 0.0, False, 1.0),
    "within": (0.4, 0.0, False, 1.0),
    "pooled": (0.4, 1.0, False, 1.0),
    "pooled_month": (0.2, 2.0, False, 3.0),
    "balanced": (0.4, 1.0, True, 1.0),
}
SMOOTHING = {
    "own": (0.0, ()),
    "slopes": (2.0, (1,)),
    "slopes_memory": (2.0, (1, 2, 3)),
    "all_light": (0.35, (0, 1, 2, 3)),
    "all": (2.0, (0, 1, 2, 3)),
    "all_strong": (15.0, (0, 1, 2, 3)),
}


def cyclic(parameters, strength, columns):
    """Minimize ||b-x||² + strength ||D2 b||², with December next to January."""
    if not columns:
        return parameters.copy()
    eye = np.eye(12)
    second = np.roll(eye, 1, axis=0) + np.roll(eye, -1, axis=0) - 2 * eye
    out = parameters.copy()
    out[:, columns] = np.linalg.solve(eye + strength * second.T @ second, out[:, columns])
    # Calendar smoothing can slightly overshoot a local extremum. Restore physical bounds.
    return np.clip(out, [-0.7, -0.35, 0.65, 0.0], [0.7, 0.35, 1.0, 1.2])


def calibration(initial, lag, x, g):
    intercept, slope, retention, feedback = map(float, x)
    adjusted = response(lag, retention, 5, 20, feedback, 1)[0]
    return replace(
        initial,
        intercept=intercept,
        slope=slope,
        retention=retention,
        feedback=feedback,
        center=float(g.mean()),
        g_min=float(g.min()),
        g_max=float(g.max()),
        delay=5,
        last_lag=20,
        normalization=float(np.sqrt(lag[0] / adjusted)),
    )


def fit_month(model, cache, lag, bg, forced, objective):
    w_center, w_pool, balance, w_month = OBJECTIVES[objective]
    years, g, length = year_table(cache)
    density = np.exp(-0.5 * ((g[:, None] - g[None, :]) / 0.55) ** 2).sum(axis=1)
    freq = (1.0 / density) if balance else np.ones_like(g)
    freq /= freq.mean()
    by_year = cache["by_year_raw"]
    within = cache["by_year_within"][:, :4]
    acf = cache["by_year_acf"][:, :, 1:]
    # Each block pools complete years and all fitting ensemble members. The
    # empirical target subtracts the space-dependent mean map over that block.
    panels = [np.arange(len(g)), *np.array_split(np.argsort(g), 3)]
    obs = np.stack(
        [
            variance(
                cache["mu"][np.isin(cache["year"], years[p])],
                cache["second"][np.isin(cache["year"], years[p])],
            )
            for p in panels
        ]
    )
    weights = np.ones(5)
    weights[-1] = w_month
    weights = np.sqrt(weights / weights.sum())

    def residual(x):
        par = calibration(model.cal, lag[0], x, g)
        raw, center, auto = projected(par, lag, g, length, bg)
        parts = [
            (np.sqrt(freq[:, None]) * weights * np.log(raw / by_year)).ravel(),
            (np.sqrt(freq[:, None, None] / 60) * (auto[:, :, 1:] - acf)).ravel(),
        ]
        if w_center:
            parts.append(
                (np.sqrt(w_center * freq[:, None] / 4) * np.log(center[:, :4] / within)).ravel()
            )
        if w_pool:
            pred = np.stack([raw[p].mean(axis=0) + np.var(g[p]) * forced for p in panels])
            parts.append(
                (np.sqrt(w_pool * len(g) / len(panels)) * weights * np.log(pred / obs)).ravel()
            )
        return np.concatenate(parts)

    start = [model.cal.intercept, model.cal.slope, model.cal.retention, model.cal.feedback]
    sol = least_squares(
        residual,
        start,
        bounds=([-0.7, -0.35, 0.65, 0.0], [0.7, 0.35, 1.0, 1.2]),
        ftol=1.0e-9,
        gtol=1.0e-9,
        xtol=1.0e-9,
        max_nfev=150,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.x


def copy_with_calibration(source, target, month, cal, recipe):
    folder = f"{month:02}"
    meta = read(source / folder / "model.json")
    out = target / "weights" / folder
    out.mkdir(exist_ok=True, parents=True)
    for name in meta["files"]:
        if not (out / name).exists():
            os.link(source / folder / name, out / name)
    ancestor = meta.pop("id")
    meta["training_fingerprint"] = digest([meta["training_fingerprint"], recipe, asdict(cal)])
    meta["weather"] = asdict(cal)
    meta["id"] = digest(meta)
    save_json(out / "model.json", meta)
    original = read(source.parent / "fit" / (folder + ".json"))
    original.update(
        id=meta["id"], calibration=asdict(cal), weather_parent=ancestor, seasonal_recipe=recipe
    )
    save_json(target / "fit" / (folder + ".json"), original)
    return meta["id"]


def run(tag, source, cohort, export):
    root = WORK / "seasonal" / tag
    validation_files = sorted(root.glob("*/windows.parquet"))
    if len(validation_files) == 12 and not export:
        return
    tick = time.monotonic()
    record("seasonal_start", tag=tag, cohort=cohort, export=export)
    # Retain only the compact traces needed outside each month's physical-data fit.
    months = []
    params = {key: [] for key in OBJECTIVES}
    for month in range(1, 13):
        model = MonthlyModel(source, month)
        tr, va = cache_read(cohort, "train", month), cache_read("original", "val", month)
        gt, gv = year_table(tr)[1], year_table(va)[1]
        lags, bg, forced = model_terms(model, np.r_[gt, gv])
        options = [export.split("__")[0]] if export else OBJECTIVES
        for key in options:
            params[key].append(fit_month(model, tr, lags, bg[: len(gt)], forced, key))
        months.append((model.cal, gt, lags, bg[len(gt) :], forced))
    solutions = {}
    for objective, seq in params.items():
        if not seq:
            continue
        for smooth, (strength, columns) in SMOOTHING.items():
            name = objective + "__" + smooth
            if export and name != export:
                continue
            solutions[name] = cyclic(np.stack(seq), strength, columns)
    for month, (init, gt, lags, bb, forced) in enumerate(months, 1):
        root_m = root / f"{month:02}"
        validation_set = cache_read("original", "val", month)
        rows = []
        fits = {}
        for name, seq in solutions.items():
            cal = calibration(init, lags[0], seq[month - 1], gt)
            fits[name] = asdict(cal)
            if export:
                dest = WORK / "exports" / "seasonal" / (tag + "__" + name)
                copy_with_calibration(source, dest, month, cal, name)
            else:
                for row in validation(
                    cal, validation_set, lags, bb, forced, -0.0115220616351514, 3.419637285227324
                ):
                    rows.append(dict(month=month, filter=name, **row))
        if not export:
            with atomic(root_m / "windows.parquet") as tmp:
                pd.DataFrame(rows).to_parquet(tmp, index=False)
            save_json(root_m / "fits.json", fits)
    if not export:
        save_json(
            root / "source.json",
            dict(
                tag=tag,
                weights=str(source),
                cohort=cohort,
                model_ids=[read(source / f"{m:02}/model.json")["id"] for m in range(1, 13)],
            ),
        )
    record(
        "seasonal_done",
        tag=tag,
        cohort=cohort,
        combos=len(solutions),
        export=export,
        seconds=round(time.monotonic() - tick, 2),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, default=DEFAULT_WORK / "weights")
    p.add_argument("--cohort", default="original")
    p.add_argument("--export")
    a = p.parse_args()
    if a.export and a.export not in [o + "__" + s for o in OBJECTIVES for s in SMOOTHING]:
        p.error("Unknown seasonal objective and smoothing pair")
    run(a.tag, a.weights, a.cohort, a.export)
