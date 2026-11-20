"""Fit the warming response of wind-relevant background covariance at fixed predictive regression."""
import argparse
import os
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from backbone import Bank
from fullfit import CASES
from joint import PhysicalValidation
from records import WORK, record
from scipy import linalg
from scipy.optimize import least_squares
from weather import cache_read, model_terms, projected, validation, year_table

from annual_lim.layout import read
from annual_lim.model import MonthlyModel
from annual_lim.statistics import metrics
from tc_emulator.common import atomic, digest, save_json, save_npz, sha
from tc_emulator.warming_circulation import transfer_matrix


def active_geometry(model, metric, whitened, fractions):
    load = model.bg.lower0.T @ transfer_matrix(model.mean)
    energy, dirs = np.linalg.eigh(load @ metric[0] @ load.T)
    e = np.maximum(energy[::-1], 0.0)
    dirs = dirs[:, ::-1]
    weights = load @ metric @ load.T
    base = np.trace(weights, axis1=1, axis2=2)
    out = {}
    for frac in fractions:
        rank = 1 + int(np.searchsorted(np.cumsum(e) / e.sum(), frac))
        U = dirs[:, :rank]
        by_mode = np.stack([np.square(a @ U).sum(axis=0) / (len(a) - 1) for a in whitened])
        average = by_mode @ e[:rank] / e[:rank].sum()
        visible = np.einsum("ir,kij,jr->k", U, weights, U, optimize=True)
        out[frac] = (U, average, visible, base)
    return out


def update(source, root, month, model, label):
    parent = source / f"{month:02}"
    out = root / "weights" / f"{month:02}"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "representation.npz").exists():
        os.link(parent / "representation.npz", out / "representation.npz")
    with np.load(parent / "parameters.npz") as archive:
        values = {k: archive[k] for k in archive.files}
    values.update(
        directions=model.directions, background_response=np.array([model.bg_amp, model.bg_slope])
    )
    save_npz(out / "parameters.npz", values)
    meta = read(parent / "model.json")
    ident = meta.pop("id")
    meta["training_fingerprint"] = digest(
        [meta["training_fingerprint"], label, model.bg_amp, model.bg_slope]
    )
    meta["files"]["parameters.npz"] = sha(out / "parameters.npz")
    meta["id"] = digest(meta)
    save_json(out / "model.json", meta)
    info = read(source.parent / "fit" / f"{month:02}.json")
    info.update(
        id=meta["id"],
        active_parent=ident,
        active_calibration=label,
        active_rank=model.directions.shape[1],
        active_variance_center=model.bg_amp,
        active_variance_slope=model.bg_slope,
    )
    save_json(root / "fit" / f"{month:02}.json", info)


def run(tag, month, export=None):
    parent = read(WORK / "parents" / (tag + ".json"))
    cohort = parent["cohort"]
    source = Path(parent["source"])
    model = MonthlyModel(source, month)
    if model.meta["format"] != 1:
        raise ValueError("The active-only comparison uses a single covariance factor")
    dest = WORK / "active_warming" / tag / f"{month:02}"
    if not export and (dest / "fits.json").is_file():
        return
    bank = Bank(cohort, month)
    val = PhysicalValidation(bank)
    z = bank.z(CASES[parent["name"]], "train")
    years = bank.scores["train_year"]
    residual = [z[years == y] - z[years == y].mean(axis=0) for y in np.unique(years)]
    white = [linalg.solve_triangular(model.bg.lower0, block.T, lower=True).T for block in residual]
    G = metrics(model.rep)[2]
    geometry = active_geometry(model, G, white, [0.55, 0.80, 0.95])
    train = cache_read(cohort, "train", month)
    held = cache_read("original", "val", month)
    _, gt, nd = year_table(train)
    _, gv, nv = year_table(held)
    lag, bb, force = model_terms(model, np.r_[gt, gv])
    wx = projected(model.cal, lag, gt, nd, np.zeros_like(bb[: len(gt)]))[0]
    # The training target is raw wind, averaged within homologous grid cells at each single warming year.
    # A separate target follows the variance of (K-transformed) real backgrounds at those same years.
    dx = gt - model.bg.center
    valdx = model.cal.temperature(gv) - model.bg.center
    details = {}
    windrows = []
    oceanrows = []
    for frac, (U, truth, visible, base) in geometry.items():
        for weather_weight in (0.0, 0.35, 1.0, 4.0, 16.0):
            for book in ("raw",) if weather_weight == 0 else ("raw", "member_corrected"):
                label = f"f{round(100*frac):02}_w{weather_weight:g}_{book}"
                if export and label != export:
                    continue
                target = train["by_year_raw" if book == "raw" else "by_year_cov"]

                def residuals(x):
                    law = np.exp(x[0] + x[1] * dx)
                    background = base[0] + (law - 1) * visible[0]
                    wind = background[:, None] + wx
                    err = np.log(wind / target)
                    wf = np.sqrt([0.05, 0.13, 0.20, 0.30, 0.32])
                    # Preserve the empirical covariance fit while allowing the training wind variance to influence it.
                    return np.r_[
                        np.log(law / truth),
                        np.sqrt(weather_weight) * (err * wf).ravel(),
                        0.15 * np.sqrt(len(gt)) * x[1],
                    ]

                fit = least_squares(
                    residuals,
                    [np.log(model.bg_amp), model.bg_slope],
                    bounds=([-1.0, -0.65], [0.8, 0.65]),
                    ftol=1.0e-10,
                    xtol=1.0e-10,
                    gtol=1.0e-10,
                )
                if not fit.success:
                    raise RuntimeError(fit.message)
                modified = copy(model)
                modified.directions = U
                modified.bg_amp = float(np.exp(fit.x[0]))
                modified.bg_slope = float(fit.x[1])
                law = np.exp(fit.x[0] + fit.x[1] * valdx)
                Cb = base + (law[:, None] - 1) * visible
                check = modified.factor(gv[-1])
                K = transfer_matrix(modified.mean)
                np.testing.assert_allclose(
                    np.trace(G[0] @ (K.T @ check @ check.T @ K)), Cb[-1, 0], rtol=5.0e-7
                )
                details[label] = dict(
                    rank=len(U.T),
                    variance_at_center=modified.bg_amp,
                    log_slope=modified.bg_slope,
                    cost=2 * fit.cost / len(gt),
                )
                if export:
                    update(
                        source,
                        WORK / "exports/active" / (tag + "__" + label),
                        month,
                        modified,
                        label,
                    )
                else:
                    windrows.extend(
                        dict(filter=label, month=month, **r)
                        for r in validation(
                            model.cal, held, lag, Cb, force, -0.0115220616351514, 3.419637285227324
                        )
                    )
                    oceanrows.append(val.ocean_scores(modified).assign(filter=label))
    if not details:
        raise ValueError("Unknown covariance recipe")
    if not export:
        with atomic(dest / "windows.parquet") as tmp:
            pd.DataFrame(windrows).to_parquet(tmp, index=False, compression="zstd")
        with atomic(dest / "ocean.parquet") as tmp:
            pd.concat(oceanrows, ignore_index=True).to_parquet(tmp, index=False, compression="zstd")
        save_json(dest / "fits.json", details)
    record(
        "warming_covariance_transfer",
        tag=tag,
        month=month,
        cohort=cohort,
        methods=len(details),
        export=export,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--export")
    a = p.parse_args()
    run(a.tag, a.month, a.export)
