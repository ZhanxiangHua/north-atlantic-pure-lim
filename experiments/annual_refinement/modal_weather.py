"""Calibrate slow/fast and shear/steering components by a similarity transform of the pure LIM."""
import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from records import WORK, record
from regional_wind import extra_metrics
from scipy import linalg
from scipy.optimize import least_squares
from seasonal import calibration
from weather import cache_read, kernels, model_terms, projected, validation, year_table

from annual_lim.layout import DEFAULT_WORK, read
from annual_lim.model import MonthlyModel
from annual_lim.statistics import metrics, trace_lags
from tc_emulator.common import atomic, digest, save_json, save_npz, sha
from tc_emulator.pure_lim_memory import response
from tc_emulator.wind_lim import Dynamics

SPEC = {
    "uniform": ("uniform", 0.4, 0.4, 0.0),
    "time": ("timescale", 0.4, 0.15, 0.02),
    "time_guarded": ("timescale", 0.8, 0.6, 0.1),
    "vertical": ("vertical", 0.4, 0.4, 0.02),
    "vertical_guarded": ("vertical", 0.8, 0.6, 0.1),
    "joint": ("joint", 0.4, 0.4, 0.02),
    "joint_guarded": ("joint", 0.8, 0.6, 0.1),
    "joint_temporal": ("joint", 0.4, 0.05, 0.1),
}


def mode_projectors(dynamics, metric, kind):
    k = dynamics.observed_dim
    identity = np.eye(k)
    if kind == "uniform":
        return [identity]
    L = np.linalg.cholesky(dynamics.sigma[:k, :k])
    # For a 15-day average, each lag h appears 15-h times in each direction.
    product = dynamics.sigma[:, :k].copy()
    slow = 15 * product[:k].copy()
    for h in range(1, 15):
        product = dynamics.a @ product
        slow += (15 - h) * (product[:k] + product[:k].T)
    slow = linalg.solve_triangular(L, slow / 225, lower=True)
    slow = linalg.solve_triangular(L, slow.T, lower=True).T
    mats = [L.T @ metric[i] @ L for i in (0, 5)]
    vertical = mats[1] / np.trace(mats[1]) - mats[0] / np.trace(mats[0])
    _, vec = np.linalg.eigh((slow + slow.T) / 2 if kind != "vertical" else vertical)
    subspaces = np.array_split(vec, 2 if kind == "joint" else 3, axis=1)
    if kind == "joint":
        parts = []
        for basis in subspaces:
            _, v = np.linalg.eigh(basis.T @ vertical @ basis)
            parts.extend(np.array_split(basis @ v, 2, axis=1))
        subspaces = parts
    projectors = [linalg.solve_triangular(L.T, (L @ u @ u.T).T, lower=False).T for u in subspaces]
    np.testing.assert_allclose(sum(projectors), identity, atol=1.0e-9)
    return projectors


def trace_basis(dynamics, metric, parts, horizon=51):
    # For S=sum_i a_i P_i, tr(G S C_h S.T) is a quadratic form in the a_i.
    k = dynamics.observed_dim
    prod = dynamics.sigma[:, :k].copy()
    cov = np.empty((horizon, k * k))
    for h in range(horizon):
        cov[h] = prod[:k].ravel()
        prod = dynamics.a @ prod
    tensor = np.empty((len(parts), len(parts), len(metric), horizon))
    for i, left in enumerate(parts):
        for j, right in enumerate(parts):
            tensor[i, j] = (left.T @ metric @ right).reshape(len(metric), -1) @ cov.T
    direct = trace_lags(dynamics, metric, horizon)
    np.testing.assert_allclose(tensor.sum(axis=(0, 1)), direct, atol=5.0e-9)
    return tensor


def adjust(dynamics, parts, gains):
    S = sum(p * g for p, g in zip(parts, gains, strict=True))
    T = linalg.block_diag(S, S)
    a = np.linalg.solve(T.T, (T @ dynamics.a).T).T
    q = T @ dynamics.q @ T.T
    sig = T @ dynamics.sigma @ T.T
    updated = Dynamics(a, (q + q.T) / 2, (sig + sig.T) / 2, 2, dynamics.observed_dim)
    np.testing.assert_allclose(updated.rho, dynamics.rho, atol=1.0e-8)
    if updated.residual > 1.0e-8:
        raise ValueError("Transformed state does not satisfy the stationary equation")
    return updated


def regional_curves(cal, lag, g, length):
    shape = response(lag, cal.retention, cal.delay, cal.last_lag, cal.feedback, 31)
    ops = np.stack([kernels(int(n))[1][:4] for n in length])
    scale = (cal.normalization * cal.amplitude(g)) ** 2
    return (ops @ shape.T).swapaxes(1, 2) * scale[:, None, None]


def regional_data(cohort, role, month):
    with np.load(WORK / "regional_wind" / cohort / role / f"{month:02}.npz") as f:
        y = f["years"]
        return np.stack([f["within"][y == yr].mean(axis=0) for yr in np.unique(y)])


def fit(model, tensor, data, bg, regional, spec):
    _, g, length = year_table(data)
    _, cw, rw, shrink = spec
    n = tensor.shape[0]

    def convert(x):
        gains = np.exp(np.r_[0.0, x[4:]])
        lag = np.einsum("i,j,ijmh->mh", gains, gains, tensor, optimize=False)
        cal = calibration(model.cal, lag[0], x[:4], g)
        return gains, lag, cal

    def loss(x):
        gains, lag, cal = convert(x)
        raw, center, acf = projected(cal, lag[:5], g, length, bg)
        reg = regional_curves(cal, lag[5:], g, length)
        err = np.log(raw / data["by_year_raw"])
        err[:, -1] *= np.sqrt(2.0)
        return np.r_[
            err.ravel() / np.sqrt(6),
            np.sqrt(cw / 4) * np.log(center[:, :4] / data["by_year_within"][:, :4]).ravel(),
            ((acf[:, :, 1:] - data["by_year_acf"][:, :, 1:]) / np.sqrt(60)).ravel(),
            np.sqrt(rw / 12) * np.log(reg / regional).ravel(),
            np.sqrt(shrink * len(g)) * x[4:],
        ]

    start = [model.cal.intercept, model.cal.slope, model.cal.retention, model.cal.feedback] + [
        0.0
    ] * (n - 1)
    out = least_squares(
        loss,
        start,
        bounds=(
            [-0.85, -0.35, 0.65, 0.0] + [-0.55] * (n - 1),
            [0.85, 0.35, 1.0, 1.25] + [0.55] * (n - 1),
        ),
        max_nfev=120,
        ftol=1.0e-8,
        gtol=1.0e-8,
        xtol=1.0e-8,
    )
    if not out.success:
        raise RuntimeError(out.message)
    return (*convert(out.x), 2 * out.cost / len(g))


def export_weights(source, folder, month, base, cal, name, gains):
    parent = source / f"{month:02}"
    out = folder / "weights" / f"{month:02}"
    out.mkdir(exist_ok=True, parents=True)
    meta = read(parent / "model.json")
    if not (out / "representation.npz").exists():
        os.link(parent / "representation.npz", out / "representation.npz")
    with np.load(parent / "parameters.npz") as f:
        pars = {k: f[k] for k in f.files}
    pars.update(transition=base.a, innovation=base.q, stationary=base.sigma)
    save_npz(out / "parameters.npz", pars)
    ancestor = meta.pop("id")
    meta["files"]["parameters.npz"] = sha(out / "parameters.npz")
    meta["training_fingerprint"] = digest(
        [meta["training_fingerprint"], name, gains.tolist(), asdict(cal)]
    )
    meta["weather"] = asdict(cal)
    meta["id"] = digest(meta)
    save_json(out / "model.json", meta)
    orig = read(source.parent / "fit" / f"{month:02}.json")
    orig.update(
        id=meta["id"],
        weather_parent=ancestor,
        weather_mode_scaling=name,
        mode_gains=gains.tolist(),
        calibration=asdict(cal),
        spectral_radius=base.rho,
        lyapunov_relative_error=base.residual,
    )
    save_json(folder / "fit" / f"{month:02}.json", orig)


def run(tag, weights, cohort, month, export):
    dest = WORK / "modal_weather" / tag / f"{month:02}"
    if not export and (dest / "fits.json").is_file():
        return
    t = time.monotonic()
    m = MonthlyModel(weights, month)
    metric = np.concatenate((metrics(m.rep)[2], extra_metrics(m.rep)))
    tr, va = cache_read(cohort, "train", month), cache_read("original", "val", month)
    gt = year_table(tr)[1]
    yv, gv, nv = year_table(va)
    _, bg, forced = model_terms(m, np.r_[gt, gv])
    train_reg = regional_data(cohort, "train", month)
    val_reg = regional_data("original", "val", month)
    sets = {}
    windows = []
    regions = []
    fits = {}
    for name in [export] if export else SPEC:
        spec = SPEC[name]
        kind = spec[0]
        if kind not in sets:
            p = mode_projectors(m.base, metric, kind)
            sets[kind] = (p, trace_basis(m.base, metric, p))
        p, tensor = sets[kind]
        gains, lag, cal, loss = fit(m, tensor, tr, bg[: len(gt)], train_reg, spec)
        fits[name] = dict(scores=loss, gains=gains.tolist(), parameters=asdict(cal))
        for row in validation(
            cal, va, lag[:5], bg[len(gt) :], forced, -0.0115220616351514, 3.419637285227324
        ):
            windows.append(dict(filter=name, month=month, **row))
        pred = regional_curves(cal, lag[5:], gv, nv)
        for i, year in enumerate(yv):
            for j, dim in enumerate(("shear_ocean", "steer_corridor", "shear_corridor")):
                for k, d in enumerate((1, 5, 10, 15)):
                    regions.append(
                        dict(
                            filter=name,
                            month=month,
                            year=year,
                            g=gv[i],
                            metric=dim,
                            days=d,
                            truth=val_reg[i, j, k],
                            model=pred[i, j, k],
                        )
                    )
        if export:
            updated = adjust(m.base, p, gains)
            np.testing.assert_allclose(
                trace_lags(updated, metric, 51), lag, rtol=1.0e-7, atol=1.0e-8
            )
            export_weights(
                weights,
                WORK / "exports" / "modal" / (tag + "__" + name),
                month,
                updated,
                cal,
                name,
                gains,
            )
    if not export:
        for key, rows in (("windows", windows), ("regions", regions)):
            with atomic(dest / (key + ".parquet")) as tmp:
                pd.DataFrame(rows).to_parquet(tmp, index=False)
        save_json(
            dest / "fits.json",
            dict(parent=m.meta["id"], source=str(weights), cohort=cohort, fits=fits),
        )
    record(
        "modal_weather_done",
        tag=tag,
        cohort=cohort,
        month=month,
        recipes=len(fits),
        export=export,
        seconds=round(time.monotonic() - t, 2),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, default=DEFAULT_WORK / "weights")
    p.add_argument("--cohort", default="original")
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--export", choices=SPEC)
    a = p.parse_args()
    run(a.tag, a.weights, a.cohort, a.month, a.export)
