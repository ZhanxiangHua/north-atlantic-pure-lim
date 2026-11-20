"""Recover unresolved ocean variance inside the wind-regression null space."""
import argparse
import os
import time
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from joint import PhysicalValidation
from ocean import FIELDS, ocean_data
from records import WORK, record
from scipy import linalg
from scipy.optimize import least_squares

from annual_lim.layout import DEFAULT_WORK, read
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic, digest, save_json, save_npz, sha
from tc_emulator.warming_circulation import transfer_matrix

METHODS = {
    "own": (0, 0.0, 0.0),
    "scalar": (1, 0.0, 0.05),
    "local": (4, 0.0, 0.03),
    "spatial": (4, 0.5, 0.03),
    "spatial_strong": (4, 0.5, 0.2),
    "spatial_first": (4, 1.5, 0.03),
}


def whitened_maps(model):
    p = model.bg.beta.shape[1]
    result = []
    keys = ("sst", "ocean", "ocean")
    offsets = {"sst": 0, "ocean": model.rep.blocks["sst"].basis.shape[1]}
    for field, key, ch in zip(FIELDS, keys, (0, 0, 1)):
        eof = model.rep.blocks[key]
        wet = eof.valid[ch]
        first = int(eof.valid[:ch].sum())
        sub = eof.basis[first : first + wet.sum()] / eof.weights[ch, wet, None]
        J = np.zeros((wet.size, p))
        cols = np.arange(sub.shape[1]) + offsets[key]
        J[np.ix_(wet.ravel(), cols)] = sub
        result.append(J @ model.bg.lower0)
    return result


def null_groups(model, maps, n):
    # U includes the warming directions. L.T K spans all whitened
    # perturbations visible in the monthly wind, including every wind PC.
    protected = np.column_stack((model.directions, model.bg.lower0.T @ transfer_matrix(model.mean)))
    N = linalg.null_space(protected.T, rcond=1.0e-10)
    if N.shape[1] == 0 or n == 0:
        return []
    if n == 1 or N.shape[1] < 4:
        return [N]
    y, x = np.meshgrid(model.rep.lat, model.rep.lon, indexing="ij")
    area = np.cos(np.deg2rad(y)).ravel()
    core = ((y >= 0) & (y <= 60) & (x >= 260) & (x <= 360)).ravel()
    mats = []
    for jm in maps:
        H = (jm @ N) * np.sqrt(area * core)[:, None]
        C = H.T @ H
        mats.append(C / max(np.trace(C), 1.0e-20))
    # Split SST-sensitive directions first. Within each half split MLD- and
    # gradient-sensitive directions. All four subspaces are mutually orthogonal.
    _, v = np.linalg.eigh(mats[0] - 0.5 * (mats[1] + mats[2]))
    groups = []
    for half in np.array_split(v, 2, axis=1):
        _, u = np.linalg.eigh(half.T @ (mats[1] - mats[2]) @ half)
        groups.extend([N @ half @ z for z in np.array_split(u, 2, axis=1)])
    return groups


def pairs(lat, lon, valid):
    y, x = np.meshgrid(lat, lon, indexing="ij")
    eligible = valid & (y >= 0) & (y <= 60) & (x >= 260) & (x <= 360)
    ids = np.arange(eligible.size).reshape(eligible.shape)
    ij = np.flatnonzero(eligible)
    yield "variance", ij, None
    for distance in (1, 3, 6):
        for axis in (0, 1):
            a = np.roll(eligible, -distance, axis=axis) & eligible
            if axis == 0:
                a[-distance:, :] = False
            else:
                a[:, -distance:] = False
            yield f"step{distance}_{axis}", ids[a], np.roll(ids, -distance, axis=axis)[a]


def design_rows(model, maps, groups, fields, meta):
    y, x = np.meshgrid(model.rep.lat, model.rep.lon, indexing="ij")
    # Equal total influence for SST, transformed MLD, and gradient, independent of original units.
    area = np.cos(np.deg2rad(y)).ravel()
    years, labels = np.unique(meta.year.to_numpy(), return_inverse=True)
    raw_g = meta.groupby("year", sort=True).g.first().to_numpy()
    rows = []
    details = []
    for ch, field in enumerate(FIELDS):
        raw = fields[:, ch].reshape(len(fields), -1).astype(float)
        if field == "mld":
            raw = np.log(np.maximum(0.01, raw))
        key = "sst" if field == "sst" else "ocean"
        valid = model.rep.blocks[key].valid[int(field == "gamma")]
        for desc, a, b in pairs(model.rep.lat, model.rep.lon, valid):
            J = maps[ch][a].copy()
            used = raw[:, a].copy()
            if b is not None:
                J -= maps[ch][b]
                used -= raw[:, b]
            active = np.square(J @ model.directions).sum(axis=1)
            constant = np.square(J).sum(axis=1)
            energies = (
                np.column_stack([np.square(J @ v).sum(axis=1) for v in groups])
                if groups
                else np.empty((len(a), 0))
            )
            for climate in np.array_split(np.argsort(raw_g), 3):
                # First estimate the across-member variance within each year. Different forced means never enter.
                accum = np.zeros(len(a))
                weight = np.zeros(len(a))
                for yi in climate:
                    block = used[labels == yi]
                    finite = np.isfinite(block)
                    count = finite.sum(axis=0)
                    mu = np.where(finite, block, 0.0).sum(axis=0) / np.maximum(count, 1)
                    accum += np.square(np.where(finite, block - mu, 0.0)).sum(axis=0)
                    weight += np.maximum(count - 1, 0)
                obs = accum / np.maximum(weight, 1)
                scale = model.bg_amp * np.exp(
                    model.bg_slope * (model.cal.temperature(raw_g[climate]) - model.bg.center)
                )
                base = constant + (scale.mean() - 1) * active
                keep = (
                    (weight > 0.75 * len(climate) * (meta.member.nunique() - 1))
                    & (obs > 1.0e-10)
                    & (base > 1.0e-10)
                )
                if not keep.any():
                    continue
                floor = 0.03 * np.median(obs[keep])
                w = area[a][keep]
                rows.append(
                    (base[keep], energies[keep], obs[keep], floor, w / w.sum(), b is not None)
                )
                details.append((field, desc, len(climate), int(keep.sum())))
    return rows, details


def optimize(rows, n, spatial, penalty):
    if n == 0:
        return np.zeros(0)

    def error(v):
        out = []
        for base, energy, obs, floor, weight, is_diff in rows:
            strength = spatial / 6 if is_diff else 1.0
            if strength:
                pred = base + energy @ np.expm1(v)
                out.append(np.sqrt(strength * weight / 9) * np.log((pred + floor) / (obs + floor)))
        out.append(np.sqrt(penalty / n) * v)
        return np.concatenate(out)

    fit = least_squares(
        error,
        np.zeros(n),
        bounds=(np.log(0.45), np.log(4.0)),
        max_nfev=90,
        ftol=2.0e-9,
        xtol=2.0e-9,
        gtol=2.0e-9,
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def modified(parent, groups, v):
    obj = copy(parent)
    bg = copy(parent.bg)
    cov = bg.pooled.copy()
    for U, s in zip(groups, v, strict=True):
        load = bg.lower0 @ U
        cov += np.expm1(s) * (load @ load.T)
    L = np.linalg.cholesky(cov)
    active = linalg.solve_triangular(L, bg.lower0 @ parent.directions, lower=True)
    np.testing.assert_allclose(active.T @ active, np.eye(active.shape[1]), atol=1.0e-8)
    bg.lower0, bg.pooled = L, cov
    obj.bg, obj.directions = bg, active
    K = transfer_matrix(parent.mean)
    for g in (parent.cal.g_min - 1, parent.bg.center, parent.cal.g_max + 1):
        a, b = parent.factor(g), obj.factor(g)
        np.testing.assert_allclose(
            K.T @ (a @ a.T) @ K, K.T @ (b @ b.T) @ K, rtol=1.0e-8, atol=1.0e-10
        )
    return obj


def publish(source, out, month, new, recipe, gains):
    parent = source / f"{month:02}"
    folder = out / "weights" / f"{month:02}"
    folder.mkdir(exist_ok=True, parents=True)
    if not (folder / "representation.npz").exists():
        os.link(parent / "representation.npz", folder / "representation.npz")
    with np.load(parent / "parameters.npz") as f:
        values = {k: f[k] for k in f.files}
    values.update(background_root=new.bg.lower0, directions=new.directions)
    save_npz(folder / "parameters.npz", values)
    meta = read(parent / "model.json")
    oldid = meta.pop("id")
    meta["files"]["parameters.npz"] = sha(folder / "parameters.npz")
    meta["training_fingerprint"] = digest([meta["training_fingerprint"], recipe, gains.tolist()])
    meta["id"] = digest(meta)
    save_json(folder / "model.json", meta)
    fit = read(source.parent / "fit" / f"{month:02}.json")
    fit.update(
        id=meta["id"],
        background_parent=oldid,
        spread_recipe=recipe,
        variance_gains=np.exp(gains).tolist(),
    )
    save_json(out / "fit" / f"{month:02}.json", fit)


def run(tag, source, cohort, month, export):
    from types import SimpleNamespace

    tick = time.monotonic()
    dest = WORK / "ocean_spread" / tag / f"{month:02}"
    if not export and (dest / "fits.json").is_file():
        return
    original = MonthlyModel(DEFAULT_WORK / "weights", month)
    parent = MonthlyModel(source, month)
    tr, meta = ocean_data(cohort, month, "train")
    validator = PhysicalValidation(SimpleNamespace(geo=original.rep, month=month))
    maps = whitened_maps(parent)
    cached = {}
    results = []
    fits = {}
    for label in [export] if export else METHODS:
        n, sp, pen = METHODS[label]
        if n not in cached:
            groups = null_groups(parent, maps, n)
            cached[n] = (
                (groups, design_rows(parent, maps, groups, tr, meta)[0]) if groups else ([], [])
            )
        groups, rows = cached[n]
        gains = optimize(rows, len(groups), sp, pen)
        model = modified(parent, groups, gains)
        if export:
            publish(
                source,
                WORK / "exports" / "spread" / (tag + "__" + label),
                month,
                model,
                label,
                gains,
            )
        else:
            results.append(validator.ocean_scores(model).assign(filter=label))
        fits[label] = dict(
            group_ranks=[v.shape[1] for v in groups], variance_gains=np.exp(gains).tolist()
        )
    if not export:
        with atomic(dest / "ocean.parquet") as tmp:
            pd.concat(results, ignore_index=True).to_parquet(tmp, index=False, compression="zstd")
        save_json(
            dest / "fits.json",
            dict(fits=fits, source=str(source), parent=parent.meta["id"], cohort=cohort),
        )
    record(
        "ocean_spread_done",
        cohort=cohort,
        tag=tag,
        month=month,
        export=export,
        methods=len(fits),
        seconds=round(time.monotonic() - tick, 2),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--cohort", default="original")
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--export", choices=METHODS)
    a = p.parse_args()
    run(a.tag, a.weights, a.cohort, a.month, a.export)
