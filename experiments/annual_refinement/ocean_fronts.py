"""Fit front-weighted residual ocean EOFs and compare physical gradients on development members."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from features import load_block
from ocean import FIELDS, GROUP_LIMITS, joint_input, ocean_data
from paired_intervals import panel_weights, record_estimate
from records import WORK, record
from scipy import linalg
from scipy.stats import rankdata

from annual_lim.layout import DEFAULT_WORK
from annual_lim.model import MonthlyModel
from annual_lim.statistics import metrics
from tc_emulator.common import atomic

ROOT = WORK / "ocean_fronts"
STRENGTHS = (0.0, 0.5, 1.0)


def edges(lat, lon):
    """Adjacent centers, separated in units of 100 km on a sphere of radius 6371 km."""
    south, north = np.s_[:-1, :], np.s_[1:, :]
    west, east = np.s_[:, :-1], np.s_[:, 1:]
    km_per_radian = 6371.0 / 100.0
    dx = km_per_radian * np.cos(np.deg2rad(lat))[:, None] * np.abs(np.diff(np.deg2rad(lon)))
    dy = km_per_radian * np.abs(np.diff(np.deg2rad(lat)))[:, None]
    if np.any(dx <= 0) or np.any(dy <= 0):
        raise ValueError("Grid center coordinates must be distinct")
    return [(west, east, dx), (south, north, dy)]


def gradient_percentile(climatology, wet, neighbors):
    """Percentile of the RMS length-normalized difference to available wet neighbors."""
    total, count = np.zeros(wet.shape), np.zeros(wet.shape, dtype=int)
    for left, right, distance in neighbors:
        usable = (
            wet[left]
            & wet[right]
            & np.isfinite(climatology[left])
            & np.isfinite(climatology[right])
        )
        diff = np.where(usable, (climatology[left] - climatology[right]) / distance, 0)
        for cell in (left, right):
            total[cell] += diff**2
            count[cell] += usable
    used = count > 0
    gradient = np.sqrt(total[used] / count[used])
    pct = np.zeros(wet.shape)
    pct[used] = (rankdata(gradient, method="average") - 0.5) / len(gradient)
    return pct


def front_basis(parent, maps, percentiles, rank, strength):
    """Protect coarse coordinates under the deployment metric while weighting fine-scale training."""
    u = parent.basis
    valid = parent.valid
    x = maps.reshape(len(maps), -1)[:, valid.ravel()].astype(float)
    x = np.where(np.isfinite(x), x - parent.mean[valid], 0.0) * parent.weights[valid]
    residual = x - (x @ u) @ u.T
    factors = 1 + strength * percentiles
    for j, keep in enumerate(valid):
        metric = parent.weights[j, keep] ** 2
        factors[j] /= np.sqrt(np.average(factors[j, keep] ** 2, weights=metric))
    f = factors[valid]
    target = residual * f
    extra = rank - u.shape[1]
    val, vec = linalg.eigh(
        target @ target.T, subset_by_index=[len(x) - extra, len(x) - 1], check_finite=False
    )
    if val[0] <= 0:
        raise ValueError("Insufficient rank in the front-weighted training maps")
    modes = ((target.T @ vec[:, ::-1]) / np.sqrt(val[::-1])) / f[:, None]
    modes -= u @ (u.T @ modes)
    tail = linalg.qr(modes, mode="economic", check_finite=False)[0]
    basis = np.column_stack((u, tail))
    np.testing.assert_allclose(basis.T @ basis, np.eye(rank), atol=2e-9)
    return replace(
        parent,
        basis=basis,
        variance=np.var(x @ basis, axis=0, ddof=1),
        total_variance=float(np.sum(x**2) / (len(x) - 1)),
    )


def score(errors, wet, pct, area, neighbors):
    def weighted_mse(values, weights):
        valid = np.isfinite(values)
        numerator = np.sum(np.where(valid, values, 0.0) ** 2 * weights, axis=(-2, -1))
        denominator = np.sum(valid * weights, axis=(-2, -1))
        return numerator, denominator

    out = {}
    for name, region in (("basin_rmse", wet), ("front_rmse", wet & (pct >= 0.8))):
        num, den = weighted_mse(errors, area * region)
        out[name] = np.sqrt(num / den)
    num = np.zeros(len(errors))
    den = np.zeros(len(errors))
    for left, right, dist in neighbors:
        w = np.minimum(area[left], area[right]) * (wet[left] & wet[right])
        diff = (errors[(slice(None), *left)] - errors[(slice(None), *right)]) / dist
        n, d = weighted_mse(diff, w)
        num += n
        den += d
    out["gradient_rmse_per_100km"] = np.sqrt(num / den)
    if not np.isfinite(np.column_stack(list(out.values()))).all():
        raise ValueError("Incomplete validation score")
    return out


def run(month):
    dest = ROOT / f"{month:02}.parquet"
    if dest.is_file():
        return
    model = MonthlyModel(DEFAULT_WORK / "weights", month)
    area = metrics(model.rep)[0]
    neighbors = edges(model.rep.lat, model.rep.lon)
    training, tr = ocean_data("extended", month, "train")
    actual, meta = ocean_data("original", month, "val")
    if tr.member.nunique() != 35 or tr.year.nunique() != 57 or set(tr.member) & set(meta.member):
        raise ValueError("Expected 57 training years and disjoint development members")
    if len(meta) != 42 or meta.member.nunique() != 7:
        raise ValueError("Expected the seven-member, six-year development panel")
    finite = np.isfinite(training)
    climatology = np.nansum(training, axis=0, dtype=float) / np.maximum(1, finite.sum(axis=0))
    folder = WORK / "features" / "extended" / f"{month:02}"
    rows = []
    for name, ids, rank in (("sst", [0], 48), ("ocean", [1, 2], 72)):
        parent = model.rep.blocks[name]
        pct = np.stack(
            [
                gradient_percentile(climatology[i], parent.valid[j], neighbors)
                for j, i in enumerate(ids)
            ]
        )
        fit = training[:, :1] if name == "sst" else joint_input(training)
        held = actual[:, :1] if name == "sst" else joint_input(actual)
        reference = load_block(folder / f"nested_{name}.npz")
        for strength in STRENGTHS:
            block = front_basis(parent, fit, pct, rank, strength)
            z = block.transform(held)
            np.testing.assert_allclose(z[:, :18], parent.transform(held), atol=1.0e-10)
            decoded = block.inverse(z)
            if strength == 0:
                # Compare reconstructed maps to the original nested fit. EOF signs are arbitrary.
                expected = reference.inverse(reference.transform(held, k=rank))
                np.testing.assert_allclose(decoded, expected, atol=3e-6, rtol=2e-6, equal_nan=True)
            for j, fi in enumerate(ids):
                pred = decoded[:, j].astype(float)
                if FIELDS[fi] == "mld":
                    pred = np.maximum(0.01, np.exp(pred))
                if FIELDS[fi] == "gamma":
                    pred = np.maximum(pred, 0.0)
                values = score(pred - actual[:, fi], parent.valid[j], pct[j], area, neighbors)
                rows.append(
                    meta[["member", "year", "g"]].assign(
                        month=month, field=FIELDS[fi], strength=strength, rank=rank, **values
                    )
                )
    with atomic(dest) as tmp:
        pd.concat(rows, ignore_index=True).to_parquet(tmp, index=False, compression="zstd")
    record(
        "ocean_front_validation_month",
        month=month,
        training_cases=len(tr),
        validation_cases=len(meta),
    )


def summarize(out):
    frame = pd.concat(
        [pd.read_parquet(ROOT / f"{m:02}.parquet") for m in range(1, 13)], ignore_index=True
    )
    low, high = GROUP_LIMITS
    frame["warming"] = np.where(frame.g < low, "cold", np.where(frame.g > high, "warm", "inside"))
    names = ["basin_rmse", "front_rmse", "gradient_rmse_per_100km"]
    group = ["field", "strength", "warming"]
    extended = pd.concat([frame, frame.assign(warming="all")], ignore_index=True)
    summary = extended.groupby(group)[names].mean()
    summary["cases"] = extended.groupby(group).size()
    with atomic(out / "ocean_front_validation.csv") as tmp:
        summary.reset_index().to_csv(tmp, index=False)
    with atomic(out / "ocean_front_validation_cases.parquet") as tmp:
        frame.to_parquet(tmp, index=False, compression="zstd")
    intervals = []
    index = ["member", "year", "g"]
    for field, values in frame.groupby("field"):
        panels = values.groupby(index + ["strength"])[names].mean().unstack("strength")
        cases = panels.index.to_frame(index=False)
        for band in ("all", "cold", "inside", "warm"):
            weights, nmember, nyear = panel_weights(
                cases.member.to_numpy(),
                cases.year.to_numpy(),
                cases.g.to_numpy(),
                band,
                2000,
                121501,
            )
            for metric in names:
                old = weights @ panels[(metric, 0.0)].to_numpy()
                for strength in STRENGTHS[1:]:
                    new = weights @ panels[(metric, strength)].to_numpy()
                    record_estimate(
                        intervals,
                        band,
                        metric,
                        (field, str(strength)),
                        old,
                        new,
                        nmember,
                        nyear,
                        "physical",
                    )
    with atomic(out / "ocean_front_validation_intervals.csv") as tmp:
        pd.DataFrame(intervals).to_csv(tmp, index=False)
    record(
        "ocean_front_validation_complete",
        member_months=504,
        compared_fields=3,
        weight_strengths=list(STRENGTHS),
        rows=len(frame),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--month", type=int, choices=range(1, 13))
    mode.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.month is None:
        summarize(args.summary)
    else:
        run(args.month)
