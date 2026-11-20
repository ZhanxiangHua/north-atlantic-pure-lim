"""Score ocean spatial dependence and cross-variable correlations on independent MPI member maps."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joint import physical_moments
from ocean import FIELDS, ocean_data, panel_cohort
from records import WORK, record
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from annual_lim.layout import DEFAULT_WORK, read
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic


def field_loc(model, mu, F, key):
    block, ix = ("sst", 0) if key == "sst" else ("ocean", int(key == "gamma"))
    b = model.rep.blocks[block]
    mask = b.valid[ix].ravel()
    offset = int(b.valid[:ix].sum())
    first = 0 if block == "sst" else model.rep.blocks["sst"].basis.shape[1]
    last = first + b.basis.shape[1]
    map_basis = b.basis[offset : offset + mask.sum()] / b.weights[ix].ravel()[mask, None]
    mean = np.full(mask.size, np.nan)
    load = np.zeros((mask.size, F.shape[1]))
    mean[mask] = b.mean[ix].ravel()[mask] + map_basis @ mu[first:last]
    load[mask] = map_basis @ F[first:last]
    return mean, load, mask


def pairs(mask, step, axis):
    sel = np.roll(mask, -step, axis=axis) & mask
    if axis == 0:
        sel[-step:, :] = False
    else:
        sel[:, -step:] = False
    ids = np.arange(mask.size).reshape(mask.shape)
    return ids[sel], np.roll(ids, -step, axis=axis)[sel]


def paired_expected(field, m, v, n, w, c, draws):
    """First absolute moment of a spatial increment in physical units."""
    sd = np.sqrt(np.maximum(1.0e-14, v + w - 2 * c))
    if field == "sst":
        z = (m - n) / sd
        return 2 * sd * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) + (m - n) * (2 * ndtr(z) - 1)
    if field == "mld":
        # E[e^X sign(X-Y)] can be taken under a tilted normal with mean shifted by covariance.
        left = np.exp(m + 0.5 * v) * (2 * ndtr((m - n + v - c) / sd) - 1)
        right = np.exp(n + 0.5 * w) * (2 * ndtr((m - n + c - w) / sd) - 1)
        # The public decoder floors MLD at 0.01 m. If active at practical precision, use the bounded draws.
        if np.all(ndtr((np.log(0.01) - m) / np.maximum(np.sqrt(v), 1.0e-12)) < 1.0e-8) and np.all(
            ndtr((np.log(0.01) - n) / np.maximum(np.sqrt(w), 1.0e-12)) < 1.0e-8
        ):
            return np.maximum(0.0, left - right)
    sx = np.maximum(np.sqrt(v), 1.0e-10)
    common = c / sx
    x = m + draws[:, 0, None] * sx
    y = (
        n
        + draws[:, 0, None] * common
        + draws[:, 1, None] * np.sqrt(np.maximum(w - common**2, 0.0))
    )
    if field == "mld":
        x, y = np.maximum(0.01, np.exp(x)), np.maximum(0.01, np.exp(y))
    else:
        x, y = np.maximum(0.0, x), np.maximum(0.0, y)
    return np.abs(x - y).mean(axis=0)


def run(tag, weights, month, role):
    dest = WORK / "spatial_check" / tag / role / f"{month:02}.parquet"
    if dest.is_file():
        return
    if role != "val":
        receipt = read(WORK / "FROZEN.json")
        if tag not in ("published_v1", receipt["stage"]):
            raise ValueError("Only the frozen verification fit can score a test panel")
    model = MonthlyModel(weights, month)
    if (
        role != "val"
        and tag != "published_v1"
        and model.meta["id"] != receipt["months"][f"{month:02}"]
    ):
        raise ValueError("Spatial test candidate differs from frozen receipt")
    cohort, split = panel_cohort(role)
    truth, meta = ocean_data(cohort, month, split)
    y, x = np.meshgrid(model.rep.lat, model.rep.lon, indexing="ij")
    orig = MonthlyModel(DEFAULT_WORK / "weights", month)
    common = [orig.rep.blocks["sst"].valid[0], *orig.rep.blocks["ocean"].valid]
    areas = {
        "basin": np.cos(np.deg2rad(y)) * ((y >= 0) & (y <= 60) & (x >= 260) & (x <= 360)),
        "corridor": np.cos(np.deg2rad(y)) * ((y >= 10) & (y <= 30) & (x >= 280) & (x <= 340)),
    }
    # The fixed, scrambled Sobol sample is shared by every model and member.
    q = ndtri(qmc.Sobol(2, scramble=True, seed=72118).random_base2(8))
    output = []
    for year, group in meta.groupby("year", sort=True):
        g = float(group.g.iloc[0])
        mu = model.bg.mean([g])[0]
        F = model.factor(g)
        lookup = {key: field_loc(model, mu, F, key) for key in FIELDS}
        selected = group.index.to_numpy()
        for fi, key in enumerate(FIELDS):
            m, load, keep = lookup[key]
            var = np.square(load).sum(axis=1)
            maps = truth[selected, fi].reshape(len(group), -1).astype(float)
            for region, area in areas.items():
                wet = common[fi] & keep.reshape(y.shape) & (area > 0)
                for step in (1, 3, 6):
                    numer = np.zeros(len(group))
                    denom = np.zeros(len(group))
                    for axis in (0, 1):
                        a, b = pairs(wet, step, axis)
                        weights_pair = np.sqrt(area.ravel()[a] * area.ravel()[b])
                        cov = (load[a] * load[b]).sum(axis=1)
                        predicted = paired_expected(key, m[a], var[a], m[b], var[b], cov, q)
                        observed = np.abs(maps[:, a] - maps[:, b])
                        valid = np.isfinite(observed)
                        numer += (
                            np.where(valid, (observed - predicted) ** 2, 0.0) * weights_pair
                        ).sum(axis=1)
                        denom += (valid * weights_pair).sum(axis=1)
                    for who, v in zip(group.member, numer / denom, strict=True):
                        output.append(
                            dict(
                                member=who,
                                year=year,
                                g=g,
                                region=region,
                                metric="variogram_score",
                                field=key,
                                distance_steps=step,
                                value=v,
                            )
                        )
        # Compare predicted and within-year sample correlations, with one degree of freedom per MPI member.
        # SST is linear, MLD is lognormal, and gamma is rectified normal in the public decoder.
        for ix, j in ((0, 1), (0, 2), (1, 2)):
            left, right = FIELDS[ix], FIELDS[j]
            m, L, ok1 = lookup[left]
            n, R, ok2 = lookup[right]
            lv = (L * L).sum(axis=1)
            rv = (R * R).sum(axis=1)
            c = (L * R).sum(axis=1)
            sd = np.sqrt(np.maximum(rv, 1.0e-15))
            if left == "sst" and right == "mld":
                cov = c * np.exp(n + rv / 2)
            elif left == "sst":
                cov = c * ndtr(n / sd)
            else:
                original = physical_moments("gamma", n, sd)[0]
                shifted = physical_moments("gamma", n + c, sd)[0]
                cov = np.exp(m + lv / 2) * (shifted - original)
            v1 = physical_moments(left, m, np.sqrt(lv))[1]
            v2 = physical_moments(right, n, sd)[1]
            corr = np.clip(cov / np.sqrt(np.maximum(v1 * v2, 1.0e-15)), -1.0, 1.0)
            a = truth[selected, ix].reshape(len(group), -1)
            b = truth[selected, j].reshape(len(group), -1)
            finite = np.isfinite(a) & np.isfinite(b)
            count = finite.sum(axis=0)
            da = np.where(
                finite, a - np.where(finite, a, 0.0).sum(axis=0) / np.maximum(count, 1), 0.0
            )
            db = np.where(
                finite, b - np.where(finite, b, 0.0).sum(axis=0) / np.maximum(count, 1), 0.0
            )
            denom = np.sqrt((da**2).sum(axis=0) * (db**2).sum(axis=0))
            emp = (da * db).sum(axis=0) / np.maximum(denom, 1.0e-15)
            for region, area in areas.items():
                valid = (
                    ok1
                    & ok2
                    & common[ix].ravel()
                    & common[j].ravel()
                    & (count >= 4)
                    & (denom > 1.0e-8)
                    & (area.ravel() > 0)
                )
                w = area.ravel()[valid]
                w /= w.sum()
                output.append(
                    dict(
                        member="ensemble",
                        year=year,
                        g=g,
                        region=region,
                        metric="cross_correlation_rmse",
                        field=left + "/" + right,
                        distance_steps=0,
                        value=np.sqrt(w @ ((corr[valid] - emp[valid]) ** 2)),
                    )
                )
    with atomic(dest) as tmp:
        pd.DataFrame(output).to_parquet(tmp, index=False, compression="zstd")
    record(
        "spatial_verification_done",
        tag=tag,
        role=role,
        month=month,
        cases=len(meta),
        rows=len(output),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, default=DEFAULT_WORK / "weights")
    p.add_argument("--role", choices=["val", "test", "fresh", "ssp245_year"], default="val")
    p.add_argument("--month", type=int, required=True)
    a = p.parse_args()
    run(a.tag, a.weights, a.month, a.role)
