"""Compare positive transforms, local spatial regularization, parcel gradients and conditional marginals."""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
import scipy.linalg as la
from joint import PROBS, QUANTILES
from ocean import ocean_data, physical_scores, solve_basis, truncate
from records import WORK, record
from scipy.special import ndtri
from scipy.stats import qmc

from annual_lim.layout import DEFAULT_WORK
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic, save_json
from tc_emulator.representation import EOF


def encode_mld(x, p):
    h = np.maximum(x, 0.01)
    return np.log(h) if p == 0 else (h**p - 1) / p


def undo(name, x, p, sqrt_grad):
    if name == "mld":
        return np.maximum(0.01, np.exp(x) if p == 0 else np.maximum(0.0, 1 + p * x) ** (1 / p))
    return np.maximum(x, 0.0) ** (2 if sqrt_grad else 1)


def localized(base, x, lat, lon, rank, fraction):
    """Preserve basin-wide modes and learn regional corrections in a tapered residual subspace."""
    valid = base.valid
    z = x.reshape(len(x), -1)[:, valid.ravel()].astype(float)
    z = np.where(np.isfinite(z), z - base.mean[valid], 0.0) * base.weights[valid]
    ng = max(2, round(rank * fraction))
    b = base.basis[:, :ng]
    res = z - (z @ b) @ b.T
    y, xx = np.meshgrid(lat, lon, indexing="ij")
    bases = []
    for cy in (12.0, 33.0, 54.0):
        for cx in (278.0, 310.0, 342.0):
            taper = np.exp(-0.5 * ((y - cy) / 16) ** 2 - 0.5 * ((xx - cx) / 24) ** 2)
            t = np.broadcast_to(taper, valid.shape)[valid]
            block = res * t
            ne = min(24, rank - ng)
            v, e = la.eigh(
                block @ block.T, subset_by_index=[len(z) - ne, len(z) - 1], check_finite=False
            )
            local = (block.T @ e[:, ::-1]) / np.sqrt(v[::-1])
            bases.append(local * t[:, None])
    seeds = np.column_stack(bases)
    seeds -= b @ (b.T @ seeds)
    q = la.qr(seeds, mode="economic", check_finite=False)[0]
    projected = res @ q
    _, axis = la.eigh(
        projected.T @ projected,
        subset_by_index=[q.shape[1] - (rank - ng), q.shape[1] - 1],
        check_finite=False,
    )
    u = np.column_stack((b, q @ axis[:, ::-1]))
    return EOF(
        base.mean, base.weights, base.valid, u, np.var(z @ u, axis=0, ddof=1), base.total_variance
    )


def gaussian_scores(train_z, truth_z, g, year, valid_g):
    # A common quadratic/linear blend and pooled within-year covariance isolate the spatial/transform choice.
    center, scale = g.mean(), g.std()
    h = (g - center) / scale
    H = np.column_stack((np.ones(len(g)), h, h * h))
    b2 = np.linalg.lstsq(H, train_z, rcond=None)[0]
    b1 = np.vstack((np.linalg.lstsq(H[:, :2], train_z, rcond=None)[0], np.zeros(train_z.shape[1])))
    b = 0.5 * (b1 + b2)
    cov = np.mean([np.cov(train_z[year == yr], rowvar=False) for yr in np.unique(year)], axis=0)
    cov = 0.95 * cov + 0.05 * np.diag(cov.diagonal())
    F = np.linalg.cholesky(cov)
    t = (valid_g - center) / scale
    return np.column_stack((np.ones(len(t)), t, t * t)) @ b, F


def evaluate(b, tr, valid_raw, physical, tm, vm, p, sq, region, clim, tail, base_masks, ys, xs):
    z = b.transform(tr)
    v = b.transform(valid_raw)
    avg, F = gaussian_scores(z, v, tm.g.to_numpy(), tm.year.to_numpy(), vm.g.to_numpy())
    decoded = b.inverse(v)
    res = []
    pairs = []
    # Shared Sobol integration in two normal coordinates evaluates spatial contrast without drawing full random maps.
    draws = ndtri(np.clip(qmc.Sobol(2, scramble=True, seed=462).random_base2(6), 1e-9, 1 - 1e-9))
    for c, name in enumerate(("mld", "gamma")):
        mask = b.valid[c]
        first = 0 if c == 0 else b.valid[0].sum()
        J = b.basis[first : first + mask.sum()] / b.weights[c][mask, None]
        loc = avg @ J.T + b.mean[c][mask]
        loading = J @ F
        sd = np.linalg.norm(loading, axis=1)
        mu = loc[:, None, :]
        sigma = sd[None, None, :]
        qs = undo(name, mu + QUANTILES[None, :, None] * sigma, p, sq)
        actual = physical[:, c + 1]
        recon = undo(name, decoded[:, c], p, sq)
        ob = actual[:, mask]
        diff = np.where(np.isfinite(ob), ob, 0.0)[:, None, :] - qs
        crps = np.mean(2 * (PROBS[None, :, None] - (diff < 0)) * diff, axis=1)
        flags = np.isfinite(ob)
        for label, a in region.items():
            frame = physical_scores(
                actual, recon, clim[c + 1], tail[c + 1], a, base_masks[c], vm, name
            )
            w = a[mask] * flags
            frame["grid_crps"] = (w * crps).sum(axis=1) / w.sum(axis=1)
            frame["region"] = label
            res.append(frame)
        # Neighbor pairs at native-grid separations. Exclude boundaries and any unrepresented wet cell.
        index = np.full(mask.shape, -1)
        index[mask] = np.arange(mask.sum())
        for dy, dx in ((0, 1), (0, 4), (3, 0)):
            hi = mask.shape[0] - dy
            wi = mask.shape[1] - dx
            i = index[:hi, :wi].ravel()
            j = index[dy : dy + hi, dx : dx + wi].ravel()
            aa = (
                region["basin"][:hi, :wi].ravel()
                * region["basin"][dy : dy + hi, dx : dx + wi].ravel()
            )
            keep = (i >= 0) & (j >= 0) & (aa > 0)
            i, j = i[keep], j[keep]
            aa = np.sqrt(aa[keep])
            corr = np.einsum("ij,ij->i", loading[i], loading[j]) / np.maximum(sd[i] * sd[j], 1e-15)
            corr = np.clip(corr, -1.0, 1.0)
            q1 = draws[:, 0, None]
            q2 = q1 * corr + draws[:, 1, None] * np.sqrt(np.maximum(1 - corr * corr, 0.0))
            # A year shares the same forced mean and predictive covariance across its members.
            for y in vm.year.unique():
                ii = np.flatnonzero(vm.year.to_numpy() == y)
                ii0 = ii[0]
                m1 = undo(name, loc[ii0, i] + q1 * sd[i], p, sq)
                m2 = undo(name, loc[ii0, j] + q2 * sd[j], p, sq)
                expect = np.abs(m1 - m2).mean(axis=0)
                obs_pair = np.abs(ob[ii][:, i] - ob[ii][:, j])
                ok = np.isfinite(obs_pair)
                w = ok * aa
                vs = (np.where(ok, obs_pair - expect, 0.0) ** 2 * w).sum(axis=1) / w.sum(axis=1)
                actual_grad = ob[ii][:, i] - ob[ii][:, j]
                pred_grad = recon[ii][:, mask][:, i] - recon[ii][:, mask][:, j]
                ge = (np.where(ok, pred_grad - actual_grad, 0.0) ** 2 * w).sum(axis=1) / w.sum(
                    axis=1
                )
                pairs.append(
                    vm.iloc[ii][["member", "year", "g"]].assign(
                        field=name, dy=dy, dx=dx, variogram_score=vs, gradient_mse=ge
                    )
                )
    return pd.concat(res, ignore_index=True), pd.concat(pairs, ignore_index=True)


def run(cohort, month):
    target = WORK / "ocean_alternatives" / cohort / f"{month:02}"
    if (target / "complete.json").is_file():
        return
    start = time.monotonic()
    record("ocean_alternatives_start", month=month, cohort=cohort)
    train, tm = ocean_data(cohort, month, "train")
    truth, vm = ocean_data("original", month, "val")
    rep = MonthlyModel(DEFAULT_WORK / "weights", month).rep
    y, x = np.meshgrid(rep.lat, rep.lon, indexing="ij")
    area = np.cos(np.deg2rad(y))
    region = {
        "basin": area * ((y >= 0) & (y <= 60) & (x >= 260) & (x <= 360)),
        "corridor": area * ((y >= 10) & (y <= 30) & (x >= 280) & (x <= 340)),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clim = np.nanmean(train, axis=0)
        tail = np.nanquantile(train, 0.90, axis=0)
    results = []
    variograms = []
    for p, sq in (
        (0.0, False),
        (0.25, False),
        (0.5, False),
        (1.0, False),
        (0.0, True),
        (0.25, True),
        (0.5, True),
    ):
        a = train[:, 1:].copy()
        b = truth[:, 1:].copy()
        for v in (a, b):
            v[:, 0] = encode_mld(v[:, 0], p)
            if sq:
                v[:, 1] = np.sqrt(np.maximum(0.0, v[:, 1]))
        for depth in (0.0, 0.7) if p == 0 else (0.0,):
            base = solve_basis(a, area, [0.5, 0.5], 96, depth_power=depth)
            for rank in (18, 48, 72, 96):
                for loc in (0.0, 0.5) if rank in (18, 72) else (0.0,):
                    block = (
                        localized(base, a, rep.lat, rep.lon, rank, loc)
                        if loc
                        else truncate(base, slice(rank))
                    )
                    res, vari = evaluate(
                        block,
                        a,
                        b,
                        truth,
                        tm,
                        vm,
                        p,
                        sq,
                        region,
                        clim,
                        tail,
                        rep.blocks["ocean"].valid,
                        rep.lat,
                        rep.lon,
                    )
                    key = dict(
                        mld_power=p,
                        gamma_sqrt=sq,
                        depth_weight=depth,
                        rank=rank,
                        local_fraction=loc,
                    )
                    results.append(res.assign(**key))
                    variograms.append(vari.assign(**key))
    for k, rows in (("scores", results), ("spatial", variograms)):
        with atomic(target / (k + ".parquet")) as tmp:
            pd.concat(rows, ignore_index=True).to_parquet(tmp, index=False, compression="zstd")
    save_json(
        target / "complete.json",
        dict(
            cohort=cohort,
            month=month,
            candidates=len(results),
            training=len(tm),
            validation=len(vm),
        ),
    )
    record(
        "ocean_alternatives_done",
        month=month,
        cohort=cohort,
        candidates=len(results),
        seconds=round(time.monotonic() - start, 1),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--cohort", default="original")
    a = p.parse_args()
    run(a.cohort, a.month)
