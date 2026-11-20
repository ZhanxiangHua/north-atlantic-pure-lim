"""Extend the original ocean coordinates while preserving their climate-dependent marginal."""
import argparse
import time
from dataclasses import replace

import numpy as np
from features import MAX_OCEAN, save_block
from ocean import joint_input, ocean_data
from records import WORK, record
from scipy import linalg

from annual_lim.layout import DEFAULT_WORK
from annual_lim.model import MonthlyModel
from tc_emulator.background import Background, Circulation, polynomial
from tc_emulator.common import atomic, save_json


def extend_basis(parent, fields, depth_power=0.0):
    """Append EOFs of residual maps, retaining the published encoder as the first columns."""
    columns = parent.valid
    values = fields.reshape(len(fields), -1)[:, columns.ravel()].astype(float)
    weighted = (
        np.where(np.isfinite(values), values - parent.mean[columns], 0.0) * parent.weights[columns]
    )
    u = parent.basis
    residual = weighted - (weighted @ u) @ u.T
    emphasis = np.ones(columns.shape)
    if depth_power:
        h = np.exp(np.clip(parent.mean[0], np.log(1.0), np.log(1000.0))) ** depth_power
        area_weights = parent.weights[0, columns[0]] ** 2
        emphasis[0] = h / np.sqrt(np.average(h[columns[0]] ** 2, weights=area_weights))
    factors = emphasis[columns]
    target = residual * factors
    count = MAX_OCEAN - u.shape[1]
    eig, v = linalg.eigh(
        target @ target.T,
        subset_by_index=[len(target) - count, len(target) - 1],
        check_finite=False,
    )
    maps = ((target.T @ v[:, ::-1]) / np.sqrt(eig[::-1])) / factors[:, None]
    maps -= u @ (u.T @ maps)
    tail = linalg.qr(maps, mode="economic", check_finite=False)[0]
    basis = np.column_stack((u, tail))
    np.testing.assert_allclose(basis.T @ basis, np.eye(MAX_OCEAN), atol=2.0e-9)
    return replace(
        parent,
        basis=basis,
        variance=np.var(weighted @ basis, axis=0, ddof=1),
        total_variance=float(np.sum(weighted**2) / (len(fields) - 1)),
    )


def prepare(cohort, month):
    folder = WORK / "features" / cohort / f"{month:02}"
    marker = folder / "nested.json"
    if marker.is_file():
        return
    tick = time.monotonic()
    record("nested_basis_start", cohort=cohort, month=month)
    parent = MonthlyModel(DEFAULT_WORK / "weights", month)
    fields = {
        r: ocean_data(cohort if r == "train" else "original", month, r)[0] for r in ("train", "val")
    }
    scores = {}
    for name, source, depth in (
        ("nested_sst", "sst", 0.0),
        ("nested_ocean", "ocean", 0.0),
        ("nested_ocean_depth", "ocean", 0.7),
    ):
        x = {k: a[:, :1] if source == "sst" else joint_input(a) for k, a in fields.items()}
        block = extend_basis(parent.rep.blocks[source], x["train"], depth)
        save_block(folder / (name + ".npz"), block)
        for role, data in x.items():
            values = block.transform(data)
            np.testing.assert_allclose(
                values[:, :18], parent.rep.blocks[source].transform(data), atol=1.0e-11
            )
            scores[role + "_" + name] = values
    with atomic(folder / "scores_nested.npz") as tmp:
        np.savez(tmp, **scores)
    save_json(marker, dict(cohort=cohort, month=month, rank=MAX_OCEAN, coarse_rank=18))
    record(
        "nested_basis_done", cohort=cohort, month=month, seconds=round(time.monotonic() - tick, 2)
    )


def attach(parent, z, w, g, years, sst, ocean, tail_gain, ridge, curve):
    """Stack t = m_t(g) + B[c-m_c(g)] + e. High-resolution wind uses the independent residual e."""
    coarse = np.r_[np.arange(18), sst + np.arange(18), sst + ocean + np.arange(42)]
    tail = np.setdiff1d(np.arange(z.shape[1]), coarse)
    design = polynomial(parent.bg.standard_g(g))
    full = np.linalg.lstsq(design, z[:, tail], rcond=None)[0]
    linear = np.linalg.lstsq(design[:, :2], z[:, tail], rcond=None)[0]
    beta = np.empty((3, z.shape[1]))
    beta[:, coarse] = parent.bg.beta
    beta[:, tail] = curve * full + (1 - curve) * np.vstack((linear, np.zeros(len(tail))))
    d = z - design @ beta
    # Only between-member anomalies within the same year contribute to stochastic covariance.
    noise = d.copy()
    for year in np.unique(years):
        group = years == year
        noise[group] -= noise[group].mean(axis=0)
    c, t = noise[:, coarse], noise[:, tail]
    c_std = np.maximum(c.std(axis=0), 1.0e-12)
    x = c / c_std
    loading = np.linalg.solve(x.T @ x + 0.06 * len(x) * np.eye(x.shape[1]), x.T @ t).T / c_std
    eps = t - c @ loading.T
    error_cov = eps.T @ eps / (len(g) - len(np.unique(years)))
    error_cov = 0.95 * error_cov + 0.05 * np.diag(np.diag(error_cov))
    embed = np.zeros((z.shape[1], len(coarse)))
    embed[coarse] = np.eye(len(coarse))
    embed[tail] = loading
    cov = embed @ parent.bg.pooled @ embed.T
    cov[np.ix_(tail, tail)] += error_cov
    root = np.linalg.cholesky(cov)
    axes = linalg.solve_triangular(root, embed @ parent.bg.lower0 @ parent.directions, lower=True)
    np.testing.assert_allclose(axes.T @ axes, np.eye(axes.shape[1]), atol=2.0e-8)
    bg = Background(
        parent.bg.center, parent.bg.scale, beta, beta[:2], root, np.zeros_like(root), cov
    )
    raw = np.zeros((z.shape[1], w.shape[1]))
    raw[coarse] = parent.mean.coefficients[2:] / parent.mean.scale[:, None]
    if tail_gain:
        innovation = d[:, tail] - d[:, coarse] @ loading.T
        scale = np.maximum(innovation.std(axis=0), 1.0e-10)
        X = innovation / scale
        dw = w - parent.mean.predict(parent.bg, g, z[:, coarse])
        corr = (
            np.linalg.solve(X.T @ X + ridge * len(g) * np.eye(len(tail)), X.T @ dw) / scale[:, None]
        )
        raw[tail] = tail_gain * corr
        raw[coarse] -= tail_gain * loading.T @ corr
    scale = np.maximum(np.std(d, axis=0), 1.0e-10)
    regression = Circulation(np.vstack((parent.mean.coefficients[:2], raw * scale[:, None])), scale)
    return bg, regression, axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="original")
    parser.add_argument("--month", type=int, required=True)
    a = parser.parse_args()
    prepare(a.cohort, a.month)
