"""Test Gaussian variability lost when a training ocean map is truncated to its EOF scores."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joint import PhysicalValidation
from ocean import FIELDS, joint_input, ocean_data
from records import WORK, record
from scipy.linalg import eigh

from annual_lim.layout import DEFAULT_WORK
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic


def residual_fit(block, maps, metadata, ranks=(16, 64)):
    """Fit within-year residual covariance in the existing weighted map coordinates."""
    values = maps[:, block.valid].astype(float)
    weighted = (
        np.where(np.isfinite(values), values - block.mean[block.valid], 0)
        * block.weights[block.valid]
    )
    residual = weighted - (weighted @ block.basis) @ block.basis.T
    groups = metadata.groupby("year", sort=True).indices
    variance, warming = [], []
    for indices in groups.values():
        row = residual[indices]
        row -= row.mean(axis=0)
        residual[indices] = row
        variance.append(np.einsum("ij,ij->", row, row) / (len(row) - 1))
        warming.append(metadata.g.iloc[indices[0]])
    ndf = len(residual) - len(groups)
    sample_gram = residual @ residual.T / ndf
    # Multiplying Rᵀ by the leading sample-space eigenvectors yields spatial covariance factors.
    _, vectors = eigh(
        sample_gram,
        subset_by_index=[len(residual) - max(ranks), len(residual) - 1],
        check_finite=False,
    )
    modes = (residual.T @ vectors[:, ::-1]) / np.sqrt(ndf)
    np.testing.assert_allclose(block.basis.T @ modes, 0, atol=1e-7)
    diag = {"full": np.einsum("ij,ij->j", residual, residual) / ndf}
    for rank in ranks:
        diag[f"rank{rank}"] = (modes[:, :rank] ** 2).sum(axis=1)
    if np.any(diag[f"rank{max(ranks)}"] > diag["full"] + 1e-10):
        raise ValueError("Residual EOF variance exceeds the untruncated covariance")
    # A single scale per field block preserves the fitted residual correlations at each warming.
    warming = np.asarray(warming)
    delta = warming - warming.mean()
    log_variance = np.log(np.asarray(variance))
    slope = (delta @ (log_variance - log_variance.mean())) / (delta @ delta)
    divisor = np.mean(np.exp(slope * delta))
    response = dict(
        slope=float(slope),
        g0=float(warming.mean()),
        low=float(warming.min()),
        high=float(warming.max()),
        divisor=float(divisor),
    )
    full = diag["full"].sum()
    fractions = {name: float(v.sum() / full) for name, v in diag.items()}
    # Convert the diagonal of the weighted covariance back to model coordinates at each wet cell.
    grids = {}
    for arm, v in diag.items():
        grid = np.zeros(block.valid.shape)
        grid[block.valid] = v / block.weights[block.valid] ** 2
        grids[arm] = grid
    return grids, fractions, response


def run(tag, weights, cohort, month):
    out = WORK / "ocean_residual" / tag / f"{month:02}.parquet"
    if out.is_file():
        return
    model = MonthlyModel(weights, month)
    reference = MonthlyModel(DEFAULT_WORK / "weights", month)
    train, training = ocean_data(cohort, month, "train")
    truth, validation = ocean_data("original", month, "val")
    if set(training.member) & set(validation.member):
        raise ValueError("Training and validation share members")
    scorer = PhysicalValidation(
        SimpleNamespace(geo=reference.rep, month=month),
        truth,
        validation,
        np.empty((len(validation), 0)),
    )
    components = {}
    for name, maps, fields in (
        ("sst", train[:, :1], ("sst",)),
        ("ocean", joint_input(train), ("mld", "gamma")),
    ):
        grids, fractions, response = residual_fit(model.rep.blocks[name], maps, training)
        for i, field in enumerate(fields):
            components[field] = (
                {arm: a[i].ravel() for arm, a in grids.items()},
                fractions,
                response,
            )
    frames = []
    for arm in ("none", "rank16", "rank64", "full", "rank64_temperature"):
        key = "rank64" if arm == "rank64_temperature" else arm

        def extra(field, g):
            arrays, _, fit = components[field]
            scale = 1.0
            if arm == "rank64_temperature":
                centered = np.clip(g, fit["low"], fit["high"]) - fit["g0"]
                scale = np.exp(fit["slope"] * centered) / fit["divisor"]
            return scale * arrays[key]

        scores = scorer.ocean_scores(model, None if arm == "none" else extra)
        scores["extra_fraction"] = scores.field.map(
            {f: (0.0 if arm == "none" else components[f][1][key]) for f in FIELDS}
        )
        scores["training_log_variance_slope"] = scores.field.map(
            {f: components[f][2]["slope"] for f in FIELDS}
        )
        frames.append(scores.assign(arm=arm, tag=tag, cohort=cohort, month=month))
    data = pd.concat(frames, ignore_index=True)
    with atomic(out) as tmp:
        data.to_parquet(tmp, index=False, compression="zstd")
    record(
        "ocean_residual_validation",
        tag=tag,
        cohort=cohort,
        month=month,
        training_cases=len(train),
        validation_cases=len(truth),
        arms=len(frames),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--cohort", choices=["original", "expanded", "extended"], required=True)
    p.add_argument("--month", type=int, required=True)
    args = p.parse_args()
    run(args.tag, args.weights, args.cohort, args.month)
