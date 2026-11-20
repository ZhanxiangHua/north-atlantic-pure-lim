"""Compare positive covariance responses at fixed EOF coordinates and fixed steering regression."""
import argparse

import numpy as np
import pandas as pd
from ocean import GROUP_LIMITS
from records import WORK, record

from annual_lim.layout import DEFAULT_WORK
from annual_lim.model import MonthlyModel
from annual_lim.statistics import metrics
from tc_emulator.common import atomic
from tc_emulator.warming_circulation import transfer_matrix

FIELDS = ("sst", "ocean", "psl", "profiles")


def partition(covariance, kernel, cuts):
    # Summing all 16 cells gives trace(G Kᵀ C K), including both halves of every cross term.
    return np.add.reduceat(np.add.reduceat(covariance * kernel, cuts, axis=0), cuts, axis=1)


def empirical(cohort, month, role, kernel, cuts):
    source = WORK / "features" / cohort / f"{month:02}/scores.npz"
    with np.load(source) as bank:
        scores = np.column_stack([bank[role + "_published_" + field] for field in FIELDS])
        years = bank[role + "_year"]
        g = bank[role + "_g"]
    unique, first = np.unique(years, return_index=True)
    matrices = [partition(np.cov(scores[years == y], rowvar=False), kernel, cuts) for y in unique]
    return unique, g[first], np.stack(matrices)


def predictions(train_g, train_matrix, held_g, extension):
    pooled = train_matrix.mean(axis=0)
    p = len(pooled)
    centered = train_g - train_g.mean()
    bounded = np.clip(held_g, train_g.min() - extension, train_g.max() + extension) - train_g.mean()

    def response(annual):
        logged = np.log(annual)
        slope = centered @ (logged - logged.mean(axis=0)) / (centered @ centered)
        divisor = np.exp(centered[:, None] * np.atleast_1d(slope)).mean(axis=0)
        return np.exp(bounded[:, None] * np.atleast_1d(slope)) / divisor

    common = response(train_matrix.sum(axis=(1, 2)))[:, 0]
    component = np.sqrt(response(np.diagonal(train_matrix, axis1=1, axis2=2)))
    blocks = np.array_split(np.argsort(train_g), 3)
    anchors = np.array([train_g[rows].mean() for rows in blocks])
    values = np.stack([train_matrix[rows].mean(axis=0).ravel() for rows in blocks])
    interpolated = np.column_stack([np.interp(held_g, anchors, x) for x in values.T])
    interpolated = interpolated.reshape(-1, p, p)
    return dict(
        pooled=np.broadcast_to(pooled, (len(held_g), p, p)),
        common_loglinear=pooled * common[:, None, None],
        field_loglinear=component[:, :, None] * pooled * component[:, None, :],
        three_anchors=interpolated,
        half_pooled_anchors=0.5 * (pooled + interpolated),
    )


def run(month):
    dest = WORK / "covariance_laws" / f"{month:02}.csv"
    if dest.is_file():
        return
    model = MonthlyModel(DEFAULT_WORK / "weights", month)
    k = transfer_matrix(model.mean)
    kernel = k @ metrics(model.rep)[2][0] @ k.T
    cuts = np.cumsum([0] + [model.rep.blocks[f].basis.shape[1] for f in FIELDS[:-1]])
    years, g, truth = empirical("original", month, "val", kernel, cuts)
    climate = []
    for temp in g:
        factor = model.factor(temp)
        climate.append(partition(factor @ factor.T, kernel, cuts))
    climate = np.stack(climate)
    # Validate the field-space trace against actual gridded monthly steering on the same MPI backgrounds.
    grid = pd.read_csv(WORK / "attribution/published_v1/val" / f"{month:02}_years.csv")
    grid = grid[grid.point.eq("member_years")].set_index("year").loc[years]
    np.testing.assert_allclose(truth.sum(axis=(1, 2)), grid.conditional_mean_on_mpi, rtol=3e-6)
    np.testing.assert_allclose(climate.sum(axis=(1, 2)), grid.predicted_background, rtol=3e-7)
    rows = []
    for cohort in ("original", "expanded", "extended"):
        _, train_g, training = empirical(cohort, month, "train", kernel, cuts)
        laws = predictions(train_g, training, g, model.cal.extension)
        laws["published"] = climate
        for label, matrices in laws.items():
            if np.linalg.eigvalsh(matrices).min() < -1e-8:
                raise ValueError("Invalid steering contribution covariance")
            for year, temp, target, value in zip(years, g, truth, matrices, strict=True):
                total = target.sum()
                actual_ocean, model_ocean = target[:2, :2].sum(), value[:2, :2].sum()
                cross, model_cross = 2 * target[:2, 2:].sum(), 2 * value[:2, 2:].sum()
                rows.append(
                    dict(
                        cohort=cohort,
                        training_years=len(train_g),
                        month=month,
                        year=year,
                        g=temp,
                        warming="cold"
                        if temp < GROUP_LIMITS[0]
                        else "warm"
                        if temp > GROUP_LIMITS[1]
                        else "inside",
                        law=label,
                        truth_total=total,
                        predicted_total=value.sum(),
                        total_abs_log=abs(np.log(value.sum() / total)),
                        truth_ocean=actual_ocean,
                        predicted_ocean=model_ocean,
                        ocean_abs_log=abs(np.log(model_ocean / actual_ocean)),
                        truth_cross=cross,
                        predicted_cross=model_cross,
                        cross_error=abs(model_cross - cross) / total,
                        matrix_error=np.linalg.norm(value - target) / np.trace(target),
                    )
                )
    with atomic(dest) as tmp:
        pd.DataFrame(rows).to_csv(tmp, index=False)
    record(
        "fixed_transfer_covariance_validation",
        month=month,
        laws=len(laws),
        training_cohorts=3,
        validation_members=7,
        validation_years=len(years),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--month", type=int, required=True)
    run(p.parse_args().month)
