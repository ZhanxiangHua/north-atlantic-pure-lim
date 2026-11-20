"""Borrow neighboring months' ocean patterns while retaining each month's own climate law."""
import argparse
from dataclasses import replace

import numpy as np
from features import MAX_OCEAN, save_block
from ocean import joint_input, ocean_data
from records import WORK, record
from scipy import linalg

from annual_lim.layout import DEFAULT_WORK
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic, save_json


def ocean_basis(parent, fields, depth_power, fraction=0.30, rank=MAX_OCEAN):
    """Fit a common set of tail maps in a fixed metric. fields[0] is the target month."""
    keep = parent.valid
    u = parent.basis
    factors = np.ones(keep.shape)
    if depth_power:
        depth_weight = np.exp(np.clip(parent.mean[0], np.log(1.0), np.log(1000.0))) ** depth_power
        factors[0] = depth_weight / np.sqrt(
            np.average(depth_weight[keep[0]] ** 2, weights=parent.weights[0, keep[0]] ** 2)
        )
    factors = factors[keep]
    covariance = np.zeros((int(keep.sum()), int(keep.sum())))
    target_energy = None
    for i, values in enumerate(fields):
        x = values.reshape(len(values), -1)[:, keep.ravel()].astype(float)
        x = np.where(np.isfinite(x), x - parent.mean[keep], 0.0) * parent.weights[keep]
        if i == 0:
            target = x
        else:
            # Adjacent calendar climatologies must not become modes of the target month.
            x -= x.mean(axis=0)
        residual = (x - (x @ u) @ u.T) * factors
        cov = (residual.T @ residual) / (len(x) - 1)
        energy = np.trace(cov)
        if energy <= 0:
            raise ValueError("Ocean training anomalies have no spatial variation")
        if i == 0:
            target_energy = energy
        weight = 1 - fraction if i == 0 else fraction / 2
        covariance += weight * (target_energy / energy) * cov
    count = rank - u.shape[1]
    eig, vec = linalg.eigh(
        covariance,
        subset_by_index=[len(covariance) - count, len(covariance) - 1],
        check_finite=False,
    )
    if eig[0] <= 0:
        raise ValueError("Insufficient rank in the calendar-pooled residuals")
    tail = vec[:, ::-1] / factors[:, None]
    tail -= u @ (u.T @ tail)
    basis = np.column_stack((u, linalg.qr(tail, mode="economic", check_finite=False)[0]))
    np.testing.assert_allclose(basis.T @ basis, np.eye(rank), atol=2e-9)
    return replace(
        parent,
        basis=basis,
        variance=np.var(target @ basis, axis=0, ddof=1),
        total_variance=float(np.sum(target**2) / (len(target) - 1)),
    )


def prepare(cohort, month):
    root = WORK / "features" / cohort / f"{month:02}"
    marker = root / "calendar_ocean.json"
    if marker.is_file():
        return
    record("calendar_pooled_ocean_start", cohort=cohort, month=month)
    months = [month, (month - 2) % 12 + 1, month % 12 + 1]
    sample = []
    ids = None
    for m in months:
        maps, meta = ocean_data(cohort, m, "train")
        cases = meta[["member", "year", "g"]].to_numpy()
        if ids is not None:
            np.testing.assert_array_equal(ids, cases)
        ids = cases
        sample.append(maps)
    validation, _ = ocean_data("original", month, "val")
    original = MonthlyModel(DEFAULT_WORK / "weights", month)
    scores = {}
    for key, source, depth in (
        ("calendar_sst", "sst", 0.0),
        ("calendar_ocean_depth", "ocean", 0.7),
    ):

        def channels(maps):
            return maps[:, :1] if source == "sst" else joint_input(maps)

        block = ocean_basis(original.rep.blocks[source], [channels(x) for x in sample], depth)
        save_block(root / (key + ".npz"), block)
        for role, maps in (("train", sample[0]), ("val", validation)):
            x = channels(maps)
            coords = block.transform(x)
            np.testing.assert_allclose(
                coords[:, :18], original.rep.blocks[source].transform(x), atol=1e-11
            )
            scores[role + "_" + key] = coords
    with atomic(root / "scores_calendar.npz") as tmp:
        np.savez(tmp, **scores)
    save_json(
        marker,
        dict(
            cohort=cohort,
            month=month,
            training_months=months,
            target_weight=0.7,
            neighbor_weight_each=0.15,
            member_years=len(ids),
        ),
    )
    record(
        "calendar_pooled_ocean_done",
        cohort=cohort,
        month=month,
        cases=len(ids),
        calendar_months=months,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--month", type=int, required=True)
    a = p.parse_args()
    prepare(a.cohort, a.month)
