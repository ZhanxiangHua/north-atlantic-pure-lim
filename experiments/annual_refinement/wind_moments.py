"""Sufficient physical-space targets for fitting and testing time scales without storing tracks."""
import argparse
import time

import numpy as np
from records import WORK, record

from annual_lim.layout import CONFIG, DEFAULT_WORK
from annual_lim.model import MonthlyModel
from annual_lim.statistics import component_cov, metrics, variance, window_moments
from annual_lim.train import months
from tc_emulator.common import atomic, save_json


def observed_arrays(winds, records, area, ocean):
    mus, seconds = window_moments(winds, area)
    # A window statistic after subtracting the same realization's actual monthly mean.
    # Average over window starts analytically. Overlapping windows supply no extra members.
    month_mean = mus[:, -1]
    within = (
        seconds
        - 2 * np.einsum("nwf,nf->nw", mus, month_mean)
        + np.einsum("nf,nf->n", month_mean, month_mean)[:, None]
    )
    np.testing.assert_allclose(within[:, -1], 0.0, atol=4e-12)
    years = np.array([r["year"] for r in records])
    members = np.array([r["member"] for r in records])
    g = np.array([r["g"] for r in records])
    length = np.array([len(w) for w in winds])
    unique = np.unique(years)
    totals, unbiased, acf, centered = [], [], [], []
    for y in unique:
        ii = np.flatnonzero(years == y)
        totals.append(variance(mus[ii], seconds[ii]))
        # E||sample mean||² includes Var(member's mean)/N. Undo this finite-member term.
        correction = np.square(mus[ii] - mus[ii].mean(axis=0)).sum(axis=(0, 2)) / (
            len(ii) * (len(ii) - 1)
        )
        unbiased.append(totals[-1] + correction)
        centered.append(within[ii].mean(axis=0))
        _, c = component_cov([winds[i] for i in ii], ocean)
        acf.append(c / c[:, :1])
    return dict(
        year=years,
        member=members,
        g=g,
        length=length,
        unique=unique,
        mu=mus,
        second=seconds,
        within=within,
        by_year_raw=np.array(totals),
        by_year_cov=np.array(unbiased),
        by_year_within=np.array(centered),
        by_year_acf=np.array(acf),
    )


def cache(cohort, month, role):
    path = WORK / "wind_moments" / cohort / role / f"{month:02}.npz"
    if path.exists():
        return
    start = time.monotonic()
    record("wind_targets_start", cohort=cohort, role=role, month=month)
    cfg = CONFIG if cohort == "original" else WORK / "configs" / cohort
    data, records, _ = months(cfg, WORK / "data", month, role)
    rep = MonthlyModel(DEFAULT_WORK / "weights", month).rep
    area, ocean, _ = metrics(rep)
    arrays = observed_arrays([d["wind"] for d in data], records, area, ocean)
    with atomic(path) as tmp:
        np.savez(tmp, **arrays)
    save_json(
        path.with_suffix(".json"),
        {
            "records": records,
            "area_mask": "published steering basin and original SST ocean mask",
            "groups": "each year and ensemble identity retain all consecutive days",
        },
    )
    record(
        "wind_targets_done",
        month=month,
        cohort=cohort,
        role=role,
        members=len(set(arrays["member"])),
        years=len(arrays["unique"]),
        cases=len(records),
        seconds=round(time.monotonic() - start, 2),
    )
    print(cohort, role, month, len(records), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", choices=["original", "expanded", "extended"], default="original")
    p.add_argument("--month", type=int, required=True)
    args = p.parse_args()
    cache(args.cohort, args.month, "train")
    if args.cohort == "original":
        cache(args.cohort, args.month, "val")
