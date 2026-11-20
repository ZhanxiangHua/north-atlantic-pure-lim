"""Nested tests of warming-dependent retention and two bands of Gaussian wind memory."""
import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from records import WORK, record
from scipy.optimize import least_squares
from weather import cache_read, kernels, model_terms, year_table

from annual_lim.layout import DEFAULT_WORK
from annual_lim.model import MonthlyModel
from annual_lim.statistics import variance
from tc_emulator.common import atomic, save_json


@dataclass(frozen=True)
class ThermalFilter:
    center: float
    gmin: float
    gmax: float
    intercept: float
    slope: float
    retention: float
    retention_slope: float
    feedback: float
    first_fraction: float

    def local(self, g):
        x = np.clip(g, self.gmin - 1.25, self.gmax + 1.25) - self.center
        return np.clip(self.retention + self.retention_slope * x, 0.60, 1.0), np.exp(
            self.intercept + self.slope * x
        )

    def taps(self):
        t = np.zeros(21)
        t[0] = 1.0
        t[5:12] = -self.feedback * self.first_fraction / 7
        t[12:21] = -self.feedback * (1 - self.first_fraction) / 9
        return t


def calculated(f, lags, g, length, bg):
    r, amp = f.local(g)
    t = f.taps()
    corr = np.correlate(t, t, mode="full")
    ind = np.abs(np.arange(31)[:, None] + np.arange(-20, 21))
    decay = r[:, None] ** np.arange(51)
    weighted = (lags[None] * decay[:, None, :])[:, :, ind] @ corr
    norm = lags[0, 0] / weighted[:, 0, 0]
    weather = weighted * (norm * amp)[:, None, None]
    # Each month length has its own window centering operator, including leap February.
    raw = np.einsum("nij,nj->ni", np.array([kernels(int(n))[0] for n in length]), weather[:, 0])
    within = np.einsum("nij,nj->ni", np.array([kernels(int(n))[1] for n in length]), weather[:, 0])
    acvf = bg[:, 1:, None] + weather[:, 1:, :16]
    return raw + bg[:, :1], within, acvf / acvf[:, :, :1]


def fit(model, data, lags, bg, kind, within_weight, penalty):
    years, g, nd = year_table(data)
    orig = model.cal
    obs = data["by_year_raw"]
    acf = data["by_year_acf"][:, :, 1:]
    add_r = kind in ("thermal", "both")
    add_p = kind in ("bands", "both")
    start = [orig.intercept, orig.slope, orig.retention, orig.feedback]
    lo = [-0.7, -0.35, 0.65, 0.0]
    hi = [0.7, 0.35, 1.0, 1.4]
    if add_r:
        start.append(0.0)
        lo.append(-0.10)
        hi.append(0.10)
    if add_p:
        start.append(7 / 16)
        lo.append(0.0)
        hi.append(1.0)

    def convert(x):
        return ThermalFilter(
            float(np.mean(g)),
            float(g.min()),
            float(g.max()),
            *map(float, x[:3]),
            float(x[4]) if add_r else 0.0,
            float(x[3]),
            float(x[-1]) if add_p else 7 / 16,
        )

    def fun(x):
        f = convert(x)
        p, c, a = calculated(f, lags, g, nd, bg)
        parts = [
            (np.log(p / obs) / np.sqrt(5)).ravel(),
            ((a[:, :, 1:] - acf) / np.sqrt(60)).ravel(),
        ]
        if within_weight:
            parts.append(
                (
                    np.sqrt(within_weight / 4) * np.log(c[:, :4] / data["by_year_within"][:, :4])
                ).ravel()
            )
        # Weak shrinkage in units of the training climate range and the number of independent years.
        if add_r:
            parts.append(np.array([f.retention_slope * np.ptp(g) * np.sqrt(len(g) * penalty)]))
        if add_p:
            parts.append(np.array([(f.first_fraction - 7 / 16) * np.sqrt(len(g) * penalty)]))
        return np.concatenate(parts)

    result = least_squares(
        fun, start, bounds=(lo, hi), max_nfev=160, ftol=1e-9, gtol=1e-9, xtol=1e-9
    )
    if not result.success:
        raise RuntimeError(result.message)
    return convert(result.x), 2 * result.cost / len(g)


def run(month, weights, cohort, tag):
    path = WORK / "thermal" / tag / f"{month:02}.parquet"
    if path.exists():
        return
    t = time.monotonic()
    record("thermal_start", cohort=cohort, tag=tag, month=month)
    model = MonthlyModel(weights, month)
    tr = cache_read(cohort, "train", month)
    v = cache_read("original", "val", month)
    yt, gt, nt = year_table(tr)
    yv, gv, nv = year_table(v)
    lags, bb, force = model_terms(model, np.r_[gt, gv])
    ids = np.searchsorted(yv, v["year"])
    rows = []
    fits = {}
    for kind in ("constant", "thermal", "bands", "both"):
        for w in (0.0, 0.5, 1.5):
            for strength in (0.0,) if kind == "constant" else (0.0, 0.01, 0.1):
                f, l = fit(model, tr, lags, bb[: len(gt)], kind, w, strength)
                name = f"{kind}_w{w:g}_p{strength:g}"
                fits[name] = dict(parameters=asdict(f), loss=l)
                tot, cc, acal = calculated(f, lags, gv, nv, bb[len(gt) :])
                for grp, mm in [
                    ("all", np.ones(len(ids), bool)),
                    ("cold", v["g"] < -0.0115220616351514),
                    ("warm", v["g"] > 3.419637285227324),
                    ("inside", (v["g"] >= -0.0115220616351514) & (v["g"] <= 3.419637285227324)),
                ]:
                    ii = ids[mm]
                    obs = variance(v["mu"][mm], v["second"][mm])
                    totals = (tot[ii].mean(axis=0) + np.var(v["g"][mm]) * force) / obs
                    centers = cc[ii, :4].mean(axis=0) / v["within"][mm, :4].mean(axis=0)
                    for task, seq in (("total", totals), ("within", centers)):
                        for d, ratio in zip(("1", "5", "10", "15", "month"), seq):
                            rows.append(
                                dict(
                                    month=month,
                                    recipe=name,
                                    group=grp,
                                    target=task,
                                    window=d,
                                    ratio=ratio,
                                    abs_log_error=abs(np.log(ratio)),
                                )
                            )
    with atomic(path) as tmp:
        pd.DataFrame(rows).to_parquet(tmp, index=False, compression="zstd")
    save_json(path.with_suffix(".json"), fits)
    record(
        "thermal_done",
        tag=tag,
        cohort=cohort,
        month=month,
        recipes=len(fits),
        seconds=round(time.monotonic() - t, 2),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--cohort", default="original")
    p.add_argument("--tag", default="published_backbone")
    p.add_argument("--weights", type=Path, default=DEFAULT_WORK / "weights")
    a = p.parse_args()
    run(a.month, a.weights, a.cohort, a.tag)
