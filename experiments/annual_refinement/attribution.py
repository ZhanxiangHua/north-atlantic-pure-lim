"""Trace warming-conditioned background covariance into steering and test its observed variance budget."""
import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
from ocean import data_paths, panel_cohort
from records import WORK, record
from weather import model_terms, projected

from annual_lim.layout import DEFAULT_WORK, case_dir, design, read
from annual_lim.model import MonthlyModel
from annual_lim.statistics import metrics
from tc_emulator.common import atomic, sha
from tc_emulator.diagnostics import steering
from tc_emulator.warming_circulation import transfer_matrix


def summarize(tag, source, month, role):
    loc = WORK / "attribution" / tag / role
    by_year = loc / f"{month:02}_years.csv"
    if by_year.is_file():
        return
    if role != "val":
        receipt = read(WORK / "FROZEN.json")
        if tag not in ("published_v1", receipt["stage"]):
            raise ValueError("Only a frozen model may see independent climates")
    model = MonthlyModel(source, month)
    if (
        role != "val"
        and tag != "published_v1"
        and model.meta["id"] != receipt["months"][f"{month:02}"]
    ):
        raise ValueError("The model differs from the sealed receipt")
    area, _, metric = metrics(model.rep)
    panel, wanted = panel_cohort(role)
    cfg, data_root = data_paths(panel)
    cases = [(m, y, g) for rr, m, y, g in design(cfg)[1] if rr == wanted]
    path = WORK / "verification/published_v1" / role / f"{month:02}/panel_moments.npz"
    with np.load(path) as d:
        assert cases == [
            (m, int(y), float(g)) for m, y, g in zip(d["member"], d["year"], d["g"], strict=True)
        ]
        actual = d["mu"][:, -1]
    meta = pd.DataFrame(cases, columns=["member", "year", "g"])
    z = []
    for m, y, _ in cases:
        folder = case_dir(data_root, m, y)
        fn = f"{month:02}.npz"
        if sha(folder / fn) != read(folder / "complete.json")["hashes"][fn]:
            raise ValueError("Changed reference input")
        with np.load(folder / fn) as data:
            z.append(
                model.rep.encode({k: data[k] for k in ("sst", "mld", "gamma", "psl", "ta", "hus")})
            )
    z = np.stack(z)
    means = model.mean.predict(model.bg, meta.g.to_numpy(), z)
    cond = (steering(model.rep.blocks["wind"].inverse(means)) * np.sqrt(area / 2)).reshape(
        len(z), -1
    )
    residual = actual - cond
    ys, first = np.unique(meta.year, return_index=True)
    fixed = np.array([-0.18, 0.0, 1.0, 2.0, 3.0, model.cal.g_max, 4.0, 4.34])
    observed = meta.g.to_numpy()[first]
    levels = np.r_[observed, fixed]
    n = np.array(
        [calendar.monthrange(y, month)[1] for y in ys]
        + [calendar.monthrange(2050, month)[1]] * len(fixed)
    )
    lags, bg, force = model_terms(model, levels)
    total, _, _ = projected(model.cal, lags, levels, n, bg)
    transfer = transfer_matrix(model.mean)
    blocks = ["sst", "ocean", "psl", "profiles"]
    nmode = [model.rep.blocks[b].basis.shape[1] for b in blocks]
    offsets = np.cumsum([0] + nmode)
    weight_kernel = transfer @ metric[0] @ transfer.T
    report = []
    partitions = []
    for i, g in enumerate(levels):
        factor = model.factor(g)
        cov = factor @ factor.T
        entries = np.empty((4, 4))
        for l in range(4):
            for r in range(4):
                s, t = slice(offsets[l], offsets[l + 1]), slice(offsets[r], offsets[r + 1])
                entries[l, r] = np.sum(weight_kernel[s, t] * cov[s, t])
        np.testing.assert_allclose(entries.sum(), bg[i, 0], rtol=1.0e-7, atol=1.0e-10)
        year = int(ys[i]) if i < len(ys) else None
        common = dict(
            month=month,
            year=year,
            g=g,
            point="member_years" if year is not None else "fixed_temperature",
            predicted_background=bg[i, 0],
            predicted_monthly_weather=total[i, -1] - bg[i, 0],
            predicted_15day_weather=total[i, 3] - bg[i, 0],
            background_share_of_month=bg[i, 0] / total[i, -1],
        )
        if year is not None:
            select = meta.year.to_numpy() == year
            loc_mean = cond[select] - cond[select].mean(axis=0)
            resid = residual[select] - residual[select].mean(axis=0)
            dof = select.sum() - 1
            empirical_background = np.square(loc_mean).sum() / dof
            empirical_weather = np.square(resid).sum() / dof
            cross = 2 * np.sum(loc_mean * resid) / dof
            truth = np.square(actual[select] - actual[select].mean(axis=0)).sum() / dof
            np.testing.assert_allclose(
                truth, empirical_background + empirical_weather + cross, rtol=3.0e-6
            )
            common.update(
                members=int(select.sum()),
                mpi_monthly_variance=truth,
                conditional_mean_on_mpi=empirical_background,
                remaining_mpi_variance=empirical_weather,
                twice_covariance=cross,
            )
        report.append(common)
        for l, name in enumerate(blocks):
            partitions.append(
                dict(
                    month=month,
                    year=year,
                    g=g,
                    point=common["point"],
                    block=name,
                    own_variance=entries[l, l],
                    net_contribution=entries[l].sum(),
                    **{f"cov_with_{other}": entries[l, r] for r, other in enumerate(blocks)},
                )
            )
    with atomic(loc / f"{month:02}_blocks.csv") as tmp:
        pd.DataFrame(partitions).to_csv(tmp, index=False)
    with atomic(by_year) as tmp:
        pd.DataFrame(report).to_csv(tmp, index=False)
    record(
        "background_transfer_verified", tag=tag, month=month, role=role, matched_cases=len(cases)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, default=DEFAULT_WORK / "weights")
    p.add_argument("--month", required=True, type=int)
    p.add_argument("--role", choices=["val", "test", "fresh", "ssp245_year"], default="val")
    a = p.parse_args()
    summarize(a.tag, a.weights, a.month, a.role)
