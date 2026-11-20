"""Physical within-month variances for deep shear and steering in the TC corridor."""
import argparse
import time

import numpy as np
from records import WORK, record

from annual_lim.layout import CONFIG, DEFAULT_WORK
from annual_lim.model import MonthlyModel
from annual_lim.statistics import metrics
from annual_lim.train import months
from tc_emulator.common import atomic
from tc_emulator.diagnostics import steering


def domains(rep):
    area, ocean, _ = metrics(rep)
    lat, lon = np.meshgrid(rep.lat, rep.lon, indexing="ij")
    corridor = ocean * ((10 <= lat) & (lat <= 30) & (280 <= lon) & (lon <= 340))
    corridor /= corridor.sum()
    return (
        ("shear_ocean", "shear", ocean),
        ("steer_corridor", "steer", corridor),
        ("shear_corridor", "shear", corridor),
    )


def contrast(wind, kind):
    return steering(wind) if kind == "steer" else wind[..., :2, :, :] - wind[..., 2:, :, :]


def extra_metrics(rep):
    block = rep.blocks["wind"]
    k = block.basis.shape[1]
    maps = block.inverse(np.eye(k)) - block.inverse(np.zeros((1, k)))
    out = []
    for _, kind, area in domains(rep):
        b = (contrast(maps, kind) * np.sqrt(area / 2)).reshape(k, -1)
        out.append(b @ b.T)
    return np.stack(out)


def make(cohort, role, month):
    out = WORK / "regional_wind" / cohort / role / f"{month:02}.npz"
    if out.is_file():
        return
    t = time.monotonic()
    cfg = CONFIG if cohort == "original" else WORK / "configs" / cohort
    samples, records, _ = months(cfg, WORK / "data", month, role)
    dom = domains(MonthlyModel(DEFAULT_WORK / "weights", month).rep)
    stats = np.empty((len(samples), len(dom), 4))
    for i, case in enumerate(samples):
        wind = case["wind"]
        for j, (_, kind, area) in enumerate(dom):
            x = contrast(wind, kind).astype(float)
            x -= x.mean(axis=0)
            sums = np.concatenate((np.zeros_like(x[:1]), x.cumsum(axis=0)))
            for k, d in enumerate((1, 5, 10, 15)):
                avg = (sums[d:] - sums[:-d]) / d
                stats[i, j, k] = np.sum(np.square(avg).mean(axis=(0, 1)) * area)
    with atomic(out) as tmp:
        np.savez(
            tmp,
            within=stats,
            metric=[a[0] for a in dom],
            years=[r["year"] for r in records],
            members=[r["member"] for r in records],
            g=[r["g"] for r in records],
        )
    record(
        "regional_wind_targets",
        cohort=cohort,
        role=role,
        month=month,
        cases=len(samples),
        seconds=round(time.monotonic() - t, 2),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default="original")
    p.add_argument("--month", type=int, required=True)
    a = p.parse_args()
    make(a.cohort, "train", a.month)
    if a.cohort == "original":
        make(a.cohort, "val", a.month)
