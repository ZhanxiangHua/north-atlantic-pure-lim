"""Rotate the 192-dimensional training wind space to preserve selected temporal and vertical contrasts."""
import argparse
import time

import numpy as np
import pandas as pd
from backbone import Bank, Recipe
from records import WORK, record

from annual_lim.layout import CONFIG
from annual_lim.train import months
from tc_emulator.common import atomic
from tc_emulator.diagnostics import region_area, steering

# Daily, five-day, fifteen-day and monthly covariances each get unit trace before mixing.
MIXES = {
    "mix_daily": ((1.0, 0.0, 0.0, 0.0), 0.0),
    "mix_slow25": ((0.75, 0.10, 0.10, 0.05), 0.0),
    "mix_slow50": ((0.50, 0.20, 0.20, 0.10), 0.0),
    "mix_slow75": ((0.25, 0.25, 0.30, 0.2), 0.0),
    "mix_slow50_shear15": ((0.50, 0.20, 0.20, 0.10), 0.15),
    "mix_slow50_shear35": ((0.50, 0.20, 0.20, 0.10), 0.35),
    "mix_daily_shear25": ((1.0, 0.0, 0.0, 0.0), 0.25),
}


def average(x, d):
    total = np.concatenate([np.zeros_like(x[:1]), np.cumsum(x, axis=0, dtype=float)])
    return (total[d:] - total[:-d]) / d


def rotations(bank):
    scores = bank.daily(Recipe(wind_basis="wind", wind=192), "train")
    nday = bank.scores["train_days"]
    cov = np.zeros((4, 192, 192))
    mean = np.zeros((4, 192))
    for p, n in zip(scores, nday):
        for j, d in enumerate((1, 5, 15, n)):
            a = average(p[:n], d)
            cov[j] += a.T @ a / len(a)
            mean[j] += a.mean(axis=0)
    cov /= len(scores)
    mean /= len(scores)
    cov -= np.einsum("ti,tj->tij", mean, mean)
    cov /= np.trace(cov, axis1=1, axis2=2)[:, None, None]
    b = bank.block("wind")
    allmaps = np.zeros((192, b.valid.size))
    allmaps[:, b.valid.ravel()] = b.basis.T / b.weights[b.valid]
    allmaps = allmaps.reshape(192, *b.valid.shape)
    area = region_area(bank.geo.lat, bank.geo.lon)
    shear = (allmaps[:, :2] - allmaps[:, 2:]) * np.sqrt(area / 2)
    matrix = shear.reshape(192, -1) @ shear.reshape(192, -1).T
    out = {}
    for name, (w, gain) in MIXES.items():
        C = np.einsum("t,tij->ij", w, cov)
        geared = C @ matrix @ C
        target = (1 - gain) * C + gain * geared / np.trace(geared)
        _, v = np.linalg.eigh(target)
        out[name] = v[:, ::-1].copy()
    return out


def run(cohort, month):
    bank = Bank(cohort, month)
    target = bank.root / "wind_basis_validation.parquet"
    if target.exists():
        return
    started = time.monotonic()
    record("wind_basis_start", month=month, cohort=cohort)
    combos = rotations(bank)
    with atomic(bank.root / "rotations.npz") as tmp:
        np.savez(tmp, **combos)
    data, records, _ = months(CONFIG, WORK / "data", month, "val")
    lon, lat = bank.geo.lon, bank.geo.lat
    area = region_area(lat, lon)
    basin = np.broadcast_to(area, (2, *area.shape)).reshape(-1) / 2
    rows = []
    for name in ("published_wind", "wind", *combos):
        raw = bank.block(name)
        for rank in (
            (120,)
            if name == "published_wind"
            else (96, 120, 160, 192)
            if name == "wind"
            else (96, 120, 160)
        ):
            daily = bank.daily(Recipe(wind_basis=name, wind=rank), "val")
            for i, (ds, rec) in enumerate(zip(data, records)):
                path = ds["wind"]
                pc = daily[i, : len(path)]
                wind = raw.inverse(pc[:, :rank])
                residual = wind - path
                for j, d in enumerate([1, 5, 15, len(path)]):
                    dif = average(residual, d)
                    steer = steering(dif).reshape(len(dif), -1)
                    shear = (dif[:, :2] - dif[:, 2:]).reshape(len(dif), -1)
                    rows.append(
                        dict(
                            basis=name,
                            rank=rank,
                            month=month,
                            member=rec["member"],
                            year=rec["year"],
                            g=rec["g"],
                            window=("1", "5", "15", "month")[j],
                            steering_mse=np.mean((steer**2) @ basin),
                            shear_mse=np.mean((shear**2) @ basin),
                        )
                    )
    with atomic(target) as tmp:
        pd.DataFrame(rows).to_parquet(tmp, index=False, compression="zstd")
    record(
        "wind_basis_done",
        month=month,
        cohort=cohort,
        rows=len(rows),
        seconds=round(time.monotonic() - started, 2),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default="original")
    p.add_argument("--month", required=True, type=int)
    a = p.parse_args()
    run(a.cohort, a.month)
