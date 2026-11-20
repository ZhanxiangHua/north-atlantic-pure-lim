"""Reconstruct the annual forcing from the archived training-member global monthly tas."""
from pathlib import Path

import numpy as np
import pandas as pd

from tc_emulator.common import atomic
from .layout import read


def rebuild(config, output):
    with np.load(Path(config) / "global_monthly_tas.npz", allow_pickle=False) as a:
        people = a["members"].tolist()
        if people != read(Path(config) / "members.json")["train"]:
            raise ValueError("Forced warming must be based on the designated fitting members")
        month_id, days, temperature = a["month_id"], a["duration_days"], a["tas_C"]
    if (
        not np.array_equal(month_id, np.arange(1850 * 12, 2101 * 12))
        or np.any(days <= 0)
        or not np.isfinite(temperature).all()
    ):
        raise ValueError("Missing calendar month or invalid global temperature")
    # For each member, annual mean = sum(days * monthly mean)/sum(days).
    annual = (temperature * days).reshape(len(people), -1, 12).sum(axis=2) / days.reshape(
        len(people), -1, 12
    ).sum(axis=2)
    year = month_id[::12] // 12
    forced = annual.mean(axis=0)
    baseline = forced[(year >= 1850) & (year <= 1900)].mean()
    df = pd.DataFrame(
        dict(
            year=year,
            experiment=np.where(year < 2015, "historical", "ssp585"),
            g=forced - baseline,
            g_members=len(people),
        )
    )
    with atomic(Path(output)) as path:
        df.to_csv(path, index=False)
    return float(baseline)
