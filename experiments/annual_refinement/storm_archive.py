"""Publish the paired synthetic storm attempts and six-hour coordinates behind the scores."""
from pathlib import Path

import numpy as np
import pandas as pd
from records import WORK

from tc_emulator.common import atomic, sha

KEYS = ["model", "protocol", "year", "month", "member", "draw", "origin", "seed_id"]


def archive(candidate, root, panel):
    storage, count, prefix = (
        ("storm_validation", 126, "seasonal_validation")
        if panel == "val"
        else ("storm_ssp245_year", 96, "annual_mpi2050")
    )
    tags = ["mpi_daily", "published_v1", "native_fourier", candidate]
    if panel == "ssp245_year":
        tags.append("selected_fourier")
    attempts = []
    fixes = []
    for tag in tags:
        folders = sorted(path.parent for path in (WORK / storage / tag).glob("*/*/*/complete.json"))
        if len(folders) != count:
            raise ValueError(f"Incomplete storm archive {panel}/{tag}")
        for folder in folders:
            attempts.append(pd.read_parquet(folder / "attempts.parquet"))
            fixes.append(pd.read_parquet(folder / "fixes.parquet"))
    attempts = pd.concat(attempts, ignore_index=True)
    fixes = pd.concat(fixes, ignore_index=True)
    if attempts.duplicated(KEYS).any() or fixes.duplicated([*KEYS, "elapsed_h"]).any():
        raise ValueError("Duplicate storm or six-hour observation")
    actual = pd.MultiIndex.from_frame(attempts.loc[attempts.tc, KEYS]).sort_values()
    archived = pd.MultiIndex.from_frame(fixes[KEYS].drop_duplicates()).sort_values()
    if not actual.equals(archived) or not np.allclose(fixes.elapsed_h % 6.0, 0.0, atol=1e-7):
        raise ValueError("The six-hour tracks do not match the accepted storms")
    hashes = {}
    for frame, keys, name in (
        (attempts, KEYS, "attempts"),
        (fixes, [*KEYS, "elapsed_h"], "tracks_6h"),
    ):
        path = Path(root) / (prefix + "_" + name + ".parquet")
        with atomic(path) as tmp:
            frame.sort_values(keys).to_parquet(tmp, index=False, compression="zstd")
        hashes[path.name] = sha(path)
    return dict(
        attempts=len(attempts), tracks=len(actual), six_hour_fixes=len(fixes), sha256=hashes
    )
