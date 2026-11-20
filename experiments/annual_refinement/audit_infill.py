"""Audit every ocean-only training map against its independently prepared, wind-paired counterpart."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from records import WORK, record

from annual_lim.layout import CONFIG, read
from tc_emulator.common import save_json, sha


def check(case):
    member, year = case
    ocean = WORK / "ocean_only" / member / str(year)
    full = WORK / "infill/prepared" / member / str(year)
    a = read(ocean / "complete.json")
    b = read(full / "complete.json")
    for key in ("member", "year", "g", "calendar"):
        if a[key] != b[key]:
            raise ValueError(f"{member}/{year}: {key} mismatch")
    if b["role"] != "train" or sha(ocean / "maps.npy") != a["sha256"]:
        raise ValueError("Case or checksum mismatch")
    snapshots = np.load(ocean / "maps.npy", mmap_mode="r")
    count = 0
    for m, month in enumerate(snapshots, 1):
        path = full / f"{m:02}.npz"
        if sha(path) != b["hashes"][path.name]:
            raise ValueError(f"Changed prepared source {path}")
        with np.load(path) as f:
            for k, field in enumerate(a["fields"]):
                if not np.array_equal(month[k], f[field], equal_nan=True):
                    raise ValueError(f"Ocean processors differ for {member}, {year}, {m}, {field}")
                count += 1
    return count


if __name__ == "__main__":
    years = read(Path(__file__).with_name("ocean_extra_years.json"))["years"]
    cases = [(m, y) for m in read(CONFIG / "members.json")["train"] for y in years]
    with ThreadPoolExecutor(4) as pool:
        total = sum(pool.map(check, cases))
    save_json(
        WORK / "experiment_records/infill_audit.json",
        dict(
            members=35,
            years=years,
            member_years=len(cases),
            months=12,
            fields=["sst", "mld", "gamma"],
            identical_field_maps=total,
            comparison="exact, including wet-cell mask",
        ),
    )
    record("infill_ocean_audit", cases=len(cases), identical_maps=total)
