"""Prepare all twelve months of the independent 2050 MPI SSP2-4.5 reference."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
from records import WORK, record

from annual_lim.layout import CONFIG, case_dir, design, read
from annual_lim.prepare import assemble, fetch
from tc_emulator.common import MODEL, ROOT, TABLES, atomic, digest, lock, save_json, sha

YEAR = 2050
PANEL = "ssp245_year"
SOURCE = ROOT / "reports/mpi2050"


def configure():
    cfg = WORK / "configs" / PANEL
    forcing = read(SOURCE / "forcing.json")
    members = read(CONFIG / "members.json")
    if (forcing["scenario"], forcing["year"], set(forcing["members"])) != (
        "ssp245",
        YEAR,
        set(members["train"]),
    ):
        raise ValueError("Scenario forcing lineage differs from the fixed MPI benchmark")
    settings = dict(
        read(CONFIG / "settings.json"), train_years=[], validation_years=[], test_years=[YEAR]
    )
    series = pd.DataFrame({"year": [YEAR], "g": [forcing["warming_K"]]})
    with lock(cfg / ".lock"):
        save_json(cfg / "settings.json", settings)
        save_json(cfg / "members.json", members)
        with atomic(cfg / "forcing.csv") as tmp:
            series.to_csv(tmp, index=False)
    return cfg


def prepare(index, mirror, september_reference):
    cfg = configure()
    settings, cases = design(cfg)
    role, member, year, g = cases[index]
    catalog = pd.read_csv(SOURCE / "cmip6_files.csv")
    files = catalog.loc[catalog.member == member]
    if (
        len(files) != len(TABLES)
        or set(files.variable) != set(TABLES)
        or set(files.experiment) != {"ssp245"}
    ):
        raise ValueError(f"Incomplete or duplicate SSP2-4.5 release for {member}")
    folder = case_dir(WORK / PANEL, member, year)
    mark = folder / "complete.json"
    september = september_reference / (member + ".nc") if september_reference else None
    signature = digest(
        [
            year,
            member,
            g,
            settings["domain"],
            settings["halo_degrees"],
            settings["depth_m"],
            sorted(files.checksum.tolist()),
            sha(september) if september else None,
        ]
    )
    with lock(folder / ".lock"):
        if mark.exists():
            old = read(mark)
            if old["signature"] == signature and all(
                sha(folder / n) == h for n, h in old["hashes"].items()
            ):
                return
        record("annual_ssp245_preparation_start", member=member, year=year)
        raw, sources = {}, {}
        for row in files.itertuples(index=False):
            rel = (
                Path("ScenarioMIP/MPI-M")
                / MODEL
                / row.experiment
                / member
                / row.table
                / row.variable
                / "gn"
                / row.version
                / row.file
            )
            spec = SimpleNamespace(
                **row._asdict(), aws="", local=str(mirror / rel) if mirror else ""
            )
            raw[row.variable], sources[row.variable] = fetch(spec, year, settings)
        grid = raw["ta"]
        if int(sum(grid["ndays"])) != 365 or len(grid["ndays"]) != 12:
            raise ValueError(f"Incomplete 2050 calendar in {member}")
        hashes = {}
        for month in range(1, 13):
            fields = assemble(raw, settings, month)
            if month == 9 and september:
                with xr.open_dataset(september) as ref:
                    for field, arr in fields.items():
                        np.testing.assert_array_equal(
                            arr, ref[field].values, err_msg=f"{member}/{field}"
                        )
                    for coord in ("lat", "lon", "plev"):
                        np.testing.assert_array_equal(grid[coord], ref[coord].values)
            fname = f"{month:02}.npz"
            with atomic(folder / fname) as tmp:
                np.savez(
                    tmp,
                    **fields,
                    lat=grid["lat"],
                    lon=grid["lon"],
                    plev=grid["plev"],
                    year=year,
                    month=month,
                    member=member,
                    calendar=grid["calendar"],
                )
            hashes[fname] = sha(folder / fname)
        save_json(
            mark,
            dict(
                signature=signature,
                role=role,
                member=member,
                year=year,
                g=g,
                sources=sources,
                hashes=hashes,
                days=grid["ndays"].tolist(),
                calendar=grid["calendar"],
                september_reference_sha256=sha(september) if september else None,
            ),
        )
    record(
        "annual_ssp245_prepared",
        member=member,
        year=year,
        months=12,
        daily_fields=365,
        september_fields_crosschecked=7 if september else 0,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True, type=int, choices=range(8))
    p.add_argument("--mirror", type=Path)
    p.add_argument("--september-reference", type=Path)
    a = p.parse_args()
    prepare(a.index, a.mirror, a.september_reference)
