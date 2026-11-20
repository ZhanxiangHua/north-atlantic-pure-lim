"""Prepare scenario-specific warming and complete calendar years of held-out MPI weather."""
import argparse
import calendar
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from annual_lim.layout import CONFIG, case_dir, design, read
from annual_lim.prepare import assemble, fetch
from midcentury_sources import PLAN, REPORT
from tc_emulator.common import MODEL, ROOT, TABLES, atomic, digest, lock, save_json, sha

STUDY = Path(os.environ.get("TC_MIDCENTURY_WORK", ROOT / "runs/ssp245_multiyear")).resolve()
CFG = STUDY / "configs/ssp245_year"


@lru_cache
def sources():
    return pd.read_csv(REPORT / "cmip6_files.csv")


def files(member, year, mirror):
    catalog = sources()
    match = catalog[(catalog.member == member) & (catalog.first_year <= year) & (year <= catalog.last_year)]
    result = {}
    for row in match.itertuples(index=False):
        rel = Path("ScenarioMIP/MPI-M") / MODEL / "ssp245" / member / row.table / row.variable / "gn" / row.version / row.file
        result[row.variable] = SimpleNamespace(**row._asdict(), local=str(mirror / rel), aws="")
    return result


def global_tas(job):
    member, year, mirror = job
    source = files(member, year, mirror)["tas"]
    raw, _ = fetch(source, year, read(CONFIG / "settings.json"))
    return dict(member=member, year=year, annual_tas_C=np.average(raw["data"], weights=raw["ndays"]),
                days=sum(raw["ndays"]), source_sha256=source.checksum)


def configure(mirror, workers):
    plan, split = read(PLAN), read(CONFIG / "members.json")
    years = sorted([*plan["new_years"], plan["reference_year"]])
    with ProcessPoolExecutor(workers) as pool:
        tas = pd.DataFrame(pool.map(global_tas, [(m, y, mirror) for y in years for m in split["train"]]))
    annual = tas.groupby("year").annual_tas_C.mean().rename("annual_tas_C").to_frame()
    annual["historical_reference_C"] = split["reference_degC"]["g"]
    annual["g"] = annual.annual_tas_C - annual.historical_reference_C
    old = read(ROOT / "reports/mpi2050/forcing.json")
    np.testing.assert_allclose(annual.loc[old["year"], "g"], old["warming_K"], atol=1e-9, rtol=0)
    if len(split["test"]) != 8 or set(split["test"]) & set(tas.member) or not (tas.groupby("year").member.nunique() == 35).all():
        raise ValueError("Incorrect set of forcing or evaluation members")
    for y in years:
        np.testing.assert_array_equal(tas.loc[tas.year == y, "days"].unique(), [365 + calendar.isleap(y)])
    settings = dict(read(CONFIG / "settings.json"), train_years=[], validation_years=[], test_years=plan["new_years"])
    for frame, path in [(tas, REPORT / "forcing_members.csv"), (annual.reset_index(), REPORT / "forcing.csv"),
                        (annual.loc[plan["new_years"], ["g"]].reset_index(), CFG / "forcing.csv")]:
        with atomic(path) as tmp:
            frame.to_csv(tmp, index=False)
    save_json(CFG / "settings.json", settings)
    save_json(CFG / "members.json", split)
    save_json(STUDY / "FROZEN.json", read(ROOT / "reports/annual_refinement/freeze_receipt.json"))
    print(annual.to_string(), flush=True)


def prepare(index, mirror):
    settings, cases = design(CFG)
    role, member, year, g = cases[index]
    specs = files(member, year, mirror)
    if set(specs) != set(TABLES):
        raise ValueError(f"Missing fields for {member}/{year}")
    folder = case_dir(STUDY / "ssp245_year", member, year)
    signature = digest([member, year, g, settings["domain"], settings["depth_m"], settings["halo_degrees"],
                        sorted(s.checksum for s in specs.values())])
    with lock(folder / ".lock"):
        marker = folder / "complete.json"
        if marker.is_file():
            old = read(marker)
            if old["signature"] == signature and all(sha(folder / k) == h for k, h in old["hashes"].items()):
                return
        raw, provenance = {}, {}
        for var, spec in specs.items():
            raw[var], provenance[var] = fetch(spec, year, settings)
        grid = raw["ta"]
        np.testing.assert_array_equal(grid["ndays"], [calendar.monthrange(year, m)[1] for m in range(1, 13)])
        hashes = {}
        for month in range(1, 13):
            fields = assemble(raw, settings, month)
            path = folder / f"{month:02}.npz"
            with atomic(path) as tmp:
                np.savez(tmp, **fields, **{k: grid[k] for k in ("lat", "lon", "plev", "calendar")},
                         year=year, month=month, member=member)
            hashes[path.name] = sha(path)
        save_json(marker, dict(signature=signature, experiment="ssp245", role=role, member=member,
                               year=year, g=g, sources=provenance, hashes=hashes, days=grid["ndays"].tolist(),
                               calendar=grid["calendar"]))
    print(f"Prepared {member}/{year} with {int(sum(grid['ndays']))} daily winds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("forcing", "prepare"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mirror", type=Path, default=Path("/glade/collections/cmip/CMIP6"))
    args = parser.parse_args()
    if args.stage == "forcing":
        configure(args.mirror, args.workers)
    elif args.index is None:
        parser.error("prepare requires --index")
    else:
        prepare(args.index, args.mirror)
