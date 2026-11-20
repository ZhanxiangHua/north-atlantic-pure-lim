"""Fetch the pinned SSP2-4.5 reference months and compute the MPI annual warming."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from annual_lim.layout import CONFIG, read
from annual_lim.prepare import assemble, fetch
from tc_emulator.common import COMPONENTS, MODEL, ROOT, TABLES, atomic, save_json, sha
from tc_emulator.physical import thermodynamics
from tc_emulator.sampling import UNITS

REPORT = ROOT / "reports/mpi2050"
WORK = ROOT / "runs/mpi2050"
YEAR, MONTH = 2050, 9


def get_case(member, *, files, roles, settings, work, mirror):
    rows = files[member]
    fitting_member = member in roles["train"]
    target = work / (f"tas/{member}.json" if fitting_member else f"reference/{member}.nc")
    marker = target.with_suffix(".verified.json")
    signature = rows.checksum.sort_values().tolist()
    if target.exists() and marker.exists():
        record = read(marker)
        if record["sources"] == signature and sha(target) == record["output_sha256"]:
            return str(target)

    raw = {}
    for row in rows.itertuples(index=False):
        rel = Path("ScenarioMIP/MPI-M") / MODEL / row.experiment / member / row.table / row.variable / "gn" / row.version / row.file
        spec = SimpleNamespace(**row._asdict(), member=member, aws="", local=str(mirror / rel) if mirror else "")
        raw[row.variable], _ = fetch(spec, YEAR, settings, month=None if fitting_member else MONTH)
    if fitting_member:
        mean = np.average(raw["tas"]["data"], weights=raw["tas"]["ndays"])
        save_json(target, {"member": member, "annual_tas_C": float(mean)})
    else:
        if set(raw) != set(TABLES):
            raise ValueError(f"Incomplete reference fields for {member}")
        fields = assemble(raw, settings, MONTH)
        grid = raw["ta"]
        fields.update(thermodynamics(fields["sst"], fields["psl"], grid["plev"], fields["ta"], fields["hus"]))
        coords = {name: grid[name] for name in ("plev", "lat", "lon")}
        coords.update(component=COMPONENTS, day=np.arange(1, 31),
                      time=("day", np.arange(30) + 0.5, {"units": "days since 2050-09-01", "calendar": grid["calendar"]}))
        ds = xr.Dataset(coords=coords, attrs=dict(source=MODEL, experiment="ssp245", member=member,
                                                 year=YEAR, month=MONTH, calendar=grid["calendar"]))
        for name, array in fields.items():
            dims = ("day", "component") if name == "wind" else ("plev",) if name in ("ta", "hus") else ()
            ds[name] = (dims + ("lat", "lon"), array.astype("float32"),
                        {"units": "m s-1" if name == "wind" else UNITS[name]})
        if ds.wind.shape[:2] != (30, 4) or not np.isfinite(ds.wind.values).all():
            raise ValueError("Missing daily wind")
        with atomic(target) as temp:
            ds.to_netcdf(temp)
    save_json(marker, {"sources": signature, "output_sha256": sha(target)})
    return str(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--mirror", type=Path, help="Optional local CMIP6 archive root")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    roles = read(CONFIG / "members.json")
    files = pd.read_csv(REPORT / "cmip6_files.csv")
    if set(files.member) != set(roles["train"] + roles["test"]) or files.duplicated(["member", "variable"]).any():
        raise ValueError("Source manifest does not match the published split")
    grouped = {m: f.drop(columns="member") for m, f in files.groupby("member")}
    read_member = partial(get_case, files=grouped, roles=roles, settings=read(CONFIG / "settings.json"),
                          work=args.work, mirror=args.mirror)
    with ProcessPoolExecutor(args.workers) as pool:
        for path in pool.map(read_member, roles["test"] + roles["train"]):
            print(path, flush=True)
    annual = [read(args.work / f"tas/{m}.json")["annual_tas_C"] for m in roles["train"]]
    baseline = float(roles["reference_degC"]["g"])
    save_json(args.work / "forcing.json", {"year": YEAR, "scenario": "ssp245", "members": roles["train"],
              "annual_tas_C": float(np.mean(annual)), "historical_reference_C": baseline,
              "warming_K": float(np.mean(annual) - baseline)})


if __name__ == "__main__":
    main()
