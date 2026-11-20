"""Matched seasonal FAST replays with daily MPI, pure LIM and native Fourier winds."""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from ocean import data_paths, panel_cohort
from records import WORK, record

from annual_lim.layout import DEFAULT_WORK, case_dir, design, read, rng_for
from annual_lim.sample import event
from tc_emulator.common import COMPONENTS, ROOT, atomic, save_json, sha
from tc_emulator.fourier_weather import NativeFourierWind, WindStatistics
from tc_emulator.physical import thermodynamics
from tc_emulator.sampling import UNITS
from tc_emulator.tc_adapter import panel, replay_event

sys.path.insert(0, str(ROOT / "notebooks"))
from benchmark_mpi2050 import sampled_window, storm_records
from tc_benchmark import fourier_inputs

SEED = 20261120
MONTHS = (7, 9, 11)
ORIGINS = (5, 12)
SEEDS = 96


def storage(panel_name):
    return WORK / ("storm_validation" if panel_name == "val" else "storm_ssp245_year")


def observed(member, year, month, panel_name):
    cohort, split = panel_cohort(panel_name)
    _, root = data_paths(cohort)
    folder = case_dir(root, member, year)
    info = read(folder / "complete.json")
    fn = f"{month:02}.npz"
    if info["role"] != split or sha(folder / fn) != info["hashes"][fn]:
        raise ValueError("Invalid MPI source release or member split")
    with np.load(folder / fn) as f:
        dim = {k: f[k] for k in ("lat", "lon", "plev")}
        fields = {k: f[k] for k in ("sst", "psl", "mld", "gamma", "ta", "hus", "wind")}
    fields.update(
        thermodynamics(fields["sst"], fields["psl"], dim["plev"], fields["ta"], fields["hus"])
    )
    n = len(fields["wind"])
    dim.update(
        component=COMPONENTS,
        day=np.arange(1, n + 1),
        time=(
            "day",
            np.arange(n) + 0.5,
            {"units": f"days since {year:04}-{month:02}-01", "calendar": info["calendar"]},
        ),
    )
    ds = xr.Dataset(
        coords=dim,
        attrs=dict(member=member, month=month, year=year, experiment=info.get("experiment", "MPI")),
    )
    for key, v in fields.items():
        dims = ("day", "component") if key == "wind" else ("plev",) if key in ("ta", "hus") else ()
        ds[key] = (dims + ("lat", "lon"), v.astype("float32"), {"units": UNITS.get(key, "m s-1")})
    return ds, info["hashes"][fn]


def simulate(item):
    tag, weights, method, member, year, g, month, panel_name = item
    dest = storage(panel_name) / tag / f"{month:02}" / str(year) / member
    if (dest / "complete.json").is_file():
        return
    ref, checksum = observed(member, year, month, panel_name)
    rows = []
    fixes = []
    for draw, origin in enumerate(ORIGINS):
        key = f"{year}/{month}/{member}/{draw}"
        seeds = panel(int(rng_for(SEED, "FAST", key, origin).integers(2**32)), SEEDS)
        for protocol in (
            ("MPI daily",)
            if method == "MPI"
            else ("MPI background",)
            if method == "Fourier"
            else ("MPI background", "warming only")
        ):
            provider = None
            if method == "MPI":
                ev = replay_event(ref, 15, origin_day=origin)
            elif method == "Fourier":
                ev = replay_event(ref, 15, origin_day=origin)
                model, cov = fourier_inputs(weights, month)
                mean = model.wind_mean(g, model.rep.encode(ref))
                wind = model.rep.blocks["wind"].inverse(mean[None])[0]
                stats = WindStatistics(wind, cov * model.cal.amplitude(g) ** 2)
                provider = NativeFourierWind(
                    stats,
                    ref.lon.values,
                    ref.lat.values,
                    SEED,
                    (month, year, member),
                    draw,
                    independent_storms=True,
                    origin_day=origin,
                )
            else:
                ds = event(
                    weights,
                    month,
                    year,
                    g,
                    ref.sizes["day"],
                    SEED,
                    key,
                    draw,
                    observed=ref if protocol == "MPI background" else None,
                )
                ev = sampled_window(ds, origin, 15)
            attempts, coords = storm_records(
                ev, seeds, provider, protocol, method, member, draw, origin
            )
            rows.append(attempts.assign(year=year, month=month, g=g, model=tag))
            fixes.append(coords.assign(year=year, month=month, g=g, model=tag))
    for name, frames in (("attempts", rows), ("fixes", fixes)):
        with atomic(dest / (name + ".parquet")) as tmp:
            pd.concat(frames, ignore_index=True).to_parquet(tmp, index=False, compression="zstd")
    save_json(
        dest / "complete.json",
        dict(
            member=member,
            year=year,
            month=month,
            g=g,
            method=method,
            tag=tag,
            input_sha256=checksum,
            origins=ORIGINS,
            seeds=SEEDS,
            draws=len(ORIGINS),
            attempts=sum(map(len, rows)),
        ),
    )
    record(
        "storm_case_done",
        panel=panel_name,
        tag=tag,
        member=member,
        year=year,
        month=month,
        attempts=sum(map(len, rows)),
    )


def run(tag, weights, method, lane, lanes, workers, panel_name="val"):
    cohort, split = panel_cohort(panel_name)
    cfg, _ = data_paths(cohort)
    months = MONTHS if panel_name == "val" else range(1, 13)
    if panel_name != "val":
        receipt = read(WORK / "FROZEN.json")
        fixed = {"mpi_daily": "MPI", "native_fourier": "Fourier", "published_v1": "LIM"}
        selected = {receipt["stage"]: "LIM", "selected_fourier": "Fourier"}
        if {**fixed, **selected}.get(tag) != method:
            raise ValueError("Storm driver is absent from the frozen test comparison")
        required = (
            receipt["months"]
            if tag in selected
            else {
                f"{m:02}": read(DEFAULT_WORK / "weights" / f"{m:02}/model.json")["id"]
                for m in months
            }
        )
        if any(read(weights / f"{m:02}/model.json")["id"] != required[f"{m:02}"] for m in months):
            raise ValueError("Storm driver weights differ from the frozen specification")
    cases = [
        (tag, weights, method, m, y, g, k, panel_name)
        for role, m, y, g in design(cfg)[1]
        if role == split
        for k in months
    ]
    with ProcessPoolExecutor(workers) as pool:
        for _ in pool.map(simulate, cases[lane::lanes]):
            pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, default=DEFAULT_WORK / "weights")
    p.add_argument("--method", choices=["MPI", "Fourier", "LIM"], default="LIM")
    p.add_argument("--lane", type=int, default=0)
    p.add_argument("--lanes", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--panel", choices=["val", "ssp245_year"], default="val")
    a = p.parse_args()
    run(a.tag, a.weights, a.method, a.lane, a.lanes, a.workers, a.panel)
