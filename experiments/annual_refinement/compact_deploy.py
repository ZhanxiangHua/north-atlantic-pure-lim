"""Build optional 192-coordinate wind weights and bound physical MLD at MPI's sea floor."""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from partial_wind_memory import past_directions, project_state

from annual_lim.layout import read
from annual_lim.model import MonthlyModel
from annual_lim.sample import write
from tc_emulator.common import ROOT, digest, save_json, sha


PLAN = Path(__file__).with_name("compact_ocean_confirmation_plan.json")
GRID = ROOT / "reports/annual_refinement/data/mpi_bathymetry_grid.npz"


def export(source, destination):
    plan = read(PLAN)
    rank = plan["compact_past_rank"]
    if destination.exists():
        for month in range(1, 13):
            saved = MonthlyModel(destination, month)
            if (
                saved.meta.get("compact_parent_id")
                != plan["frozen_primary"][f"{month:02}"]
                or saved.meta.get("past_wind_rank") != rank
                or saved.meta.get("confirmation_plan_sha256") != sha(PLAN)
                or len(saved.base.a) != saved.base.observed_dim + rank
            ):
                raise ValueError(
                    f"Existing directory contains different weights: {destination}"
                )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".compact-wind-", dir=destination.parent
    ) as tmp:
        stage = Path(tmp) / "weights"
        for month in range(1, 13):
            key = f"{month:02}"
            original = MonthlyModel(source, month)
            if original.meta["id"] != plan["frozen_primary"][key]:
                raise ValueError(f"Month {key} requires its frozen research weights")
            back, axes, _ = past_directions(original.base)
            short = project_state(original.base, back, axes, rank)
            folder = stage / key
            folder.mkdir(parents=True)
            info = dict(original.meta)
            info.pop("id")
            info.update(
                compact_parent_id=original.meta["id"],
                current_wind_modes=short.observed_dim,
                past_wind_rank=rank,
                wind_state_coordinates=len(short.a),
                confirmation_plan_sha256=sha(PLAN),
                wind_state_description="Current wind PCs followed by whitened predictive contrasts of the previous day.",
            )
            for filename in original.meta["files"]:
                src = source / key / filename
                if filename != "parameters.npz":
                    shutil.copyfile(src, folder / filename)
                    continue
                with np.load(src, allow_pickle=False) as data:
                    parameters = {name: data[name] for name in data.files}
                parameters.update(
                    transition=short.a, stationary=short.sigma, innovation=short.q
                )
                np.savez_compressed(folder / filename, **parameters)
            info["files"] = {name: sha(folder / name) for name in info["files"]}
            save_json(folder / "model.json", dict(info, id=digest(info)))
            loaded = MonthlyModel(stage, month)
            if loaded.base.a.shape != (short.observed_dim + rank,) * 2:
                raise ValueError("Exported wind state has the wrong shape")
        os.rename(stage, destination)


def cap_mld(dataset, grid):
    if "mld" not in dataset:
        raise ValueError("Input must include mixed-layer depth")
    units = dataset.mld.attrs.get("units")
    # Annual files from the original year sampler lost per-variable attributes in xarray.concat.
    legacy_annual = (
        units is None
        and dataset.attrs.get("weather_sampler") == "pure_lim"
        and dataset.mld.dims == ("calendar_month", "lat", "lon")
        and "model_id" in dataset
        and dataset.model_id.size == 12
    )
    if units != "m" and not legacy_annual:
        raise ValueError(
            "Expected mixed-layer depth in metres or an original annual LIM file"
        )
    with np.load(grid) as source:
        for coordinate in ("lat", "lon"):
            np.testing.assert_array_equal(
                source[coordinate], dataset[coordinate].values
            )
        depths = xr.DataArray(
            source["depth_m"],
            dims=("lat", "lon"),
            coords={"lat": dataset.lat, "lon": dataset.lon},
        )
    mld = dataset.mld
    bottom = depths.astype(mld.dtype)
    exceeds = np.isfinite(bottom) & (mld > bottom)
    output = dataset.copy(deep=False)
    output["mld"] = (
        xr.where(exceeds, bottom, mld)
        .transpose(*mld.dims)
        .assign_attrs(dict(mld.attrs, units="m"))
    )
    output.attrs = dict(
        dataset.attrs,
        mld_ceiling="MPI-ESM1-2-LR native sea-floor depth",
        mld_ceiling_cells=int(exceeds.sum()),
        mld_ceiling_grid_sha256=sha(grid),
    )
    return output


def bound(source, destination, grid):
    if source.resolve() == destination.resolve():
        raise ValueError(
            "Use a distinct output file for the physically constrained sample"
        )
    if sha(grid) != read(PLAN)["bathymetry_sha256"]:
        raise ValueError("Expected the registered North Atlantic MPI bathymetry grid")
    with xr.open_dataset(source, decode_times=False) as ds:
        write(cap_mld(ds, grid), destination)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    methods = p.add_subparsers(dest="method", required=True)
    w = methods.add_parser(
        "weights",
        help="Convert the twelve frozen weights for the standard sampling command",
    )
    w.add_argument("--source", type=Path, required=True)
    w.add_argument("--output", type=Path, required=True)
    c = methods.add_parser(
        "ocean", help="Apply the sea-floor bound to a generated month or annual file"
    )
    c.add_argument("--input", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--grid", type=Path, default=GRID)
    args = p.parse_args()
    if args.method == "weights":
        export(args.source, args.output)
    else:
        bound(args.input, args.output, args.grid)
