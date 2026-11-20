"""Check if MPI's own bathymetry improves the upper tail of physical mixed-layer depth."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from deep_mixed_layer import exceedance, quantile
from marine_regions import BOXES
from ocean import ocean_data
from paired_intervals import panel_weights, record_estimate
from records import WORK, record
from spatial_check import field_loc

from annual_lim.layout import CONFIG, read
from annual_lim.model import MonthlyModel
from annual_lim.prepare import rows_for
from tc_emulator.assets import array, attr
from tc_emulator.common import ROOT, atomic, save_json, sha
from tc_emulator.physical import ocean_geometry, remap


PUBLIC = ROOT / "reports/annual_refinement/data"
HERE = WORK / "bathymetric_mld"
FILE = "deptho_Ofx_MPI-ESM1-2-LR_piControl_r1i1p1f1_gn.nc"
SOURCE = (
    Path(
        "/glade/collections/cmip/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/piControl/r1i1p1f1/Ofx/deptho/gn/v20190710/deptho"
    )
    / FILE
)


def prepare(path, output, historical):
    """Map native MPI wet vertices using the same geometry and coordinate precision as MLD."""
    model = MonthlyModel(read(WORK / "FROZEN.json")["weights"], 1)
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(path) as source, h5py.File(historical) as reference:
        assert (
            attr(source, "source_id") == "MPI-ESM1-2-LR"
            and attr(source["deptho"], "units") == "m"
        )
        lat, lon = array(source["latitude"]), array(source["longitude"])
        np.testing.assert_allclose(lat, array(reference["latitude"]), rtol=0, atol=1e-5)
        np.testing.assert_allclose(
            lon, array(reference["longitude"]), rtol=0, atol=1e-5
        )
        depth = array(source["deptho"])
        depth = np.where(depth > 0, depth, np.nan)
        geometry = ocean_geometry(lat, lon, model.rep.lat, model.rep.lon)
        mapped = remap(depth, geometry)
        meta = dict(
            source_filename=path.name,
            source_sha256=sha(path),
            tracking_id=attr(source, "tracking_id"),
            license=attr(source, "license"),
            institution=attr(source, "institution_id"),
            historical_reference=historical.name,
            field="deptho",
            units="m",
            table="Ofx",
            grid="gn",
            version="v20190710",
            fixed_experiment="piControl",
            valid_map_points=int(np.isfinite(mapped).sum()),
            geometry="Same wet-vertex normalized Delaunay remapping as MPI mlotst.",
            coordinate_match_tolerance_deg=1e-5,
        )
    out = output / "mpi_bathymetry_grid.npz"
    with atomic(out) as temp:
        np.savez_compressed(temp, lat=model.rep.lat, lon=model.rep.lon, depth_m=mapped)
    meta["mapped_sha256"] = sha(out)
    save_json(output / "mpi_bathymetry_source.json", meta)
    record(
        "mpi_native_bathymetry_mapped",
        source_sha256=meta["source_sha256"],
        map_sha256=meta["mapped_sha256"],
    )


def marine_masks(latitude, longitude, depth):
    """Exclude the preparation halo from each reported marine region."""

    def in_region(name):
        south, north, west, east = BOXES[name]
        return (
            (latitude >= south)
            & (latitude <= north)
            & (longitude >= west)
            & (longitude <= east)
        )

    basin = in_region("whole_basin")
    return {
        "whole_basin": basin,
        "shelf_at_most_500m": basin & (depth <= 500),
        "subpolar_Atlantic": in_region("subpolar_Atlantic"),
    }


def run(month, data):
    model = MonthlyModel(read(WORK / "FROZEN.json")["weights"], month)
    path = data / "mpi_bathymetry_grid.npz"
    assert sha(path) == read(data / "mpi_bathymetry_source.json")["mapped_sha256"]
    with np.load(path) as grid:
        np.testing.assert_array_equal(grid["lat"], model.rep.lat)
        np.testing.assert_array_equal(grid["lon"], model.rep.lon)
        depth = grid["depth_m"].ravel()
    sea = model.rep.blocks["ocean"].valid[0].ravel()
    use = sea & np.isfinite(depth) & (depth > 0)
    bottom = depth[use]
    lat, lon = np.meshgrid(model.rep.lat, model.rep.lon, indexing="ij")
    lat, lon = lat.ravel()[use], lon.ravel()[use]
    regions = marine_masks(lat, lon, bottom)
    train, tm = ocean_data("extended", month, "train")
    held, meta = ocean_data("original", month, "val")
    if (
        tm.year.nunique() != 57
        or tm.member.nunique() != 35
        or meta.year.nunique() != 6
        or set(meta.member) & set(tm.member)
    ):
        raise ValueError("Unexpected MLD evaluation panels")
    training = train[:, 1].reshape(len(train), -1)[:, use]
    maximum = np.nanmax(training, axis=0)
    if not np.isfinite(maximum).all():
        raise ValueError("A scored cell has no training maximum")
    truth = held[:, 1].reshape(len(meta), -1)[:, use]
    output = []
    for year, members in meta.groupby("year"):
        g = float(members.g.iloc[0])
        loc, basis, mask = field_loc(
            model, model.bg.mean(np.array([g]))[0], model.factor(g), "mld"
        )
        np.testing.assert_array_equal(mask, sea)
        loc, sd = loc[use], np.linalg.norm(basis[use], axis=1)
        obs = truth[members.index]
        for method, cap in (
            ("research_log", np.inf),
            ("sea_floor", bottom),
            ("training_max", maximum),
        ):
            q99 = quantile(loc, sd, 0.99, 1.0, cap)
            qdiff = obs - q99
            below = exceedance(loc, sd, bottom, 1.0, cap)
            for region, selected in regions.items():
                weights = np.isfinite(obs) * selected * np.cos(np.deg2rad(lat))
                if np.any(weights.sum(axis=1) <= 0):
                    raise ValueError("No scored marine cells in the requested region")
                weights = weights / weights.sum(axis=1, keepdims=True)

                def average(x):
                    return np.sum(weights * np.where(np.isfinite(x), x, 0.0), axis=1)

                rows = members[["member", "year", "g"]].assign(
                    month=month,
                    method=method,
                    region=region,
                    scored_cells=selected.sum(),
                    common_wet_fraction=use.sum() / sea.sum(),
                    q99_loss_m=average(np.maximum(0.99 * qdiff, -0.01 * qdiff)),
                    q99_exceedance=average(obs > q99),
                    model_below_floor=average(below),
                    actual_below_floor=average(obs > bottom),
                    actual_over_by_1m=average(obs > bottom + 1.0),
                )
                for cut in (1000, 2000, 3000):
                    p = exceedance(loc, sd, cut, 1.0, cap)
                    event = obs > cut
                    rows[f"expected_over_{cut}m"] = average(p)
                    rows[f"observed_over_{cut}m"] = average(event)
                    rows[f"brier_{cut}m"] = average((p - event) ** 2)
                output.append(rows)
    with atomic(HERE / f"{month:02}.parquet") as temp:
        pd.concat(output, ignore_index=True).to_parquet(
            temp, index=False, compression="zstd"
        )
    record(
        "bathymetric_mld_validation_month",
        month=month,
        members=meta.member.nunique(),
        mapped_wet_fraction=float(use.sum() / sea.sum()),
        shelf_cells=int(regions["shelf_at_most_500m"].sum()),
    )


def summarize(out):
    raw = pd.concat(
        [pd.read_parquet(HERE / f"{m:02}.parquet") for m in range(1, 13)],
        ignore_index=True,
    )
    cols = [
        c
        for c in raw
        if c.startswith(
            ("q99_", "model_", "actual_", "expected_", "observed_", "brier_")
        )
    ]
    summary = raw.groupby(["region", "method"], as_index=False)[
        ["scored_cells", "common_wet_fraction"] + cols
    ].mean()
    # Match against the earlier independently scored lognormal and training-cap controls.
    old = pd.read_csv(out / "deep_mixed_layer_validation.csv")
    for row in summary[
        summary.region.isin(["whole_basin", "subpolar_Atlantic"])
        & summary.method.isin(["training_max", "research_log"])
    ].itertuples():
        if row.common_wet_fraction == 1.0:
            prior = old[
                (old.region == row.region)
                & (old.model == row.method)
                & (old.diagnostic == "quantile")
                & (old.level == 0.99)
            ]
            np.testing.assert_allclose(row.q99_loss_m, prior.score.item(), rtol=2e-6)
    intervals = []
    for region, part in raw.groupby("region"):
        groups = (
            part.groupby(["member", "year", "g", "method"])[
                ["q99_loss_m", "brier_2000m", "brier_3000m"]
            ]
            .mean()
            .unstack("method")
        )
        member, year, g = [
            groups.index.get_level_values(i).to_numpy() for i in range(3)
        ]
        for ii, band in enumerate(("all", "cold", "inside", "warm")):
            wt, nm, ny = panel_weights(member, year, g, band, 2000, 20261422 + ii)
            for metric in ("q99_loss_m", "brier_2000m", "brier_3000m"):
                for method in ("sea_floor", "training_max"):
                    record_estimate(
                        intervals,
                        band,
                        metric,
                        [region, method],
                        wt @ groups[metric, "research_log"].to_numpy(),
                        wt @ groups[metric, method].to_numpy(),
                        nm,
                        ny,
                        "raw",
                    )
    raw.to_parquet(
        out / "bathymetric_mld_cases.parquet", index=False, compression="zstd"
    )
    summary.to_csv(out / "bathymetric_mld_validation.csv", index=False)
    pd.DataFrame(intervals).to_csv(out / "bathymetric_mld_paired.csv", index=False)
    record(
        "bathymetric_mld_validation_summary",
        member_months=504,
        case_scores=len(raw),
        regions=3,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--map", action="store_true")
    mode.add_argument("--month", type=int, choices=range(1, 13))
    mode.add_argument("--summary", action="store_true")
    p.add_argument("--data", type=Path, default=PUBLIC)
    p.add_argument("--deptho", type=Path, default=SOURCE)
    p.add_argument("--historical-mlotst", type=Path)
    args = p.parse_args()
    if args.map:
        ref = args.historical_mlotst or Path(
            rows_for(CONFIG, "r1i1p1f1", 2000)["mlotst"].local
        )
        prepare(args.deptho, args.data, ref)
    elif args.month:
        run(args.month, args.data)
    else:
        summarize(args.data)
