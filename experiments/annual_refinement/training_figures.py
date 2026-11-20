"""Draw the front-weighting and seasonal-variance experiments; public tables suffice for rendering."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from ocean_fronts import edges, gradient_percentile

from annual_lim.layout import DEFAULT_WORK
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic


def prepare_map(data):
    from ocean import ocean_data

    model = MonthlyModel(DEFAULT_WORK / "weights", 2)
    fields, meta = ocean_data("extended", 2, "train")
    mld = fields[:, 1]
    wet = model.rep.blocks["ocean"].valid[0]
    clim = np.nansum(mld, axis=0, dtype=float) / np.maximum(np.isfinite(mld).sum(axis=0), 1)
    pct = gradient_percentile(clim, wet, edges(model.rep.lat, model.rep.lon))
    with atomic(data / "february_ocean_fronts.npz") as tmp:
        np.savez_compressed(
            tmp,
            lat=model.rep.lat,
            lon=model.rep.lon,
            wet=wet,
            climatology_m=np.where(wet, clim, np.nan),
            gradient_percentile=pct,
            training_years=meta.year.nunique(),
            training_members=meta.member.nunique(),
        )


def front_figure(report):
    data = report / "data"
    methods = pd.read_csv(data / "ocean_front_validation_intervals.csv")
    fig, (maps, points) = plt.subplots(
        1, 2, figsize=(11.1, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1.14, 1]}
    )
    with np.load(data / "february_ocean_fronts.npz") as f:
        lon = ((f["lon"] + 180) % 360) - 180
        levels = [0, 20, 40, 70, 100, 150, 250, 400, 600, 900, 1500]
        colors = plt.get_cmap("Blues", len(levels))
        im = maps.pcolormesh(
            lon,
            f["lat"],
            f["climatology_m"],
            shading="auto",
            cmap=colors,
            norm=BoundaryNorm(levels, colors.N, extend="max"),
            rasterized=True,
        )
        maps.set_facecolor("#e1e4e4")
        front = np.ma.array(f["gradient_percentile"] >= 0.8, mask=~f["wet"])
        maps.contour(lon, f["lat"], front, levels=[0.5], colors=["#c15324"], linewidths=0.8)
    maps.set(xlim=(-100, 0), ylim=(0, 60), xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    maps.set_title("February mixed-layer depth", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=maps, location="bottom", shrink=0.86, pad=0.1, aspect=38)
    cb.set_label("57-year training mean (m)")
    maps.legend(
        [Line2D([0], [0], c="#c15324", lw=1.6)],
        ["Strongest 20% of local gradients"],
        loc="lower left",
        fontsize=8,
        framealpha=0.96,
    )
    colors = {"sst": "#007b75", "mld": "#9a5327", "gamma": "#335c9b"}
    labels = {
        "sst": "Sea surface temperature",
        "mld": "Mixed-layer depth",
        "gamma": "Subsurface temperature gradient",
    }
    keys = ["basin_rmse", "front_rmse", "gradient_rmse_per_100km"]
    for j, field in enumerate(colors):
        rows = (
            methods[(methods.group == "all") & (methods.dimension == field + "/1.0")]
            .set_index("metric")
            .loc[keys]
        )
        estimate = rows.change_percent.to_numpy()
        bounds = np.stack([estimate - rows.pct_ci_low, rows.pct_ci_high - estimate])
        y = np.array([2.0, 1.0, 0.0]) + (0.19 - 0.19 * j)
        points.errorbar(
            estimate,
            y,
            xerr=bounds,
            fmt="o",
            capsize=2.7,
            lw=1.7,
            ms=5,
            c=colors[field],
            label=labels[field],
        )
    points.axvline(0, color="#68787c", ls="--", lw=1)
    points.set_yticks([2, 1, 0], ["Whole basin", "Strong fronts", "Neighboring-cell\ngradient"])
    points.set(xlim=(-5.5, 2), ylim=(-0.5, 2.7), xlabel="Change in validation RMSE (%)")
    points.set_title("Effect of front weighting at fixed rank", loc="left", fontweight="bold")
    points.grid(axis="x", color="#dde3e5", lw=0.7)
    points.legend(loc="upper left", bbox_to_anchor=(0.01, -0.20), frameon=False, fontsize=8.2)
    fig.savefig(report / "figures/ocean_front_weighting.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def slope_figure(report):
    data = pd.read_csv(report / "data/seasonal_covariance_training_cv.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, layout="constrained")
    keys = ("constant", "separate_months", "cyclic_1", "annual_slope")
    names = ("Constant variance", "Monthly slopes", "Smoothed monthly slopes", "One annual slope")
    colors = ("#566671", "#cf773a", "#06777a", "#4b63a2")
    for ax, target, title in zip(
        axes, ("all_background", "ocean"), ("All background fields", "Ocean fields only")
    ):
        for i, (key, name, color) in enumerate(zip(keys, names, colors)):
            for j, (ny, nm) in enumerate(((12, 14), (28, 28))):
                rows = data[
                    (data.target == target)
                    & (data.method == key)
                    & (data.fit_years == ny)
                    & (data.fit_members == nm)
                ]
                ordered = rows.set_index("regime").loc[["cold", "inside", "warm"]]
                x = np.arange(3) + j * 3.8 + (i - 1.5) * 0.15
                ax.scatter(
                    x,
                    ordered.abs_log_error,
                    label=name if j == 0 else "",
                    color=color,
                    s=28,
                    zorder=4,
                )
        ax.set_xticks([0, 1, 2, 3.8, 4.8, 5.8], ["Cold", "Inside", "Warm"] * 2, fontsize=9)
        ax.axvline(2.9, color="#d5dcdf", lw=1)
        ax.text(1, 0.157, "12 years\n14 members", ha="center", va="top", fontsize=9)
        ax.text(4.8, 0.157, "28 years\n28 members", ha="center", va="top", fontsize=9)
        ax.set_ylim(0.143, 0.222)
        ax.grid(axis="y", lw=0.7, color="#e0e7e9")
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].set_ylabel("Mean absolute log variance error")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside upper center",
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.savefig(report / "figures/seasonal_warming_slopes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def warm_confirmation_figure(report):
    results = pd.read_csv(report / "data/warm_confirmation_pass15.csv")
    budgets = pd.read_csv(report / "data/warm_confirmation_variance_budget.csv")
    full = np.arange(1, 13)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.3, 5.9),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.25, 1]},
    )
    temporal, bg = axes
    chosen = results[results.driver == "primary_LIM"].set_index("month")
    for m, row in chosen.iterrows():
        temporal.fill_between(
            [m - 0.44, m + 0.44],
            row.band_low,
            row.band_high,
            color="#dce6e9",
            label="Training acceptance band" if m == 1 else None,
        )
    entries = (
        ("primary_LIM", "Research LIM", "#007d77", "o"),
        ("gain_LIM", "Constant monthly gain", "#b25225", "s"),
        ("primary_calibrated_Fourier", "Calibrated Fourier", "#53538a", "^"),
        ("primary_native_Fourier", "Native Fourier", "#62727c", "x"),
    )
    for method, name, color, mark in entries:
        arr = results[results.driver == method].set_index("month").ratio.reindex(full)
        temporal.plot(full, arr, marker=mark, ms=4, lw=1.5, c=color, label=name)
    temporal.set_ylabel("15-day variance / MPI")
    temporal.set_title("Five new warm years under SSP5-8.5", loc="left", fontweight="bold")
    temporal.legend(ncol=3, loc="center", bbox_to_anchor=(0.5, 0.40), fontsize=8, frameon=False)
    for i, (col, label, color) in enumerate(
        (
            ("primary_transmitted", "Research LIM", "#007d77"),
            ("gain_transmitted", "Monthly gain", "#b25225"),
        )
    ):
        ratios = budgets.assign(ratio=budgets[col] / budgets.mpi_background)
        grouped = ratios.groupby("month").ratio
        x = grouped.mean().index.to_numpy() + (-0.1 if i == 0 else 0.1)
        quant = grouped.quantile([0, 0.5, 1]).unstack()
        bg.vlines(x, quant[0], quant[1], color=color, alpha=0.65, lw=1.6)
        bg.scatter(
            x, quant[0.5], color=color, marker=("o" if i == 0 else "s"), s=24, label=label, zorder=3
        )
    bg.set_ylabel("Background-induced\nvariance / MPI backgrounds")
    bg.set_title("Median and range across the five years", loc="left", fontsize=10)
    for ax in axes:
        ax.axhline(1, color="#3f4c52", ls="--", lw=1)
        ax.axvspan(8.55, 9.45, color="#eee", zorder=-1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0.5, 12.5)
    bg.set_xticks(
        full, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep*", "Oct", "Nov", "Dec"]
    )
    bg.set_xlabel(
        "* September was excluded before the new MPI fields were opened.", loc="left", fontsize=8
    )
    fig.savefig(report / "figures/warm_confirmation_15day.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prepare-map", action="store_true")
    args = parser.parse_args()
    if args.prepare_map:
        prepare_map(args.report / "data")
    args.report.joinpath("figures").mkdir(parents=True, exist_ok=True)
    front_figure(args.report)
    slope_figure(args.report)
    if (args.report / "data/warm_confirmation_pass15.csv").exists():
        warm_confirmation_figure(args.report)
