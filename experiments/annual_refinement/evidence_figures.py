"""Render the training ablation and the all-month independent 2050 storm benchmark."""
import argparse
import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figures import BLUE, INK, ORANGE, TEAL, finish, style


def training(root, out):
    data = pd.read_csv(root / "ocean_rank_validation.csv")
    rich = pd.read_csv(root / "ocean_training_years.csv")
    ranks = [18, 32, 48, 72, 96, 128]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), layout="constrained")
    for ax, field, title, unit in zip(
        axes,
        ("sst", "mld", "gamma"),
        ("Sea surface temperature", "Mixed-layer depth", "Subsurface gradient"),
        ("K", "m", "K per 100 m"),
        strict=True,
    ):
        a = data[(data.field == field) & (data.warming == "all") & (data.region == "basin")]
        for cohort, years, color in (
            ("original", 15, BLUE),
            ("expanded", 30, TEAL),
            ("extended", 57, ORANGE),
        ):
            q = (
                a[(a.cohort == cohort) & (a.layout == ("sst" if field == "sst" else "joint"))]
                .set_index("rank")
                .sort_index()
            )
            if not q.empty:
                ax.plot(
                    q.index,
                    q.rmse,
                    marker="o",
                    ms=3.5,
                    color=color,
                    label=f"{years} training years",
                )
        if field != "sst":
            q = rich[
                (rich.cohort == "ocean57")
                & (rich.field == field)
                & (rich.basis == "rich_ocean_d070")
                & (rich.warming == "all")
                & (rich.region == "basin")
            ].sort_values("rank")
            ax.plot(
                q["rank"],
                q.rmse,
                marker="D",
                ms=3,
                color=INK,
                ls="--",
                label="57 years, depth-weighted",
            )
        ax.set_title(title)
        ax.set_xlabel("Retained EOFs")
        ax.set_xticks(ranks, ["18", "32", "48", "72", "96", "128"])
        ax.set_ylabel(f"Reconstruction RMSE ({unit})")
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        "What changes when the training record grows?\nThe same validation members, years and calendar months for every line",
        fontsize=14,
    )
    finish(fig, out / "ocean_training_ablation.png")


def scenario(root, out, tag):
    windows = pd.read_csv(root / "wind_windows_and_fourier.csv", dtype={"window": str})
    windows = windows[
        (windows.panel == "ssp245_year")
        & (windows.group == "all")
        & (windows.target == "total")
        & (windows.window == "15")
    ]
    storms = pd.read_csv(root / "annual_mpi2050_storms.csv", dtype={"month": str})
    storms = storms[(storms.month != "all") & (storms.protocol == "MPI background")].copy()
    storms["month"] = storms.month.astype(int)
    models = (
        ("published_v1", "v1_LIM", "Published LIM", BLUE, "--"),
        (tag, "candidate_LIM", "Research LIM", TEAL, "-"),
        ("selected_fourier", "candidate_Fourier", "Native Fourier, matched moments", ORANGE, "-."),
    )
    fig, ax = plt.subplots(2, 2, figsize=(11.8, 7.3), sharex=True, layout="constrained")
    for driver, label, caption, color, ls in models:
        q = windows[windows.model == label].sort_values("month")
        ax[0, 0].plot(q.month, q.ratio, color=color, label=caption, ls=ls, marker="o", ms=3)
        for chart, metric, column in (
            (ax[0, 1], "tc", "value"),
            (ax[1, 0], "displacement_48h_km", "error"),
            (ax[1, 1], "lmi", "error"),
        ):
            q = storms[(storms.driver == driver) & (storms.metric == metric)].sort_values("month")
            chart.plot(q.month, q[column], color=color, ls=ls, marker="o", ms=3)
    q = storms[(storms.driver == tag) & (storms.metric == "tc")].sort_values("month")
    (ref,) = ax[0, 1].plot(
        q.month, q.reference, color=INK, ls=":", marker="s", ms=3, label="MPI daily replay"
    )
    ax[0, 0].axhline(1, color=INK, lw=1, ls=":")
    ax[0, 0].set_title("15-day steering-wind variance")
    ax[0, 0].set_ylabel("Model / daily MPI")
    ax[0, 1].set_title("Storm formation")
    ax[0, 1].set_ylabel("TCs per 100 disturbances")
    ax[1, 0].set_title("48-hour displacement distribution")
    ax[1, 0].set_ylabel("Distance from MPI (km)")
    ax[1, 1].set_title("Lifetime maximum wind distribution")
    ax[1, 1].set_ylabel("Distance from MPI (m/s)")
    for a in ax.flat:
        a.set_xticks(range(1, 13), calendar.month_abbr[1:])
        a.tick_params(labelbottom=True, axis="x", labelsize=8)
        a.grid(axis="y", alpha=0.2)
    lines, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(
        [*lines, ref],
        [*labels, "MPI daily replay"],
        ncol=4,
        frameon=False,
        loc="outside lower center",
        fontsize=9,
    )
    fig.suptitle(
        "MPI-ESM1-2-LR, 2050 under SSP2-4.5\nEight held-out members • fixed MPI backgrounds • 1,536 disturbances per calendar month",
        fontsize=14,
    )
    finish(fig, out / "ssp245_calendar.png")


def track_maps(root, out, tag, panel="ssp245_year"):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from storm_archive import KEYS

    from tc_emulator.tc_adapter import geography

    if panel == "ssp245_year":
        prefix, driver, fourier = (
            "annual_mpi2050",
            "selected_fourier",
            "Native Fourier, matched moments",
        )
        heading = "Synthetic North Atlantic tracks in 2050 under SSP2-4.5"
        detail = "All eight MPI members, identical disturbances and observed monthly backgrounds"
        filename = "ssp245_annual_tracks.png"
    elif panel == "val":
        prefix, driver, fourier = (
            "seasonal_validation",
            "native_fourier",
            "Native Fourier, published moments",
        )
        heading = "Synthetic tracks during July, September and November"
        detail = "Seven validation members and six years, identical disturbances and observed monthly backgrounds"
        filename = "validation_seasonal_tracks.png"
    else:
        raise ValueError("Unknown storm panel")
    tracks = pd.read_parquet(root / (prefix + "_tracks_6h.parquet"))
    attempts = pd.read_parquet(
        root / (prefix + "_attempts.parquet"), columns=["model", "protocol", "tc"]
    )
    lon = np.linspace(260.0, 360.0, 350)
    lat = np.linspace(0.0, 60.0, 230)
    land = geography()[3](lon, lat).T >= 0.5
    groups = (
        ("mpi_daily", "MPI daily", "MPI daily replay"),
        (tag, "MPI background", "Research LIM"),
        (driver, "MPI background", fourier),
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(13.2, 4.3), sharex=True, sharey=True, layout="constrained"
    )
    norm = Normalize(1, 12) if panel == "ssp245_year" else Normalize(7, 11)
    cm = plt.get_cmap("viridis")
    for ax, (model, protocol, title) in zip(axes, groups, strict=True):
        chosen = tracks[(tracks.model == model) & (tracks.protocol == protocol)]
        lines = []
        colors = []
        for _, points in chosen.groupby(KEYS[2:]):
            points = points.sort_values("elapsed_h")
            xx = (points.lon_e.to_numpy() + 180) % 360 - 180
            lines.append(np.column_stack((xx, points.lat_n.to_numpy())))
            colors.append(cm(norm(points.month.iloc[0])))
        expected = attempts[(attempts.model == model) & (attempts.protocol == protocol)].tc.sum()
        if len(lines) != expected:
            raise ValueError("Plotted track count differs from the attempt ledger")
        ax.contourf(lon - 360, lat, land, levels=[0.5, 1.5], colors=["#e0e1de"])
        ax.contour(lon - 360, lat, land, levels=[0.5], colors=["#89908e"], linewidths=0.4)
        ax.add_collection(LineCollection(lines, colors=colors, linewidths=0.6, alpha=0.46))
        ax.set(
            xlim=(-100, 0),
            ylim=(0, 60),
            xlabel="Longitude (°E)",
            title=f"{title}\n{len(lines)} accepted TCs",
        )
        ax.set_aspect(1.15)
    axes[0].set_ylabel("Latitude (°N)")
    legend = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cm),
        ax=axes,
        orientation="horizontal",
        shrink=0.46,
        aspect=45,
        pad=0.07,
    )
    ticks = [1, 4, 7, 10, 12] if panel == "ssp245_year" else [7, 9, 11]
    legend.set_ticks(ticks, labels=[calendar.month_abbr[m] for m in ticks])
    legend.set_label("Starting month")
    fig.suptitle(heading + "\n" + detail, fontsize=14)
    finish(fig, out / filename)


def wind_direction(root, out, panel="fresh"):
    panel_label = {"val": "validation panel", "fresh": "new annual holdout"}[panel]
    table = pd.read_csv(root / f"leadlag_intervals_{panel}.csv")
    table = table[table.group == "all"]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.9), sharex=True, layout="constrained")
    for column, region in enumerate(("basin", "corridor")):
        for row, metric in enumerate(("all_components", "directional")):
            ax = axes[row, column]
            focus = table[
                (table.metric == metric) & table.dimension.str.startswith(region + "/")
            ].copy()
            focus["lag"] = focus.dimension.str.split("/").str[-1].astype(int)
            a = focus[focus.comparator == "candidate_Fourier"].sort_values("lag")
            ax.plot(a.lag, a.candidate, color=TEAL, marker="o", label="Research LIM")
            ax.plot(a.lag, a.baseline, color=ORANGE, marker="s", ls="--", label="Native Fourier")
            if metric == "all_components":
                b = focus[focus.comparator == "candidate_tuned_Fourier"].sort_values("lag")
                ax.plot(
                    b.lag,
                    b.baseline,
                    color=BLUE,
                    ls=":",
                    marker="^",
                    label="Training-calibrated Fourier",
                )
            ax.set_xticks([1, 3, 5, 10, 15])
            ax.set_ylabel(
                ("Full matrix" if row == 0 else "Time-antisymmetric part")
                + "\nRMS error (correlation units)"
            )
            ax.grid(axis="y", alpha=0.2)
            if row == 0:
                ax.set_title("Whole North Atlantic" if column == 0 else "Tropical cyclone corridor")
            if row == 1:
                ax.set_xlabel("Lag (days)")
    lines, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(lines, labels, ncol=3, loc="outside lower center", frameon=False)
    fig.suptitle(
        f"Which wind components tend to lead the others?\nFour components at 250 and 850 hPa • MPI {panel_label} • monthly means removed",
        fontsize=14,
    )
    finish(fig, out / "wind_direction.png")

    raw = pd.read_csv(root / f"leadlag_matrices_{panel}.csv")
    raw = raw[(raw.region == "corridor") & (raw.lag == 3)]
    order = ["u250", "v250", "u850", "v850"]
    matrices = []
    for name in ("MPI", "candidate_LIM", "candidate_Fourier"):
        cov = (
            raw[raw.driver == name]
            .pivot_table(index="future", columns="past", values="covariance")
            .loc[order, order]
            .to_numpy()
        )
        matrices.append((cov - cov.T) / 2)
    scale = max(np.max(np.abs(m)) for m in matrices)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.75), layout="constrained")
    for ax, m, title in zip(
        axes, matrices, ("MPI daily winds", "Research LIM", "Shared-spectrum Fourier"), strict=True
    ):
        im = ax.imshow(m, vmin=-scale, vmax=scale, cmap="RdBu_r")
        ax.set_xticks(range(4), order, rotation=30)
        ax.set_yticks(range(4), order)
        ax.set_xlabel("Preceding component")
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("Component three days later")
    fig.colorbar(im, ax=axes, label="Antisymmetric covariance (m²/s²)", shrink=0.8)
    fig.suptitle("Three-day directional wind covariance in the tropical corridor", fontsize=14)
    finish(fig, out / "directional_covariance.png")


def forecast_skill(root, out, panel="fresh"):
    frame = pd.read_csv(root / ("two_day_forecasts_" + panel + ".csv"))
    frame = frame[frame.group == "all"].copy()
    frame["lead"] = frame.dimension.astype(int)
    colors = {
        "native_fourier": "#a05025",
        "trained_fourier": "#526b9d",
        "persistence_mse": "#797578",
        "mean_mse": "#6a7951",
    }
    names = {
        "native_fourier": "Native Fourier",
        "trained_fourier": "Training-calibrated Fourier",
        "persistence_mse": "Persistence",
        "mean_mse": "Background wind mean",
    }
    fig, (raw, change) = plt.subplots(1, 2, figsize=(10.8, 4.6), layout="constrained")
    ref = frame[frame.comparator == "native_fourier"].sort_values("lead")
    raw.plot(ref.lead, ref.candidate, color=TEAL, marker="o", lw=2.4, label="Research LIM")
    for key in names:
        part = frame[frame.comparator == key].sort_values("lead")
        raw.plot(
            part.lead, part.baseline, "o-", color=colors[key], label=names[key], ms=3.5, lw=1.2
        )
    for key, shift in (("native_fourier", -0.22), ("trained_fourier", 0.22)):
        part = frame[frame.comparator == key].sort_values("lead")
        change.vlines(part.lead + shift, part.ci_low, part.ci_high, color=colors[key], lw=2)
        change.scatter(
            part.lead + shift, part.difference, color=colors[key], s=22, label=names[key]
        )
    change.axhline(0, color=INK, lw=0.9, ls="--")
    for ax in (raw, change):
        ax.set_xticks([1, 5, 10, 15])
        ax.set_xlabel("Forecast lead (days)")
        ax.grid(axis="y", alpha=0.18)
    raw.set_ylabel("Steering RMSE (m/s)")
    change.set_ylabel("LIM minus Fourier RMSE (m/s)")
    raw.set_title("Absolute errors on the same daily winds", fontsize=11)
    change.set_title("Paired difference and 95% interval", fontsize=11)
    raw.legend(frameon=False, fontsize=8)
    change.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Forecasts given two preceding MPI wind maps and the actual monthly background", fontsize=14
    )
    finish(fig, out / "conditional_forecasts.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    style()
    training(a.report / "data", a.report / "figures")
    scenario(a.report / "data", a.report / "figures", a.tag)
    track_maps(a.report / "data", a.report / "figures", a.tag)
    track_maps(a.report / "data", a.report / "figures", a.tag, "val")
    wind_direction(a.report / "data", a.report / "figures")
    forecast_skill(a.report / "data", a.report / "figures")
