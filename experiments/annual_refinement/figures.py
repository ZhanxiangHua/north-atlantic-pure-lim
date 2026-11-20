"""Render shareable scientific figures from the recorded paired comparisons."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from records import WORK

from annual_lim.layout import CONFIG, DEFAULT_WORK, case_dir, design, read
from annual_lim.model import MonthlyModel

TEAL = "#087c82"
BLUE = "#376ca0"
ORANGE = "#ac5a27"
INK = "#29333b"
LABELS = {
    "val": "Validation • 7 members × 6 years",
    "test": "Original holdout • 8 members × 15 years",
    "fresh": "New holdout • 8 members × 9 years",
}


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#abb8be",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "savefig.dpi": 170,
            "savefig.facecolor": "white",
        }
    )


def finish(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def ocean_scores(tag, role, out):
    tab = pd.read_csv(WORK / "comparison" / tag / role / "paired.csv")
    t = tab[(tab.group == "all") & tab.dimension.str.startswith("basin/", na=False)]
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 4.5), layout="constrained")
    titles = ["Sea surface temperature", "Mixed-layer depth", "Subsurface temperature gradient"]
    for ax, field, title, unit in zip(
        axes, ("sst", "mld", "gamma"), titles, ("K", "m", "K / 100 m")
    ):
        s = t[t.dimension == "basin/" + field].set_index("metric")
        keys = ["projection_rmse", "crps", "mean_rmse"]
        a = s.loc[keys]
        x = np.arange(3)
        ax.bar(x - 0.18, a.baseline, 0.34, color=BLUE, label="Published LIM")
        ax.bar(x + 0.18, a.candidate, 0.34, color=TEAL, label="Research LIM")
        for i, row in enumerate(a.itertuples()):
            ax.text(
                i + 0.18,
                row.candidate + max(a.baseline.max(), a.candidate.max()) * 0.027,
                f"{row.candidate:.2f}",
                ha="center",
                fontsize=8,
                color=TEAL,
            )
        ax.set_xticks(x, ["Map\nRMSE", "Marginal\nCRPS", "Ensemble mean\nRMSE"])
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(a.baseline.max(), a.candidate.max()) * 1.18)
        ax.yaxis.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Ocean reconstruction and probabilistic prediction\n"
        + LABELS[role]
        + " • North Atlantic basin",
        fontsize=16,
    )
    finish(fig, out / "ocean_scores.png")


def wind_windows(tag, role, out):
    f = pd.read_csv(
        WORK / "comparison" / tag / role / "weather_controls.csv", dtype={"window": str}
    )
    f = f[f.group == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), layout="constrained")
    for ax, target, title in zip(
        axes, ("total", "within"), ("Total steering variance", "Within-month steering variance")
    ):
        order = ("1", "5", "10", "15", "month") if target == "total" else ("1", "5", "10", "15")
        for method, name, c, marker in [
            ("v1_LIM", "Published LIM", BLUE, "s"),
            ("candidate_LIM", "Research LIM", TEAL, "o"),
            ("candidate_Fourier", "Matched native Fourier", ORANGE, "^"),
        ]:
            seq = f[(f.target == target) & (f.model == method)].groupby("window").ratio
            med = seq.mean().reindex(order)
            low = seq.quantile(0.25).reindex(order)
            high = seq.quantile(0.75).reindex(order)
            xx = np.arange(len(order))
            ax.fill_between(xx, low, high, color=c, alpha=0.10)
            ax.plot(xx, med, "-", color=c, marker=marker, linewidth=2, label=name)
        ax.axhline(1, color=INK, ls="--", linewidth=0.9)
        ax.set_xticks(np.arange(len(order)), [x if x != "month" else "Full month" for x in order])
        ax.set_title(title)
        ax.set_xlabel("Averaging window (days)")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Model / MPI daily variance")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Steering variability from 1 day to a full calendar month\n"
        + LABELS[role]
        + " • shading spans the middle six calendar months",
        fontsize=15,
    )
    finish(fig, out / "wind_windows.png")


def regimes(tag, role, out):
    p = pd.read_csv(WORK / "comparison" / tag / role / "paired.csv")
    cols = [
        ("projection_rmse", "basin/mld", "Mixed-layer map RMSE (m)"),
        ("monthly_wind_rmse", "", "Conditional monthly wind RMSE (m/s)"),
        ("variance_log_error", "total/15", "15-day total variance log error"),
        ("variance_log_error", "within/15", "15-day within-month log error"),
    ]
    p["dimension"] = p.dimension.fillna("")
    fig, axs = plt.subplots(2, 2, figsize=(10.9, 6.8), layout="constrained")
    for ax, (metric, dim, title) in zip(axs.flat, cols):
        data = (
            p[(p.metric == metric) & (p.dimension == dim)]
            .set_index("group")
            .loc[["cold", "inside", "warm"]]
        )
        for k, (_, r) in enumerate(data.iterrows()):
            ax.plot([r.ci_low, r.ci_high], [k, k], lw=2, color=BLUE)
            ax.plot(r.difference, k, "o", color=TEAL, markersize=6)
        ax.axvline(0, color=INK, ls="--", lw=0.9)
        ax.set_title(title, fontsize=11)
        ax.set_yticks(range(3), ["Colder than training", "Inside training", "Warmer than training"])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel("Research LIM minus published LIM")
    fig.suptitle(
        "Changes by warming range\n"
        + LABELS[role]
        + " • 95% paired member–year bootstrap intervals",
        fontsize=15,
    )
    finish(fig, out / "warming_ranges.png")


def attribution(tag, role, out):
    sources = [("published_v1", "Published LIM", BLUE), (tag, "Research LIM", TEAL)]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), layout="constrained")
    for name, caption, color in sources:
        p = WORK / "attribution" / name / role
        df = pd.concat(pd.read_csv(f) for f in p.glob("*_years.csv"))
        fix = df[df.point == "fixed_temperature"]
        empirical = df[df.point == "member_years"]
        for ax, field, measured in zip(
            axes,
            ("predicted_background", "predicted_monthly_weather"),
            ("conditional_mean_on_mpi", "remaining_mpi_variance"),
            strict=True,
        ):
            seq = fix.groupby("g")[field]
            mean = seq.mean()
            ax.plot(mean.index, mean, "-", color=color, label=caption)
            ax.fill_between(
                mean.index, seq.quantile(0.25), seq.quantile(0.75), color=color, alpha=0.11
            )
            points = empirical.groupby("g")[measured].mean()
            ax.scatter(
                points.index,
                points,
                marker="x" if name == "published_v1" else "+",
                s=36,
                color=color,
                linewidths=1.4,
                zorder=4,
            )
    for ax, title in zip(
        axes,
        (
            "Background-induced monthly steering variance",
            "Averaged daily weather and MPI wind residuals",
        ),
    ):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Global warming relative to 1850–1900 (K)")
        ax.set_ylabel("Monthly steering variance (m²/s²)")
        for limit in (-0.011522062, 3.419637285):
            ax.axvline(limit, ls=":", color=INK, lw=0.8)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "How warming reaches the monthly wind\n"
        "Curves and bands show the calendar-month mean and middle six months. "
        "Crosses use actual MPI fields at each held-out warming.\n"
        "Left crosses use monthly wind predicted from MPI backgrounds. Right crosses use the remaining MPI wind variance.",
        fontsize=12,
    )
    finish(fig, out / "background_transfer.png")


def ocean_maps(tag, weights, role, out):
    cfg = WORK / "configs/fresh" if role == "fresh" else CONFIG
    rr = "test" if role == "fresh" else role
    cases = [c for c in design(cfg)[1] if c[0] == rr]
    # Always choose the last calendar year and the first designated member. This choice precedes evaluation.
    yr = max(c[2] for c in cases)
    _, member, _, g = next(c for c in cases if c[2] == yr)
    m = 9
    with np.load(case_dir(WORK / "data", member, yr) / "09.npz") as f:
        observed = {k: f[k] for k in ("sst", "mld", "gamma", "psl", "ta", "hus")}
        lat, lon = f["lat"], f["lon"]
    decoded = []
    for src in (DEFAULT_WORK / "weights", weights):
        model = MonthlyModel(src, m)
        fields = model.rep.decode(model.rep.encode(observed)[None])
        decoded.append(
            {k: np.maximum(fields[k][0], 0.0 if k == "gamma" else 0.01) for k in ("mld", "gamma")}
        )
    xx, yy = np.meshgrid(lon - 360, lat)
    basin = (yy >= 0) & (yy <= 60) & (xx >= -100) & (xx <= 0)
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 6.2), sharex=True, sharey=True, layout="constrained"
    )
    for i, (key, name, unit) in enumerate(
        (("mld", "Mixed-layer depth", "m"), ("gamma", "Subsurface gradient", "K / 100 m"))
    ):
        truth = observed[key]
        values = np.where(basin, truth, np.nan)
        absmax = np.nanpercentile(values, 98)
        im = axes[i, 0].pcolormesh(
            xx, yy, values, cmap="viridis", vmin=0, vmax=absmax, shading="nearest", rasterized=True
        )
        fig.colorbar(im, ax=axes[i, 0], label=unit, shrink=0.8)
        residual = [np.where(basin, a[key] - truth, np.nan) for a in decoded]
        limit = np.nanpercentile(np.abs(np.stack(residual)), 97)
        for j, e in enumerate(residual, 1):
            field = axes[i, j].pcolormesh(
                xx,
                yy,
                e,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                shading="nearest",
                rasterized=True,
            )
        fig.colorbar(
            field, ax=axes[i, 1:].tolist(), label="Model minus MPI (" + unit + ")", shrink=0.8
        )
        axes[i, 0].set_ylabel(name + "\nLatitude (°N)")
    for ax, title in zip(
        axes[0], ("MPI field", "Published reconstruction error", "Research reconstruction error")
    ):
        ax.set_title(title, fontsize=11)
    for ax in axes.flat:
        ax.set_facecolor("#e6e9e7")
        ax.set_xlim(-100, 0)
        ax.set_ylim(0, 60)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    for ax in axes[-1]:
        ax.set_xlabel("Longitude (°E)")
    fig.suptitle(
        f"Reconstructing one held-out September ocean state\nMPI-ESM1-2-LR • {member} • {yr} • warming {g:.2f} K",
        fontsize=15,
    )
    finish(fig, out / "example_ocean_maps.png")


def covariance_workflow(out):
    """Connect the matrix formula to one steering component with a two-score example."""
    fig = plt.figure(figsize=(9.4, 7.5), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[2.8, 1])
    flow = fig.add_subplot(grid[0])
    flow.set(xlim=(0, 1), ylim=(0, 1))
    flow.axis("off")
    steps = [
        (
            "Input",
            "Choose a calendar month and warming level $g$",
            "The saved model supplies the mean background scores $m(g)$ and their covariance $C(g)$.",
        ),
        (
            "Background",
            r"Draw scores $z=m(g)+F(g)e$, with $C(g)=F(g)F(g)^\mathsf{T}$",
            "Each entry of $e$ is an independent standard normal draw. Diagonal entries of $C$ give\n"
            "individual score variances. An off-diagonal entry gives the covariance of two scores.",
        ),
        (
            "Monthly wind",
            r"Map the score anomalies into wind     $\Delta u=k^\mathsf{T}[z-m(g)]$",
            "For one wind component, each entry of $k$ is its fitted response to a one-unit change\n"
            "in a background score. The forced wind mean is fixed at a given warming.",
        ),
        (
            "Output",
            r"Compute its variance     $\mathrm{Var}(\Delta u\mid g)=k^\mathsf{T}C(g)k$",
            "This measures the part of monthly wind variance supplied by the background. The monthly\n"
            "average of daily LIM weather contributes a second, separately measured variance.",
        ),
    ]
    for i, (label, equation, explanation) in enumerate(steps):
        y = 0.93 - 0.25 * i
        flow.text(0.005, y, label, color=TEAL, weight="bold", va="top", fontsize=11)
        flow.text(0.19, y, equation, va="top", fontsize=12.4, weight="medium")
        flow.text(0.19, y - 0.077, explanation, va="top", fontsize=10.4)
        if i < len(steps) - 1:
            flow.annotate(
                "",
                xy=(0.052, y - 0.223),
                xytext=(0.052, y - 0.078),
                arrowprops={"arrowstyle": "-|>", "color": "#91a8ae", "linewidth": 1},
            )
    fig.suptitle(
        "How a change in background covariance reaches the monthly wind",
        x=0.025,
        ha="left",
        fontsize=17,
        weight="medium",
    )
    ax = fig.add_subplot(grid[1])
    wind = np.array([0.6, 0.3])
    covs = np.array([[[1, 0.2], [0.2, 1]], [[1.4, 0.3], [0.3, 1.2]]])
    parts = np.column_stack(
        (
            wind[0] ** 2 * covs[:, 0, 0],
            wind[1] ** 2 * covs[:, 1, 1],
            2 * wind[0] * wind[1] * covs[:, 0, 1],
        )
    )
    totals = wind @ covs @ wind
    np.testing.assert_allclose(parts.sum(axis=1), totals)
    left = np.zeros(len(covs))
    for col, label, color in zip(
        parts.T,
        ("First score", "Second score", "Covariance term"),
        (BLUE, TEAL, ORANGE),
        strict=True,
    ):
        ax.barh([1, 0], col, left=left, height=0.45, label=label, color=color)
        for y, x, v in zip((1, 0), left, col, strict=True):
            ax.text(x + v / 2, y, f"{v:.3f}", color="white", va="center", ha="center", fontsize=9)
        left += col
    for y, total in zip((1, 0), totals, strict=True):
        ax.text(total + 0.008, y, f"{total:.3f}", va="center", weight="bold")
    ax.set(
        yticks=[1, 0],
        yticklabels=["Cooler", "Warmer"],
        xlim=(0, 0.8),
        xlabel=r"Background contribution to monthly wind variance ($\mathrm{m^2\,s^{-2}}$)",
    )
    ax.set_title(
        "Two-score teaching example, with fixed wind responses of 0.6 and 0.3 m/s",
        loc="left",
        fontsize=11.5,
        pad=8,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=3, frameon=False)
    finish(fig, out / "covariance_to_wind.png")


def run(tag, weights, role, out):
    if role != "val":
        receipt = read(WORK / "FROZEN.json")
        if tag != receipt["stage"]:
            raise ValueError("Use the frozen model for independent figures")
    style()
    covariance_workflow(out)
    ocean_scores(tag, role, out)
    wind_windows(tag, role, out)
    regimes(tag, role, out)
    attribution(tag, role, out)
    ocean_maps(tag, weights, role, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--role", choices=["val", "test", "fresh"], default="val")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.tag, a.weights, a.role, a.output)
