"""Plot the complexity trade-off and physical MLD cap from the public paired score tables."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw(report):
    data = report / "data"
    dev = pd.read_csv(data / "partial_wind_memory_summary.csv").sort_values("past_rank")
    fresh = pd.read_csv(data / "compact_ocean_confirmation_paired.csv")
    checked = fresh[(fresh.group == "all") & (fresh.metric == "day_one_rmse_ms")].iloc[
        0
    ]
    bath = pd.read_csv(data / "bathymetric_mld_paired.csv")
    bath = bath[(bath.group == "all") & (bath.metric == "q99_loss_m")].copy()
    bath[["region", "decoder"]] = bath.dimension.str.split("/", n=1, expand=True)
    bath = bath[bath.decoder == "sea_floor"].set_index("region")
    new = fresh[(fresh.group == "all") & (fresh.metric == "q99_loss_m")].set_index(
        "dimension"
    )
    fig, (ax, bx) = plt.subplots(
        1,
        2,
        figsize=(11.7, 4.65),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.1, 1]},
    )
    green, blue = "#006a5d", "#385b96"
    ax.axhline(0, color="0.55", lw=0.8)
    ax.axhline(0.025, color="0.45", ls=":", label="Day-one tolerance (0.025 m/s)")
    ax.errorbar(
        dev.past_rank,
        dev.difference,
        yerr=[dev.difference - dev.ci_low, dev.ci_high - dev.difference],
        color=green,
        fmt="o-",
        capsize=3,
        label="Development, 12 months",
    )
    ax.errorbar(
        [72],
        [checked.difference],
        yerr=[
            [checked.difference - checked.ci_low],
            [checked.ci_high - checked.difference],
        ],
        color=blue,
        marker="D",
        markersize=7,
        markerfacecolor="white",
        capsize=6,
        label="Six new climates, 11 months",
    )
    ax.set_xticks(dev.past_rank)
    ax.set(
        xlabel="Previous-day directions kept alongside 120 current-day modes",
        ylabel="Day-one RMSE increase relative to full LIM (m/s)",
    )
    ax.set_title("How much wind history helps?", loc="left", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    regions = ["shelf_at_most_500m", "whole_basin", "subpolar_Atlantic"]
    y = np.arange(3)
    bx.axvline(0, color="0.55", lw=0.8)
    for score, offset, color, marker, label in [
        (bath, -0.12, green, "o", "Development, 12 months"),
        (new, 0.12, blue, "D", "Six new climates, 11 months"),
    ]:
        p = score.loc[regions]
        bx.errorbar(
            p.difference,
            y + offset,
            xerr=[p.difference - p.ci_low, p.ci_high - p.difference],
            fmt=marker,
            color=color,
            capsize=4,
            markerfacecolor="white" if marker == "D" else color,
            label=label,
        )
    bx.set_yticks(y, ["Shelf (≤500 m)", "Whole basin", "Subpolar Atlantic"])
    bx.set(
        ylim=(-0.62, 2.65),
        xlabel="Change in 99th-quantile loss after sea-floor clipping (m)",
    )
    bx.set_title(
        "What does the physical sea floor change?", loc="left", fontweight="bold"
    )
    bx.legend(loc="best", fontsize=8, frameon=False)
    for panel in (ax, bx):
        panel.spines[["top", "right"]].set_visible(False)
        panel.tick_params(labelsize=9)
    out = report / "figures/memory_and_ocean_floor.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, default=Path("reports/annual_refinement"))
    draw(p.parse_args().report)
