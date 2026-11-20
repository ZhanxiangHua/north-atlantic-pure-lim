"""Plot pooled displacement, the paired year contrasts and all accepted storm tracks."""
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

from annual_lim.layout import read
from midcentury_sources import PLAN, REPORT
from storm_archive import KEYS
from tc_emulator.tc_adapter import geography

COLORS = ["#222831", "#4969ab", "#008673", "#b2662b"]
LABELS = ["MPI daily", "Published v1 LIM", "Research LIM", "Matched Fourier"]


def main_drivers(p):
    return [p["reference_driver"], p["baseline"], p["candidate"], "selected_fourier"]


def plot_skill(attempts, changes, p):
    selection = attempts[attempts.protocol.isin(["MPI daily", "MPI background"]) & attempts.tc
                         & attempts.displacement_48h_km.notna()]
    fig, (ax, forest) = plt.subplots(1, 2, figsize=(11.8, 4.9), layout="constrained",
                                     gridspec_kw={"width_ratios": [1.05, 1.]})
    for tag, label, color in zip(main_drivers(p), LABELS, COLORS):
        x = np.sort(selection.loc[selection.model == tag, "displacement_48h_km"])
        ax.step(np.r_[0, x], np.arange(len(x) + 1) / len(x), where="post",
                label=f"{label} (n={len(x)})", color=color, lw=1.85)
    ax.set(xlabel="48-hour endpoint separation (km)", ylabel="Fraction of eligible storms",
           title="Six new years pooled", ylim=(0, 1.02), xlim=(0, None))
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    years = sorted([*p["new_years"], p["reference_year"]])
    order = [str(y) for y in years] + ["new_six", "all_seven"]
    data = changes[(changes.protocol == "MPI background") & (changes.comparator == p["baseline"])
                   & (changes.metric == "displacement_48h_km")].set_index("scope").loc[order]
    forest.axvline(0, c="#464b51", lw=.9)
    for i, (scope, row) in enumerate(data.iterrows()):
        color = "#7b6b57" if scope == str(p["reference_year"]) else COLORS[2]
        forest.hlines(i, row.ci_low, row.ci_high, color=color, linewidth=1.5)
        forest.plot(row.change, i, marker="D" if scope in ("new_six", "all_seven") else "o",
                    markersize=7 if scope == "new_six" else 5, color=color)
    forest.axhline(len(years) - .5, color="#c4c8ca", lw=.9)
    forest.set(yticks=np.arange(len(order)), yticklabels=[str(y) + (" (prior)" if y == p["reference_year"]
               else "") for y in years] + ["Six new years", "All seven"], xlabel="Research minus v1 W1 error (km)",
               title="Change in 48-hour error, paired 95% intervals")
    forest.invert_yaxis()
    for a in (ax, forest):
        a.grid(axis="x", alpha=.17)
        a.spines[["top", "right"]].set_visible(False)
    return fig


def plot_tracks(tracks, counts, p):
    fixed = tracks[tracks.protocol.isin(["MPI daily", "MPI background"])]
    totals = counts[(counts.year.isin(p["new_years"])) & counts.protocol.isin(["MPI daily", "MPI background"])].groupby("model").tc.sum()
    _, _, _, land, _ = geography()
    lon, lat = np.linspace(-100, 0, 401), np.linspace(0, 60, 241)
    coast = land(lon + 360, lat).T
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.7), sharex=True, sharey=True, layout="constrained")
    for ax, tag, label, color in zip(axes.flat, main_drivers(p), LABELS, COLORS):
        lines = []
        selected = fixed[fixed.model == tag].sort_values([*KEYS, "elapsed_h"])
        for _, group in selected.groupby(KEYS, sort=False):
            coords = group[["lon_e", "lat_n"]].to_numpy().copy()
            coords[:, 0] = (coords[:, 0] + 180) % 360 - 180
            lines.append(coords)
        if len(lines) != totals[tag]:
            raise ValueError(f"Track count mismatch for {tag}")
        ax.contourf(lon, lat, coast, levels=[.5, 1.1], colors=["#e6e6df"])
        ax.add_collection(LineCollection(lines, colors=[color], linewidths=.55, alpha=.14))
        ax.contour(lon, lat, coast, levels=[.5], colors="#495057", linewidths=.43)
        ax.set(xlim=(-100, 0), ylim=(0, 60), title=f"{label}  ·  {len(lines)} storms",
               xticks=[-100, -75, -50, -25, 0], yticks=[0, 20, 40, 60])
        ax.set_aspect(1 / np.cos(np.deg2rad(30)))
    for ax in axes[-1]:
        ax.set_xlabel("Longitude (°E)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude (°N)")
    fig.suptitle("All accepted tracks across six new SSP2-4.5 years", fontsize=14)
    return fig


if __name__ == "__main__":
    plan = read(PLAN)
    inputs = pd.read_parquet(REPORT / "attempts.parquet")
    differences = pd.read_csv(REPORT / "contrasts.csv", dtype={"scope": str})
    figures = [(plot_skill(inputs, differences, plan), "displacement.png"),
               (plot_tracks(pd.read_parquet(REPORT / "tracks_6h.parquet"), pd.read_csv(REPORT / "counts.csv"),
                            plan), "tracks.png")]
    for fig, name in figures:
        fig.savefig(REPORT / name, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
