"""Pair two September 2050 checkpoints and their Fourier controls on identical MPI members and seeds."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from records import WORK, record

from annual_lim.layout import CONFIG, read
from tc_emulator.common import ROOT, atomic

sys.path.insert(0, str(ROOT / "notebooks"))
from benchmark_mpi2050 import DESIGN, FOURIER, LIM, MPI, PROTOCOLS, SEED
from score_mpi2050 import SCALARS, SHAPES, shape_distance, weights


def track_figure(tag, new, old, output):
    groups = [
        (old, "Reference", MPI, "MPI daily replay"),
        (old, PROTOCOLS[0], LIM, "Published LIM"),
        (new, PROTOCOLS[0], LIM, "Research LIM"),
        (old, PROTOCOLS[0], FOURIER, "Native Fourier (fixed)"),
    ]
    refmeta = read(CONFIG / "members.json")["test"]
    # Use every accepted storm on the four panels of the first two prespecified members. Lines are entire six-hour tracks.
    use = refmeta[:2]
    fig, axes = plt.subplots(
        2, 2, figsize=(10.2, 7.2), sharex=True, sharey=True, layout="constrained"
    )
    from tc_emulator.tc_adapter import geography

    lon = np.linspace(260, 360, 350)
    lat = np.linspace(0, 60, 200)
    land = geography()[3](lon, lat).T >= 0.5
    for ax, (frame, protocol, method, title) in zip(axes.flat, groups):
        chosen = frame[
            (frame.protocol == protocol) & (frame.method == method) & frame.member.isin(use)
        ]
        ax.contourf(lon - 360, lat, land, levels=[0.5, 1.5], colors=["#dedfdb"])
        ax.contour(lon - 360, lat, land, levels=[0.5], colors=["#8f9894"], linewidths=0.4)
        n = 0
        for _, g in chosen.groupby(["member", "draw", "origin", "seed_id"]):
            g = g.sort_values("elapsed_h")
            n += 1
            ax.plot(
                g.lon_e - 360,
                g.lat_n,
                lw=0.8,
                color="#107d83" if method == LIM else "#54677b",
                alpha=0.6,
            )
        ax.set_xlim(-100, 0)
        ax.set_ylim(0, 58)
        ax.set_title(f"{title}  ·  {n} tracks", fontsize=11)
        ax.set_aspect(1.15)
        ax.grid(alpha=0.13)
    for ax in axes[1]:
        ax.set_xlabel("Longitude (°E)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude (°N)")
    fig.suptitle(
        "September 2050 • SSP2-4.5 • MPI monthly ocean and atmospheric background\n"
        + "All 6-hour tracks from "
        + ", ".join(use)
        + " • same proposed storms and origins",
        fontsize=14,
    )
    fig.savefig(output / "ssp245_tracks.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def compare():
    receipt = read(WORK / "FROZEN.json")
    tag = receipt["stage"]
    old = ROOT / "reports/mpi2050"
    new = WORK / "mpi2050" / tag
    summary = [
        pd.read_csv(r / "member_metrics.csv").set_index(["protocol", "method", "member"])
        for r in (old, new)
    ]
    raw = [pd.read_parquet(r / "attempts.parquet") for r in (old, new)]
    reference = [
        f[f.method == MPI]
        .sort_values(["member", "draw", "origin", "seed_id"])
        .reset_index(drop=True)
        for f in raw
    ]
    keys = [
        "member",
        "draw",
        "origin",
        "seed_id",
        "lon",
        "lat",
        "v_init",
        "filter_draw",
        "tc",
        "landfall",
    ]
    for key in keys:
        np.testing.assert_array_equal(reference[0][key], reference[1][key])
    np.testing.assert_allclose(
        reference[0].displacement_48h_km,
        reference[1].displacement_48h_km,
        rtol=2.0e-6,
        atol=0.01,
        equal_nan=True,
    )
    members = read(CONFIG / "members.json")["test"]
    boot = weights(len(members), np.random.default_rng(SEED), DESIGN["member_resamples"])
    rows = []
    for protocol in PROTOCOLS:
        for metric in (*SCALARS, *SHAPES):
            errors = {}
            values = {}
            ref_level = None
            for label, j, method in (
                ("Published LIM", 0, LIM),
                ("Research LIM", 1, LIM),
                ("Fixed native Fourier", 0, FOURIER),
                ("Matched native Fourier", 1, FOURIER),
            ):
                if metric in SCALARS:
                    target = summary[j].loc[("Reference", MPI), metric].reindex(members).to_numpy()
                    pred = summary[j].loc[(protocol, method), metric].reindex(members).to_numpy()
                    obs = boot @ target / len(members)
                    sim = boot @ pred / len(members)
                    error = abs(obs - sim)
                    ref_level = obs[0]
                    values[label] = sim[0]
                else:
                    model = raw[j][
                        (raw[j].protocol == protocol) & (raw[j].method == method) & raw[j].tc
                    ]
                    error, ref_level, values[label] = shape_distance(
                        reference[j][reference[j].tc], model, metric, members, boot
                    )
                errors[label] = error
            for label, e in errors.items():
                gain = e - errors["Research LIM"]
                low, high = np.quantile(gain[1:], [0.025, 0.975])
                rows.append(
                    dict(
                        protocol=protocol,
                        metric=metric,
                        driver=label,
                        MPI=ref_level,
                        model_level=values[label],
                        error=e[0],
                        error_minus_research=gain[0],
                        ci_low=low,
                        ci_high=high,
                        loss="Wasserstein-1"
                        if metric in SHAPES
                        else "absolute error of ensemble mean",
                    )
                )
    out = WORK / "comparison" / tag / "ssp245"
    out.mkdir(parents=True, exist_ok=True)
    with atomic(out / "paired.csv") as tmp:
        pd.DataFrame(rows).to_csv(tmp, index=False)
    track_figure(
        tag,
        pd.read_parquet(new / "tc_tracks_6h.parquet"),
        pd.read_parquet(old / "tc_tracks_6h.parquet"),
        out,
    )
    record(
        "ssp245_frozen_comparison",
        tag=tag,
        members=len(members),
        draws=DESIGN["weather_draws"],
        rows=len(rows),
    )


if __name__ == "__main__":
    compare()
