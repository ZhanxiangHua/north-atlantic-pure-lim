"""Score matched driver differences against MPI, with a whole-member paired bootstrap."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from annual_lim.layout import CONFIG, read
from benchmark_mpi2050 import DAILY, DESIGN, FOURIER, LIM, MPI, PROTOCOLS, SEED, WIND_MOMENTS
from prepare_mpi2050 import WORK
from tc_emulator.common import atomic

SCALARS = ("steering_1d", "steering_5d", "steering_10d", "steering_15d", "zonal_correlation",
           "shear_q90_ms", "weak_shear_3d_pct", "tc_per_100", "landfall_per_100")
SHAPES = ("lmi", "displacement_48h_km")


def weights(n, rng, size):
    samples = rng.integers(n, size=(size, n))
    return np.vstack((np.ones(n), np.stack([(samples == j).sum(axis=1) for j in range(n)], axis=1)))


def summary(metric, protocol, comparator, kind, values, err_l, err_f, **extra):
    advantage = err_f - err_l
    lo, hi = np.quantile(advantage[1:], [0.025, 0.975])
    return dict(protocol=protocol, metric=metric, comparator=comparator, error=kind,
                reference=values[0], lim=values[1], fourier=values[2], lim_error=err_l[0],
                fourier_error=err_f[0], advantage=advantage[0], paired_low=lo, paired_high=hi, **extra)


def shape_distance(target, model, metric, members, boot):
    """Exact empirical Wasserstein-1 reweighted by the number of copies of each member."""
    a = [np.sort(target.loc[target.member.eq(m), metric].dropna().to_numpy()) for m in members]
    b = [np.sort(model.loc[model.member.eq(m), metric].dropna().to_numpy()) for m in members]
    edges = np.unique(np.concatenate(a + b))
    width = np.diff(edges)
    ca = np.stack([np.searchsorted(x, edges[:-1], side="right") for x in a])
    cb = np.stack([np.searchsorted(x, edges[:-1], side="right") for x in b])
    na, nb = np.array(list(map(len, a))), np.array(list(map(len, b)))
    result = []
    for w in np.array_split(boot, 32):
        da, db = w @ na, w @ nb
        if np.min(da) == 0 or np.min(db) == 0:
            raise ValueError("An empty storm distribution occurred in a resample")
        delta = (w @ ca) / da[:, None] - (w @ cb) / db[:, None]
        result.extend(abs(delta) @ width)
    ref, gen = np.concatenate(a), np.concatenate(b)
    np.testing.assert_allclose(result[0], wasserstein_distance(ref, gen), rtol=1e-10)
    return np.array(result), np.median(ref), np.median(gen)


def summarize(work):
    raw = pd.read_parquet(work / "attempts.parquet")
    weather = pd.read_csv(work / "weather.csv")
    protocols = list(PROTOCOLS)
    if (work / "attempts_moments.parquet").exists() or (work / "weather_moments.csv").exists():
        raw = pd.concat([raw, pd.read_parquet(work / "attempts_moments.parquet")], ignore_index=True)
        weather = pd.concat([weather, pd.read_csv(work / "weather_moments.csv")], ignore_index=True)
        protocols.append(WIND_MOMENTS)
    members = read(CONFIG / "members.json")["test"]
    keys = ["member", "draw", "origin", "seed_id"]
    ref = raw[raw.method.eq(MPI)].sort_values(keys)
    expected_n = len(members)*len(DESIGN["origins"])*DESIGN["track_draws"]*DESIGN["seeds_per_panel"]
    if len(ref) != expected_n or ref.duplicated(keys).any():
        raise ValueError("Duplicate or missing reference seed panels")
    records = []
    for (protocol, method), group in raw.groupby(["protocol", "method"]):
        g = group.sort_values(keys)
        for col in keys + ["lon", "lat", "filter_draw", "v_init"]:
            np.testing.assert_array_equal(ref[col], g[col])
        if (g.landfall & ~g.tc).any():
            raise ValueError("TC landfall classification mismatch")
        records.append(dict(protocol=protocol, method=method, attempts=len(g), tc=int(g.tc.sum()),
                            landfalls=int(g.landfall.sum()), complete_48h=int(g.displacement_48h_km.count())))
    if len(records) != 2*len(protocols)+2:
        raise ValueError("Incomplete set of track drivers")
    counts_weather = weather.groupby(["protocol", "method", "member"]).size()
    if len(counts_weather) != len(members)*(1+2*len(protocols)) or any(
            n != (1 if method == MPI else DESIGN["weather_draws"]) for (_, method, _), n in counts_weather.items()):
        raise ValueError("Incomplete weather realizations")
    rates = raw.groupby(["protocol", "method", "member"])[["tc", "landfall"]].mean().mul(100)
    rates.columns = ["tc_per_100", "landfall_per_100"]
    weather = weather.drop(columns="draw").groupby(["protocol", "method", "member"]).mean()
    metrics = rates.join(weather)
    boot = weights(len(members), np.random.default_rng(SEED), DESIGN["member_resamples"])
    summary_rows = []
    for protocol in protocols:
        base = raw[raw.protocol.eq(protocol) & raw.method.eq(LIM) & raw.tc]
        for comparator in ([FOURIER, DAILY] if protocol == PROTOCOLS[0] else [FOURIER]):
            other = raw[raw.protocol.eq(protocol) & raw.method.eq(comparator) & raw.tc]
            for metric in SCALARS:
                if comparator == DAILY and metric not in ("tc_per_100", "landfall_per_100"):
                    continue
                triplet = [metrics.loc[(p, m), metric].reindex(members).to_numpy()
                           for p, m in [("Reference", MPI), (protocol, LIM), (protocol, comparator)]]
                a, l, f = (boot @ group / len(members) for group in triplet)
                row = summary(metric, protocol, comparator, "absolute error of the ensemble mean",
                              [a[0], l[0], f[0]], abs(l-a), abs(f-a))
                row["mean_memberwise_advantage"] = np.mean(abs(triplet[2]-triplet[0])-abs(triplet[1]-triplet[0]))
                summary_rows.append(row)
            for metric in SHAPES:
                err_l, a, l = shape_distance(ref[ref.tc], base, metric, members, boot)
                err_f, _, f = shape_distance(ref[ref.tc], other, metric, members, boot)
                summary_rows.append(summary(metric, protocol, comparator, "Wasserstein-1 (reported levels are medians)",
                                            [a, l, f], err_l, err_f))
    return pd.DataFrame(summary_rows), pd.DataFrame(records), metrics.reset_index()


def gain_scores(work):
    cases = pd.read_csv(work / "gain_validation.csv")
    members, years = sorted(cases.member.unique()), sorted(cases.year.unique())
    rng = np.random.default_rng(SEED)
    count = DESIGN["member_resamples"]
    wm, wy = weights(len(members), rng, count), weights(len(years), rng, count)
    out = []
    for metric in ["shear_variance", "daily_change", "shear_q90_ms", "weak_shear_1d_pct", "weak_shear_3d_pct"]:
        def averages(method):
            table = cases[cases.method.eq(method)].pivot(index="member", columns="year", values=metric).loc[members, years]
            return np.einsum("bi,ij,bj->b", wm, table.to_numpy(), wy)/(len(members)*len(years))
        r, lim = map(averages, [MPI, LIM])
        for candidate in ("Shear gain", "Shear gain and lag"):
            gain = averages(candidate)
            diff = abs(lim-r) - abs(gain-r)
            lo, hi = np.quantile(diff[1:], [0.025, 0.975])
            out.append(dict(metric=metric, candidate=candidate, reference=r[0], pure_lim=lim[0], value=gain[0],
                            advantage=diff[0], paired_low=lo, paired_high=hi))
    return pd.DataFrame(out)


def figure(frame, path):
    matched = frame[(frame.protocol == PROTOCOLS[0]) & (frame.comparator == FOURIER)].set_index("metric")
    days = [1, 5, 10, 15]
    window = matched.loc[[f"steering_{d}d" for d in days]]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.7), layout="constrained")
    palette = {"reference": "#333333", "lim": "#007780", "fourier": "#ad4f16"}
    for k, name in zip(palette, (MPI, LIM, FOURIER)):
        ax[0, 0].plot(days, window[k], marker="o", lw=2, label=name, color=palette[k])
    ax[0, 0].set(title="Within-month steering variance", xlabel="Averaging period (days)", ylabel="Variance (m²/s²)", xticks=days)
    ax[0, 0].legend(frameon=False)
    rates = frame[(frame.metric == "tc_per_100") & (frame.comparator == FOURIER)].set_index("protocol").loc[list(PROTOCOLS)]
    x = np.arange(3)
    ax[0, 1].axhline(rates.reference.iloc[0], color=palette["reference"], linestyle="--", label=MPI)
    for k, offset, name in zip(["lim", "fourier"], [-0.08, 0.08], [LIM, FOURIER]):
        ax[0, 1].scatter(x+offset, rates[k], color=palette[k], s=48, label=name)
    labels = ["MPI monthly\nbackground", "MPI warming\nonly", "FaIR warming\nonly"]
    ax[0, 1].set(title="TC formation in 15-day FAST runs", ylabel="TCs per 100 proposed seeds",
                 xticks=x, xticklabels=labels, xlim=(-.4, 2.4))
    ax[0, 1].legend(frameon=False, fontsize=8)
    for a, metric, name, units in zip(ax[1], ["steering_15d", "tc_per_100"],
             ["15-day steering", "TC formation"], ["m²/s²", "TCs per 100 seeds"]):
        subset = frame[(frame.metric == metric) & (frame.comparator == FOURIER)].set_index("protocol").loc[list(PROTOCOLS)]
        for i, r in enumerate(subset.itertuples()):
            a.plot([r.paired_low, r.paired_high], [i, i], color="#515865", lw=2)
            a.plot(r.advantage, i, "o", color=palette["lim"])
        a.axvline(0, color="#787878", ls="--", lw=1)
        a.set(yticks=x, yticklabels=["MPI background", "MPI warming", "FaIR warming"], ylim=(2.55, -.55),
              title=f"Paired error difference for {name}", xlabel=f"Fourier error minus LIM error ({units})")
    for a in ax.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("North Atlantic · SSP2-4.5 · September 2050", fontsize=15)
    fig.savefig(path, dpi=165)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--output", type=Path, default=WORK)
    args = p.parse_args()
    frame, counts, by_member = summarize(args.work)
    tables = [("scores.csv", frame), ("counts.csv", counts), ("member_metrics.csv", by_member)]
    if (args.work / "gain_validation.csv").is_file():
        tables.append(("gain_scores.csv", gain_scores(args.work)))
    for name, table in tables:
        with atomic(args.output / name) as dest:
            table.to_csv(dest, index=False)
    with atomic(args.output / "mpi2050.png") as dest:
        figure(frame, dest)


if __name__ == "__main__":
    main()
