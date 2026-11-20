"""Compare matched FAST attempts and track distributions with two crossed bootstrap axes."""
import argparse

import numpy as np
import pandas as pd
from paired_intervals import panel_weights
from records import WORK, record
from scipy.stats import wasserstein_distance

from tc_emulator.common import atomic

SHAPES = {"lmi": "m/s", "displacement_48h_km": "km"}
RATES = ("tc", "landfall")


def read_driver(tag, panel):
    root, expected = ("storm_validation", 126) if panel == "val" else ("storm_ssp245_year", 96)
    paths = sorted((WORK / root / tag).glob("*/*/*/attempts.parquet"))
    if len(paths) != expected:
        raise ValueError(f"{tag} has {len(paths)} of {expected} paired climate cases")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def rates(frame, metric, grid, weights):
    avg = frame.groupby(["member", "year"])[metric].mean().reindex(grid)
    if avg.isna().any():
        raise ValueError("Missing member-year")
    return 100 * (weights @ avg.to_numpy())


def empirical_tables(vals, grid, support):
    table = np.zeros((len(grid), len(support) - 1), dtype=float)
    totals = np.zeros(len(grid))
    for i, case in enumerate(grid):
        x = np.sort(vals.get(case, np.array([])))
        totals[i] = len(x)
        table[i] = np.searchsorted(x, support[:-1], side="right")
    return table, totals


def distance(ref, pred, key, grid, weights):
    samples = []
    for d in (ref, pred):
        selected = d[d.tc & np.isfinite(d[key])]
        samples.append({a: g[key].to_numpy() for a, g in selected.groupby(["member", "year"])})
    if any(not sample for sample in samples):
        # Distributional distance is undefined if either climate produces no storms.
        return np.full(len(weights), np.nan)
    support = np.unique(np.concatenate([a for tab in samples for a in tab.values()]))
    if len(support) == 1:
        counts = [weights @ np.array([len(s.get(case, [])) for case in grid]) for s in samples]
        return np.where((counts[0] > 0) & (counts[1] > 0), 0.0, np.nan)
    terms = [empirical_tables(v, grid, support) for v in samples]
    out = []
    for block in np.array_split(weights, max(1, int(np.ceil(len(weights) / 128)))):
        with np.errstate(invalid="ignore", divide="ignore"):
            cdf = [(block @ t) / (block @ n)[:, None] for t, n in terms]
            out.extend(np.abs(cdf[0] - cdf[1]) @ np.diff(support))
    out = np.asarray(out)
    direct = wasserstein_distance(
        np.concatenate(list(samples[0].values())), np.concatenate(list(samples[1].values()))
    )
    np.testing.assert_allclose(out[0], direct, rtol=1.0e-8, atol=1.0e-9)
    return out


def limits(samples):
    finite = samples[1:][np.isfinite(samples[1:])]
    if finite.size:
        return dict(
            ci_low=np.quantile(finite, 0.025),
            ci_high=np.quantile(finite, 0.975),
            valid_draws=len(finite),
        )
    return dict(ci_low=np.nan, ci_high=np.nan, valid_draws=0)


def score(candidate, draws, panel="val"):
    ref = read_driver("mpi_daily", panel)
    controls = ["published_v1", "native_fourier"]
    if panel == "ssp245_year":
        controls.append("selected_fourier")
    mod = {tag: read_driver(tag, panel) for tag in [*controls, candidate]}
    rows = []
    for season in ["all", *sorted(ref.month.unique())]:
        rr = ref if season == "all" else ref[ref.month == season]
        grid = rr.groupby(["member", "year"], sort=True).g.first()
        members, years = map(
            np.asarray, (grid.index.get_level_values(0), grid.index.get_level_values(1))
        )
        W, _, _ = panel_weights(members, years, grid.to_numpy(), "all", draws, 840792)
        keys = ["member", "year", "month", "draw", "origin", "seed_id"]
        aligned = rr.sort_values(keys).reset_index(drop=True)
        for protocol in ("MPI background", "warming only"):
            drivers = {}
            for tag, frame in mod.items():
                # Native Fourier was matched only on fixed MPI backgrounds in this experiment.
                f = frame[frame.protocol == protocol]
                if season != "all":
                    f = f[f.month == season]
                if f.empty:
                    continue
                f = f.sort_values(keys).reset_index(drop=True)
                np.testing.assert_array_equal(
                    aligned[keys + ["lon", "lat", "v_init", "filter_draw"]].to_numpy(),
                    f[keys + ["lon", "lat", "v_init", "filter_draw"]].to_numpy(),
                )
                drivers[tag] = f
            for metric in (*RATES, *SHAPES):
                err = {}
                for tag, dd in drivers.items():
                    obs = val = None
                    if metric in RATES:
                        obs = rates(rr, metric, grid.index, W)
                        val = rates(dd, metric, grid.index, W)
                        error = np.abs(val - obs)
                        unit = "per 100 attempted seeds"
                    else:
                        error = distance(rr, dd, metric, grid.index, W)
                        unit = SHAPES[metric]
                    err[tag] = error
                    rows.append(
                        dict(
                            month=season,
                            protocol=protocol,
                            metric=metric,
                            driver=tag,
                            reference=obs[0] if obs is not None else np.nan,
                            value=val[0] if val is not None else np.nan,
                            error=error[0],
                            **limits(error),
                            unit=unit,
                            attempts=len(dd),
                            tc=int(dd.tc.sum()),
                            ref_tc=int(rr.tc.sum()),
                        )
                    )
                for base in controls:
                    if base not in err:
                        continue
                    gain = err[base] - err[candidate]
                    rows.append(
                        dict(
                            month=season,
                            protocol=protocol,
                            metric=metric,
                            driver=candidate + "_gain_vs_" + base,
                            error=gain[0],
                            **limits(gain),
                            unit=unit,
                        )
                    )
    dest = WORK / "comparison" / candidate / panel / "storms.csv"
    with atomic(dest) as tmp:
        pd.DataFrame(rows).to_csv(tmp, index=False)
    record(
        "storm_summary",
        panel=panel,
        tag=candidate,
        records=len(ref),
        members=rr.member.nunique(),
        years=rr.year.nunique(),
        draws=draws,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--draws", default=1500, type=int)
    p.add_argument("--panel", choices=["val", "ssp245_year"], default="val")
    a = p.parse_args()
    score(a.candidate, a.draws, a.panel)
