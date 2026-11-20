"""Evaluate the frozen September wind drivers against MPI daily weather and FAST replay."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from annual_lim.layout import CONFIG, read, rng_for
from annual_lim.sample import event
from prepare_mpi2050 import MONTH, REPORT, WORK, YEAR
from tc_benchmark import aligned_window, fourier_inputs
from tc_emulator.common import ROOT, atomic, sha
from tc_emulator.diagnostics import region_area, steering
from tc_emulator.fourier_weather import NativeFourierWind, WindStatistics, phase_seed
from tc_emulator.tc_adapter import DAY, geography, panel, replay_event, tracks

DESIGN = read(REPORT / "design.json")
SEED = DESIGN["seed"]
PROTOCOLS = ("MPI background", "MPI warming only", "FaIR warming only")
WIND_MOMENTS = "MPI monthly wind moments"
LIM, FOURIER, DAILY, MPI = "Pure LIM", "Native Fourier", "Daily Fourier", "MPI daily"


@lru_cache(maxsize=8)
def reference(work, member):
    path = work / f"reference/{member}.nc"
    if sha(path) != read(path.with_suffix(".verified.json"))["output_sha256"]:
        raise ValueError(f"Changed input for {member}")
    with xr.open_dataset(path, decode_times=False) as ds:
        return ds.load()


def corridor_weights(lat, lon):
    c = DESIGN["corridor"]
    sea = geography()[3](lon, lat).T < 0.5
    keep = ((lat[:, None] >= c["south"]) & (lat[:, None] <= c["north"]) &
            (lon[None, :] >= c["west_degrees_east"]) & (lon[None, :] <= c["east_degrees_east"]) & sea)
    w = np.cos(np.deg2rad(lat))[:, None] * keep
    return w / w.sum()


def weather_metrics(wind, area, corridor, lon):
    """Within-month centering removes each grid point's monthly wind in all three drivers."""
    if wind.shape[:2] != (30, 4) or not np.isfinite(wind).all():
        raise ValueError("Expected all 30 daily wind fields")
    s = steering(wind).astype(float)
    s -= s.mean(axis=0)
    cumulative = np.concatenate([np.zeros_like(s[:1]), s.cumsum(axis=0)])
    result = {}
    for d in (1, 5, 10, 15):
        avg = (cumulative[d:] - cumulative[:-d]) / d
        result[f"steering_{d}d"] = float(np.sum(np.square(avg).mean(axis=(0, 1)) * area))
    # Correlate local anomalies at two ocean points on a parallel, pooling time and components.
    spacing = np.diff(lon)
    step = int(round(7.5 / spacing[0]))
    np.testing.assert_allclose(spacing * step, 7.5, atol=1e-5)
    pairs = corridor[:, :-step] * (corridor[:, step:] > 0)
    left, right = s[..., :-step], s[..., step:]
    dot = lambda x, y: np.sum((x * y).mean(axis=(0, 1)) * pairs)
    result["zonal_correlation"] = float(dot(left, right) / np.sqrt(dot(left, left) * dot(right, right)))
    shear = np.hypot(wind[:, 0] - wind[:, 2], wind[:, 1] - wind[:, 3])
    wet = corridor > 0
    vals = shear[:, wet].ravel()
    weights = np.broadcast_to(corridor[wet], (len(shear), wet.sum())).ravel()
    result["shear_q90_ms"] = float(np.quantile(vals, 0.90, weights=weights, method="inverted_cdf"))
    low = shear < 8
    result["weak_shear_3d_pct"] = float(100 * np.sum((low[:-2] & low[1:-1] & low[2:]).mean(axis=0) * corridor))
    return result


def sampled_window(ds, origin, days):
    # Day 1 at noon is t=0 in a sampled month, and daily padding starts at day 0.
    ds = ds.isel(t_seconds=slice(origin - 1, origin + days + 2))
    seconds = ds.t_seconds.values - (origin - 1) * DAY
    np.testing.assert_array_equal(seconds, np.arange(-1, days + 2) * DAY)
    return ds.assign_coords(t_seconds=("t_seconds", seconds, {"units": "s"}))


def storm_records(env, seeds, provider, protocol, method, member, draw, origin):
    trace, outcomes = tracks(env, seeds, DESIGN["event_days"], wind_provider=provider)
    sampled = trace.isel(elapsed_s=slice(None, None, 6))
    hours = sampled.elapsed_s.values / 3600
    peaks, distances = np.full(len(seeds), np.nan), np.full(len(seeds), np.nan)
    for i in np.flatnonzero(outcomes.tc):
        speed = sampled.vmax.values[i]
        segment = aligned_window(sampled.storm_lon.values[i], sampled.storm_lat.values[i], speed, hours, np.isfinite(speed))
        if segment is not None:
            _, peaks[i], distances[i] = segment
    return outcomes[["seed_id", "lon", "lat", "filter_draw", "v_init", "tc", "landfall", "lmi", "reason"]].assign(
        member=member, draw=draw, origin=origin, protocol=protocol, method=method,
        peak_48h_ms=peaks, displacement_48h_km=distances)


def local_moments(wind, source, target):
    """Map conditional LIM anomalies to the pointwise mean and covariance used by native FAST."""
    # M L_source = L_target. Solve the transposed equation at each grid point.
    M = np.linalg.solve(source.lower.swapaxes(-2, -1), target.lower.swapaxes(-2, -1)).swapaxes(-2, -1)
    shifted = np.einsum("yxij,tjyx->tiyx", M, wind-source.mean, optimize=True)
    return np.where(target.valid, shifted + target.mean, 0.).astype("float32")


def simulate(task):
    work, weights, member, draw, mpi_g, fair_g, target_moments = task
    ref = reference(work, member)
    model, cov = fourier_inputs(weights)
    lon, lat = ref.lon.values, ref.lat.values
    area, corridor = region_area(lat, lon), corridor_weights(lat, lon)
    ident = f"{member}/{draw}"
    origins = DESIGN["origins"] if draw < DESIGN["track_draws"] else []
    panels = {o: panel(int(rng_for(SEED, "seeds", ident, o).integers(2**32)), DESIGN["seeds_per_panel"]) for o in origins}
    met, records = [], []
    if draw == 0 and not target_moments:
        met.append(dict(protocol="Reference", method=MPI, member=member, draw=draw,
                        **weather_metrics(ref.wind.values, area, corridor, lon)))
    if not target_moments:
        for o in origins:
            real = replay_event(ref, DESIGN["event_days"], origin_day=o)
            records.append(storm_records(real, panels[o], None, "Reference", MPI, member, draw, o))
    for protocol in ((WIND_MOMENTS,) if target_moments else PROTOCOLS):
        observed = ref if protocol in (PROTOCOLS[0], WIND_MOMENTS) else None
        g = fair_g if protocol == PROTOCOLS[2] else mpi_g
        # The same background normals, LIM innovations, Fourier phases and seeds are reused
        # across warming protocols. Observed backgrounds use the encoder's conditional mean.
        if origins:
            env = event(weights, MONTH, YEAR, g, 30, SEED, ident, draw, observed=observed)
            wind = env.wind.values[1:31]
            mean_pc = env.conditional_wind_pc.values
        else:
            z = model.rep.encode(observed) if observed is not None else model.draw_z(g, YEAR, SEED, ident)
            pc = model.pc_path(g, z, 33, YEAR, SEED, ident, draw)
            wind = model.rep.blocks["wind"].inverse(pc[1:31])
            mean_pc = model.wind_mean(g, z)
        mean = model.rep.blocks["wind"].inverse(mean_pc[None])[0]
        stats = WindStatistics(mean, cov * model.cal.amplitude(g)**2)
        if not stats.valid.all():
            raise ValueError("Singular local Fourier covariance")
        if target_moments:
            target = WindStatistics.observed(ref.wind.values)
            if origins:
                env["wind"].values = local_moments(env.wind.values, stats, target)
                wind = env.wind.values[1:31]
            else:
                wind = local_moments(wind, stats, target)
            stats = target
        key = ("SSP245", YEAR, member)
        fw = stats.daily(phase_seed(SEED, key, draw))
        for name, path in ((LIM, wind), (FOURIER, fw)):
            met.append(dict(protocol=protocol, method=name, member=member, draw=draw,
                            **weather_metrics(path, area, corridor, lon)))
        for o in origins:
            ev = sampled_window(env, o, DESIGN["event_days"])
            records.append(storm_records(ev, panels[o], None, protocol, LIM, member, draw, o))
            methods = [FOURIER, DAILY] if protocol == PROTOCOLS[0] else [FOURIER]
            for method in methods:
                provider = NativeFourierWind(stats, lon, lat, SEED, key, draw, independent_storms=True,
                                             origin_day=o, daily_mean=method == DAILY)
                records.append(storm_records(ev, panels[o], provider, protocol, method, member, draw, o))
    return met, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--weights", type=Path, default=ROOT / "weights")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--mpi-wind-moments", action="store_true",
                   help="Additional diagnostic using the target month's wind mean and covariance in both drivers")
    args = p.parse_args()
    mpi_g = read(args.work / "forcing.json")["warming_K"]
    fair = read(ROOT / "reports/fair_to_tc/run.json")
    if fair["future_year"] != YEAR:
        raise ValueError("FaIR benchmark year mismatch")
    tasks = [(args.work, args.weights, m, i, mpi_g, fair["future_g_K"], args.mpi_wind_moments)
             for i in range(DESIGN["weather_draws"]) for m in read(CONFIG / "members.json")["test"]]
    weather, attempts = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for met, records in pool.map(simulate, tasks):
            weather.extend(met)
            attempts.extend(records)
    suffix = "_moments" if args.mpi_wind_moments else ""
    with atomic(args.work / f"weather{suffix}.csv") as dest:
        pd.DataFrame(weather).to_csv(dest, index=False)
    with atomic(args.work / f"attempts{suffix}.parquet") as dest:
        pd.concat(attempts, ignore_index=True).to_parquet(dest, index=False)


if __name__ == "__main__":
    main()
