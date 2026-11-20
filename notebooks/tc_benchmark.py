"""Paired FAST integrations and harmonized six-hour storm diagnostics for the tutorial."""
from functools import lru_cache
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr

from annual_lim.layout import rng_for
from annual_lim.model import MonthlyModel
from annual_lim.sample import event
from tc_emulator.fourier_weather import NativeFourierWind, WindStatistics, grid_covariance
from tc_emulator.tc_adapter import geography, panel, tracks

METHODS = ("Pure LIM", "Native Fourier")
MONTH = 9
DAYS = 15
KNOT_MS = 1852 / 3600


@lru_cache(maxsize=1)
def fourier_inputs(weights):
    """Return the exact grid-point weather covariance at unit temperature amplitude."""
    model = MonthlyModel(weights, MONTH)
    cal = model.cal
    k = model.base.observed_dim
    # The emitted wind is a weighted sum of states at lags 0 and 5..20.
    # Cov(x_t,x_{t-h}) = (r A)^h Sigma. Both orientations enter at lag h>0.
    taps = np.zeros(cal.last_lag + 1)
    taps[0] = 1
    taps[cal.delay:] = -cal.feedback / (cal.last_lag - cal.delay + 1)
    transition = model.weather.dynamics.a
    cross = model.base.sigma[:, :k]
    covariance = (taps @ taps) * cross[:k].copy()
    for h in range(1, len(taps)):
        cross = transition @ cross
        covariance += (taps[h:] @ taps[:-h]) * (cross[:k] + cross[:k].T)
    covariance *= cal.normalization**2
    np.linalg.cholesky(covariance)
    return model, grid_covariance(model.rep, k, covariance)


def run_pair(case):
    weights, period, number, year, g, seed, attempts = case
    model, covariance = fourier_inputs(weights)
    env = event(weights, MONTH, year, g, DAYS, seed, number, number)
    panel_seed = int(rng_for(seed, "TC seeds", period, year, number).integers(2**32))
    seeds = panel(panel_seed, attempts)
    wind_mean = model.rep.blocks["wind"].inverse(env.conditional_wind_pc.values[None])[0]
    stats = WindStatistics(wind_mean, covariance * model.cal.amplitude(g)**2)
    if not stats.valid.all():
        raise ValueError("The native sampler requires positive-definite grid-point covariance")
    geography()  # Warm the shared geography cache before timing either solver.
    providers = {METHODS[0]: None, METHODS[1]: NativeFourierWind(
        stats, env.lon.values, env.lat.values, seed, (period, year, number), number,
        independent_storms=True)}
    traces, records, timings = {}, [], []
    # Alternate order so first calls do not always belong to the same method.
    for method in METHODS[::1 if number % 2 == 0 else -1]:
        start = perf_counter()
        trace, frame = tracks(env, seeds, DAYS, wind_provider=providers[method])
        seconds = perf_counter() - start
        traces[method] = trace[["storm_lon", "storm_lat", "vmax", "deep_shear"]]
        case_id = f"{period}-{number:02}"
        identifiers = dict(case=case_id, period=period, year=year, warming_K=g, method=method)
        records.append(frame.assign(**identifiers))
        timings.append(dict(**identifiers, seconds=seconds))
    paths = xr.concat([traces[m] for m in METHODS], dim=pd.Index(METHODS, name="method"),
                      combine_attrs="drop").expand_dims(case=[case_id])
    paths = paths.assign_coords(period=("case", [period]), year=("case", [year]), warming_K=("case", [g]))
    return paths, pd.concat(records, ignore_index=True), timings


def aligned_window(lon, lat, speed, hours, tropical):
    """Find a 48 h tropical segment from first 6-hourly wind >=18 m/s, inside the core."""
    reached = np.flatnonzero(tropical & np.isfinite(speed) & (speed >= 18))
    if not len(reached):
        return None
    i = reached[0]
    j = i + 8
    sl = slice(i, j + 1)
    # Keep one contiguous 9-fix window; a late/short storm contributes to TC rates
    # but has no complete 48-hour intensity or displacement measurement.
    if j >= len(speed) or not np.all(np.diff(hours[sl]) == 6) or not tropical[sl].all():
        return None
    x = (lon[sl] + 180) % 360 - 180
    y = lat[sl]
    if not (np.isfinite(speed[sl]).all() and ((x >= -100) & (x <= 0) & (y >= 0) & (y <= 60)).all()):
        return None
    from util.sphere import haversine
    return int(i), float(speed[sl].max()), float(haversine(x[0], y[0], x[-1], y[-1]))


def window_samples(paths, attempts, observations, years):
    geography()  # Makes the original FAST spherical-distance routine importable.
    rows = []
    hourly = paths.isel(elapsed_s=slice(None, None, 6))
    hours = hourly.elapsed_s.values / 3600
    for row in attempts[attempts.tc].itertuples(index=False):
        tr = hourly.sel({"case": row.case, "method": row.method, "seed_id": row.seed_id})
        speed = tr.vmax.values
        values = aligned_window(tr.storm_lon.values, tr.storm_lat.values, speed, hours, np.isfinite(speed))
        if values is not None:
            rows.append((row.period, row.method, row.year, row.case, str(row.seed_id), *values[1:]))
    obs = pd.read_csv(observations)
    obs["stamp"] = pd.to_datetime(obs.ymdhm, format="%Y%m%d%H%M")
    for storm, group in obs.groupby("storm", sort=False):
        group = group.sort_values("stamp")
        dt = group.stamp.to_numpy("datetime64[h]").astype("int64")
        speed = group.wind_kt.to_numpy(dtype=float) * KNOT_MS
        speed[speed < 0] = np.nan
        # TS, HU and TD retain tropical character; post-tropical/subtropical
        # observations do not count toward a continuous tropical 48-hour window.
        values = aligned_window(group.lon.values, group.lat.values, speed, dt,
                                group.status.isin(["TD", "TS", "HU"]).values)
        if values is None:
            continue
        i, peak, distance = values
        stamp = group.stamp.iloc[i]
        if stamp.month == MONTH and stamp.year in years:
            rows.append(("Historical", "HURDAT2", stamp.year, str(stamp.year), storm, peak, distance))
    return pd.DataFrame(rows, columns=["period", "method", "year", "case", "storm",
                                       "peak_48h_ms", "displacement_48h_km"])


def paired_rates(attempts, seed, resamples=5000):
    """Resample entire event pairs, preserving shared weather and paired seed panels."""
    rows = []
    for period, group in attempts.groupby("period", sort=False):
        for metric in ("tc", "landfall"):
            rates = group.groupby(["case", "method"])[metric].mean().unstack()
            delta = 100 * (rates[METHODS[0]] - rates[METHODS[1]]).to_numpy()
            rng = rng_for(seed, "event bootstrap", period)
            indices = rng.integers(len(delta), size=(resamples, len(delta)))
            low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
            rows.append((period, metric, len(delta), delta.mean(), low, high))
    return pd.DataFrame(rows, columns=["period", "outcome", "paired_events", "lim_minus_fourier_pp",
                                       "event_boot_low_pp", "event_boot_high_pp"])
