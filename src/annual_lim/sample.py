"""Draw a month-conditioned event or an independently sampled calendar year."""
import calendar
import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

from tc_emulator.common import COMPONENTS, atomic
from tc_emulator.sampling import UNITS, fair_years, physical_background
from .model import MonthlyModel


def forcing(g, path, year, scenario=None, draw=None):
    if path is None:
        if g is None or not np.isfinite(g):
            raise ValueError("Supply a finite annual warming in K relative to 1850–1900")
        return float(g), {}
    rows = fair_years(path)
    if year is not None:
        rows = rows[rows.year == year]
    for column, value in (("scenario", scenario), ("draw_id", draw)):
        if value is not None:
            rows = rows[rows[column] == value]
    if len(rows) != 1:
        raise ValueError("Select exactly one scenario/draw/year in the forcing file")
    row = rows.to_dict("records")[0]
    return row.pop("g"), row


def event(weights, month, year, g, days, seed, background_id, weather_id, device=None, observed=None):
    if not 1 <= month <= 12 or not 1 <= year <= 9999 or not np.isfinite(g):
        raise ValueError("Invalid calendar month, year or warming")
    days = calendar.monthrange(year, month)[1] if days is None else int(days)
    if days < 1:
        raise ValueError("Event duration must be positive")
    model = MonthlyModel(weights, month)
    rep = model.rep
    if observed is None:
        z = model.draw_z(g, year, seed, background_id)
        fields, qc = physical_background(model, z, device)
    else:
        if int(observed.attrs["month"]) != month:
            raise ValueError("Observed background belongs to a different month")
        for name in ("lat", "lon", "plev"):
            np.testing.assert_array_equal(observed[name], getattr(rep, name))
        z = rep.encode(observed)
        fields = {k: observed[k].values for k in UNITS}
        qc = {}
    # Day 1 at noon is t=0. FAST requires one interpolation node before the
    # event and one beyond its final day. A longer request uses the same prefix.
    times = np.arange(-1, days + 2, dtype=float) * 86400.0
    w = model.pc_path(g, z, len(times), year, seed, background_id, weather_id, device)
    ds = xr.Dataset(
        coords={
            "lat": ("lat", rep.lat, {"units": "degrees_north"}),
            "lon": ("lon", rep.lon, {"units": "degrees_east"}),
            "plev": ("plev", rep.plev, {"units": "Pa", "positive": "down"}),
            "component": COMPONENTS,
            "t_seconds": ("t_seconds", times, {"units": "s"}),
        }
    )
    for name, a in fields.items():
        dims = ("plev", "lat", "lon") if a.ndim == 3 else ("lat", "lon")
        ds[name] = (dims, a.astype("float32"), {"units": UNITS[name]})
    ds["wind"] = (
        ("t_seconds", "component", "lat", "lon"),
        rep.blocks["wind"].inverse(w, device),
        {"units": "m s-1"},
    )
    ds["wind_pc"] = (("t_seconds", "wind_mode"), w)
    ds["background_pc"] = ("background_mode", z)
    ds["conditional_wind_pc"] = ("wind_mode", model.wind_mean(g, z))
    ds.attrs.update(
        model_id=model.meta["id"],
        weather_sampler="pure_lim",
        statistical_month=month,
        month=month,
        year=year,
        duration_days=days,
        background_id=str(background_id),
        weather_id=str(weather_id),
        seed=int(seed),
        g_degC=float(g),
        g_extrapolated=int(not model.cal.g_min <= g <= model.cal.g_max),
        baseline="1850-1900",
        origin_day=1,
        calendar=model.meta["calendar"],
        background_source="observed" if observed is not None else "emulated",
    )
    for k, v in qc.items():
        ds.attrs["qc_" + k] = int(v)
    return ds


def year_sample(weights, year, g, seed, bg_id, weather_id, device=None):
    monthly, daily, ids = [], [], []
    for month in range(1, 13):
        ds = event(weights, month, year, g, None, seed, bg_id, weather_id, device)
        n = calendar.monthrange(year, month)[1]
        # Only true calendar days go into the annual axis; discard interpolation padding.
        first = (dt.date(year, month, 1) - dt.date(year, 1, 1)).days
        part = ds.wind.isel(t_seconds=slice(1, n + 1)).rename(t_seconds="time")
        part = part.assign_coords(time=first + 0.5 + np.arange(n))
        daily.append(part)
        monthly.append(ds[list(UNITS)].expand_dims(calendar_month=[month]))
        ids.append(ds.attrs["model_id"])
    out = xr.concat(monthly, dim="calendar_month", combine_attrs="drop")
    out["wind"] = xr.concat(daily, dim="time")
    out.time.attrs.update(units=f"days since {year:04}-01-01", calendar="proleptic_gregorian")
    out["model_id"] = ("calendar_month", ids)
    out.attrs.update(
        year=year,
        g_degC=float(g),
        baseline="1850-1900",
        seed=int(seed),
        weather_sampler="pure_lim",
        background_id=str(bg_id),
        weather_id=str(weather_id),
        month_boundary="independent monthly backgrounds and weather histories",
    )
    return out


def write(ds, path):
    with atomic(Path(path)) as tmp:
        ds.to_netcdf(tmp)
