"""Read a real CMIP member-year once and write its twelve calendar months."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
import time

import cftime
import h5py
import numpy as np
import pandas as pd
import requests

from tc_emulator.assets import RangeFile, array, attr
from tc_emulator.common import MODEL, TABLES, atomic, digest, lock, save_json, sha
from tc_emulator.physical import (
    cut,
    gradient,
    grid_indices,
    ocean_geometry,
    pressure,
    remap,
    temperature,
)
from .layout import case_dir, design, read


@lru_cache(maxsize=4)
def catalog(config):
    frame = pd.read_parquet(Path(config) / "catalog.parquet")
    return {
        k: list(v.itertuples(index=False))
        for k, v in frame.groupby(["member", "experiment", "variable"])
    }


def rows_for(config, member, year):
    exp = "historical" if year <= 2014 else "ssp585"
    result = {}
    for var, table in TABLES.items():
        first, last = (f"{year}0101", f"{year}1231") if table == "day" else (f"{year}01", f"{year}12")
        choices = [r for r in catalog(config)[member, exp, var] if r.first <= first and r.last >= last]
        if not choices:
            raise ValueError(f"No complete {year} coverage for {exp}/{member}/{var}")
        result[var] = max(choices, key=lambda r: r.version)
    return result


def clock(h, row, year):
    t = h["time"]
    cal, units = attr(t, "calendar", "standard"), attr(t, "units")
    zero = cftime.num2date(t[0], units, calendar=cal)
    starts = [cftime.datetime(year, m, 1, calendar=cal) for m in range(1, 13)] + [
        cftime.datetime(year + 1, 1, 1, calendar=cal)
    ]
    ndays = np.array([(right - left).days for left, right in zip(starts[:-1], starts[1:])])
    if row.table == "day":
        left = (starts[0] - cftime.datetime(zero.year, zero.month, zero.day, calendar=cal)).days
        dates = [starts[0] + timedelta(days=i) for i in range(ndays.sum() + 1)]
    else:
        left = 12 * (year - zero.year) + 1 - zero.month
        dates = starts
    edges = cftime.date2num(dates, units, calendar=cal)
    if left < 0 or left + len(edges) - 1 > len(t):
        raise ValueError("Release date range does not match its CF time axis")
    return left, edges, ndays, cal


def regional(h, row, year, settings, remote, month=None):
    for k, v in {
        "source_id": MODEL,
        "experiment_id": row.experiment,
        "variant_label": row.member,
        "grid_label": "gn",
    }.items():
        if attr(h, k) != v:
            raise ValueError(f"Wrong CMIP {k} in {row.esgf}")
    if row.experiment != "historical" and attr(h, "parent_variant_label") != row.member:
        raise ValueError("Scenario lineage mismatch")
    name = row.variable
    ds = h[name]
    if "time: mean" not in attr(ds, "cell_methods", ""):
        raise ValueError(f"Expected CF time mean for {name}")
    left, edges, ndays, cal = clock(h, row, year)
    if month is not None:
        if not 1 <= month <= 12:
            raise ValueError("Month must be between 1 and 12")
        start = int(ndays[: month - 1].sum()) if row.table == "day" else month - 1
        length = int(ndays[month - 1]) if row.table == "day" else 1
        left += start
        edges = edges[start : start + length + 1]
        ndays = ndays[month - 1 : month]
    t = h["time"]
    bounds = h[attr(t, "bounds")]
    out = dict(calendar=cal, first_month=month or 1, ndays=ndays, units=attr(ds, "units"))
    if name in ("thetao", "tos", "mlotst"):
        out["latitude"], out["longitude"] = array(h["latitude"]), array(h["longitude"])
        if name == "thetao":
            if attr(h["lev"], "units") not in ("m", "meters") or attr(h["lev"], "positive") != "down":
                raise ValueError("Ocean depth must increase downwards in metres")
            out["depth"] = array(h["lev"])
    else:
        native_lat, native_lon = array(h["lat"]), array(h["lon"])
        j, i, out["lat"], out["lon"] = grid_indices(
            native_lat, native_lon, settings["domain"], settings["halo_degrees"]
        )
        if "plev" in h:
            p = pressure(array(h["plev"]), attr(h["plev"], "units"))
            if name in ("ua", "va"):
                if attr(ds, "standard_name") != {"ua": "eastward_wind", "va": "northward_wind"}[
                    name
                ] or out["units"] not in ("m s-1", "m/s"):
                    raise ValueError(f"Invalid wind identity or units for {name}")
                lev = [int(np.flatnonzero(p == v).item()) for v in (25000, 85000)]
            else:
                lev = np.argsort(-p)
            out["plev"] = p[lev]
        if name == "tas":
            lat_edges = np.deg2rad(array(h["lat_bnds"]))
            # Cell area on a regular lon grid is proportional to sin(phi_N)-sin(phi_S).
            area = abs(np.diff(np.sin(lat_edges), axis=1)[:, 0])
    # A block contains whole compressed HDF chunks. Reading nearby offsets together
    # also brings along interleaved time/bounds chunks on the MPI NetCDF replicas.
    step = 28 if row.table == "day" else 1 if name == "thetao" else 12
    parts = []
    for offset in range(0, len(edges) - 1, step):
        lo, hi = left + offset, left + min(offset + step, len(edges) - 1)
        if remote:
            remote.preload(ds, lo, hi)
        b = bounds[lo:hi]
        expect = np.column_stack(
            (edges[offset : offset + hi - lo], edges[offset + 1 : offset + hi - lo + 1])
        )
        np.testing.assert_allclose(b, expect, atol=1e-5, rtol=0)
        np.testing.assert_allclose(t[lo:hi], expect.mean(axis=1), atol=1e-5, rtol=0)
        raw = array(ds, slice(lo, hi))
        if name == "tas":
            value = (
                np.average(temperature(raw, out["units"]).mean(axis=-1), axis=-1, weights=area) - 273.15
            )
        elif name in ("ua", "va", "hus", "ta", "psl"):
            value = cut(raw[:, lev] if raw.ndim == 4 else raw, j, i)
        else:
            value = raw
        parts.append(np.asarray(value, dtype=np.float32))
    out["data"] = np.concatenate(parts)
    return out


def fetch(row, year, settings, month=None):
    errors = []
    for location in dict.fromkeys((row.local, row.aws, row.esgf)):
        if not location or (not location.startswith("http") and not Path(location).is_file()):
            continue
        remote = None
        try:
            if location.startswith("http"):
                remote = RangeFile(location, row.bytes)
            with h5py.File(remote if remote else location, "r") as h:
                result = regional(h, row, year, settings, remote, month)
            return result, dict(
                url=location,
                source_checksum=row.checksum,
                version=row.version,
                transfer_bytes=remote.bytes_read if remote else 0,
            )
        except (OSError, ValueError, KeyError, requests.RequestException) as err:
            errors.append(f"{location}: {err}")
        finally:
            if remote:
                remote.close()
    raise OSError(f"{row.member}/{year}/{row.variable} " + " | ".join(errors))


def assemble(raw, settings, month):
    ta = raw["ta"]
    m = month - ta.get("first_month", 1)
    if not 0 <= m < len(ta["ndays"]):
        raise ValueError("Requested month is missing from the prepared fields")
    for v in ("psl", "hus", "tas", "ua", "va"):
        for coord in ("lat", "lon"):
            np.testing.assert_allclose(raw[v][coord], ta[coord], rtol=0, atol=1e-5)
    for v in raw:
        np.testing.assert_array_equal(raw[v]["ndays"], ta["ndays"])
        if raw[v]["calendar"] != ta["calendar"] or raw[v].get("first_month", 1) != ta.get("first_month", 1):
            raise ValueError("Inconsistent calendar or starting month")
    np.testing.assert_array_equal(ta["plev"], raw["hus"]["plev"])
    for v in ("tos", "mlotst"):
        for coord in ("latitude", "longitude"):
            np.testing.assert_allclose(raw[v][coord], raw["thetao"][coord], rtol=0, atol=1e-5)
    geom = ocean_geometry(raw["thetao"]["latitude"], raw["thetao"]["longitude"], ta["lat"], ta["lon"])

    def values(v):
        return raw[v]["data"][m].astype(float)

    if raw["mlotst"]["units"] != "m" or raw["hus"]["units"] not in ("1", "kg kg-1", "kg/kg"):
        raise ValueError("Invalid MLD or specific humidity units")
    mld = values("mlotst")
    if np.any(mld[np.isfinite(mld)] < 0):
        raise ValueError("Negative mixed layer depth")
    # Gamma = -100 * slope(T versus depth) in the first 100 m beneath the MLD.
    # Fit on native ocean columns, then interpolate with normalized wet-vertex weights.
    grad, _ = gradient(
        raw["thetao"]["depth"],
        temperature(values("thetao"), raw["thetao"]["units"]),
        mld,
        settings["depth_m"],
    )
    bounds = np.r_[0, ta["ndays"].cumsum()]
    days = slice(bounds[m], bounds[m + 1])
    u, v = raw["ua"]["data"][days], raw["va"]["data"][days]
    wind = np.stack((u[:, 0], v[:, 0], u[:, 1], v[:, 1]), axis=1)
    if not np.isfinite(wind).all():
        raise ValueError("Missing regional winds")
    arrays = dict(
        wind=wind,
        sst=remap(temperature(values("tos"), raw["tos"]["units"]), geom),
        mld=remap(mld, geom),
        gamma=np.maximum(remap(grad, geom), 0.0),
        ta=temperature(values("ta"), ta["units"]),
        hus=values("hus"),
        psl=pressure(values("psl"), raw["psl"]["units"]),
    )
    return {v: a.astype("float32") for v, a in arrays.items()}


def prepare_case(config, work, case):
    role, member, year, g = case
    settings, _ = design(config)
    folder = case_dir(work, member, year)
    mark = folder / "complete.json"
    rows = rows_for(config, member, year)
    signature = digest(
        [
            settings["domain"],
            settings["halo_degrees"],
            settings["depth_m"],
            "v1",
            {v: r.checksum for v, r in rows.items()},
            member,
            year,
        ]
    )
    with lock(folder / ".lock"):
        if mark.exists():
            old = read(mark)
            if old["signature"] != signature:
                raise ValueError(
                    "A different source release or preprocessing configuration occupies this work directory"
                )
            if all(
                (folder / k).is_file()
                and (folder / k).stat().st_size == s
                and sha(folder / k) == old["hashes"][k]
                for k, s in old["sizes"].items()
            ):
                # The physical files only depend on the source release and grid. A new
                # member split or forcing series can reuse their arrays and refresh the labels.
                if (old["role"], old["g"]) != (role, g):
                    save_json(mark, {**old, "role": role, "g": g})
                return f"cached {member} {year}"
        start = time.monotonic()
        raw, sources = {}, {}
        for v, row in rows.items():
            raw[v], sources[v] = fetch(row, year, settings)
        hashes, sizes = {}, {}
        for month in range(1, 13):
            name = f"{month:02}.npz"
            arrays = assemble(raw, settings, month)
            with atomic(folder / name) as tmp:
                np.savez(
                    tmp,
                    **arrays,
                    lat=raw["ta"]["lat"],
                    lon=raw["ta"]["lon"],
                    plev=raw["ta"]["plev"],
                    year=year,
                    month=month,
                    member=member,
                    calendar=raw["ta"]["calendar"],
                )
            hashes[name] = sha(folder / name)
            sizes[name] = (folder / name).stat().st_size
        save_json(
            mark,
            dict(
                signature=signature,
                role=role,
                member=member,
                year=year,
                g=g,
                sources=sources,
                hashes=hashes,
                sizes=sizes,
                days=raw["ta"]["ndays"].tolist(),
                calendar=raw["ta"]["calendar"],
                global_monthly_tas_C=raw["tas"]["data"].tolist(),
                elapsed_s=round(time.monotonic() - start, 2),
            ),
        )
    return f"prepared {member} {year} {time.monotonic()-start:.1f}s"


def run(config, work, shard=0, shards=1, workers=1):
    if not 0 <= shard < shards or workers < 1:
        raise ValueError("Invalid shard or worker count")
    _, cases = design(config)
    todo = cases[shard::shards]
    errors = []
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(prepare_case, config, work, c): c for c in todo}
        for task in as_completed(futures):
            try:
                print(task.result(), flush=True)
            except Exception as e:
                errors.append((futures[task], str(e)))
                print("FAILED", *errors[-1], flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} incomplete member-years in shard {shard}: {errors[0]}")
