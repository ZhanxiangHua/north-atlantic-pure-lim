"""Prepare additional monthly ocean maps and audit agreement with the nine-variable processor."""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from records import WORK, record

from annual_lim.layout import CONFIG, DEFAULT_WORK, read
from annual_lim.prepare import fetch, rows_for
from tc_emulator.common import atomic, lock, save_json, sha
from tc_emulator.physical import gradient, ocean_geometry, remap, temperature


def prepare(task):
    member, year, g = task
    folder = WORK / "ocean_only" / member / str(year)
    with lock(folder / ".lock"):
        if (folder / "complete.json").is_file():
            return
        settings = read(CONFIG / "settings.json")
        rows = rows_for(CONFIG, member, year)
        raw = {}
        sources = {}
        for k in ("tos", "mlotst", "thetao"):
            raw[k], sources[k] = fetch(rows[k], year, settings)
        anchor = raw["thetao"]
        for k in ("tos", "mlotst"):
            assert anchor["calendar"] == raw[k]["calendar"]
            np.testing.assert_array_equal(anchor["ndays"], raw[k]["ndays"])
            for coord in ("latitude", "longitude"):
                np.testing.assert_allclose(anchor[coord], raw[k][coord], rtol=0, atol=1e-5)
        with np.load(DEFAULT_WORK / "weights/01/representation.npz") as ref:
            lat, lon = ref["lat"], ref["lon"]
        geom = ocean_geometry(anchor["latitude"], anchor["longitude"], lat, lon)
        assert raw["mlotst"]["units"] == "m"
        vals = []
        for m in range(12):
            depth = raw["mlotst"]["data"][m].astype(float)
            assert np.all(depth[np.isfinite(depth)] >= 0)
            temp = temperature(anchor["data"][m].astype(float), anchor["units"])
            slope, _ = gradient(anchor["depth"], temp, depth, settings["depth_m"])
            sst = temperature(raw["tos"]["data"][m].astype(float), raw["tos"]["units"])
            vals.append(
                np.stack(
                    (remap(sst, geom), remap(depth, geom), np.maximum(remap(slope, geom), 0.0))
                )
            )
        data = np.asarray(vals, dtype="float32")
        if member == "r1i1p1f1" and year == 2000:
            for m in range(12):
                with np.load(
                    DEFAULT_WORK / "prepared" / member / str(year) / f"{m+1:02}.npz"
                ) as base:
                    np.testing.assert_array_equal(
                        data[m], np.stack([base[k] for k in ("sst", "mld", "gamma")])
                    )
        with atomic(folder / "maps.npy") as tmp:
            np.save(tmp, data)
        save_json(
            folder / "complete.json",
            dict(
                member=member,
                year=year,
                g=g,
                calendar=anchor["calendar"],
                source=sources,
                units=["K", "m", "K/100m"],
                fields=["sst", "mld", "gamma"],
                sha256=sha(folder / "maps.npy"),
            ),
        )
        return member, year


def main(index):
    spec = read(Path(__file__).with_name("design.json"))
    yrs = read(Path(__file__).with_name("ocean_extra_years.json"))["years"]
    original = read(CONFIG / "settings.json")
    split = read(CONFIG / "members.json")
    other = sum(
        [
            original["train_years"],
            original["validation_years"],
            original["test_years"],
            spec["extra_train_years"],
            spec["fresh_test_years"],
        ],
        [],
    )
    assert not set(yrs).intersection(other)
    g = pd.read_csv(CONFIG / "forcing.csv").set_index("year").g
    assert g[yrs].between(g[original["train_years"]].min(), g[original["train_years"]].max()).all()
    all_cases = [(m, y, float(g[y])) for y in yrs for m in split["train"]]
    cases = all_cases[index::12]
    if index == 0:
        cases = [("r1i1p1f1", 2000, float(g[2000]))] + cases
    start = time.monotonic()
    record("ocean_extra_source_start", shard=index, cases=len(cases))
    with ProcessPoolExecutor(2) as pool:
        for case, out in zip(cases, pool.map(prepare, cases)):
            print(case[0], case[1], "ready", flush=True)
    record(
        "ocean_extra_source_done",
        shard=index,
        cases=len(cases),
        seconds=round(time.monotonic() - start, 1),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    main(p.parse_args().shard)
