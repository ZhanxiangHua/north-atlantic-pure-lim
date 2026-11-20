"""Compose independently calibrated ocean and wind changes from the same verified parent fit."""
import argparse
import os
from pathlib import Path

import numpy as np
from records import WORK, record

from annual_lim.layout import read
from annual_lim.model import MonthlyModel
from tc_emulator.common import digest, save_json, save_npz, sha


def paths(tag, family, weather, spread):
    p = read(WORK / "parents" / (tag + ".json"))
    base = Path(p["source"])
    w = (
        (WORK / "exports" / family / (tag + "__" + weather) / "weights")
        if family != "parent"
        else base
    )
    s = (
        (WORK / "exports" / "spread" / (tag + "__" + spread) / "weights")
        if spread != "own"
        else base
    )
    return base, w, s, p


def merge(tag, family, weather, spread, label):
    original, w, s, p = paths(tag, family, weather, spread)
    dest = WORK / "exports" / "combined" / label
    changed = set(("background_root", "directions"))
    for m in range(1, 13):
        month = f"{m:02}"
        paths_m = [r / month for r in (original, w, s)]
        parents = [read(q / "model.json") for q in paths_m]
        referenced = parents[0]["id"]
        meta = parents[1].copy()
        if len(set(v["files"]["representation.npz"] for v in parents)) != 1:
            raise ValueError("Combined candidates do not share their spatial representation")
        details = [read(q.parent / "fit" / (month + ".json")) for q in (original, w, s)]
        if family != "parent" and details[1]["weather_parent"] != referenced:
            raise ValueError("Weather calibration has a different parent")
        if spread != "own" and details[2]["background_parent"] != referenced:
            raise ValueError("Spatial calibration has a different parent")
        out = dest / "weights" / month
        out.mkdir(exist_ok=True, parents=True)
        with np.load(paths_m[0] / "parameters.npz") as a, np.load(
            paths_m[1] / "parameters.npz"
        ) as b, np.load(paths_m[2] / "parameters.npz") as c:
            for k in a.files:
                if k not in changed:
                    np.testing.assert_array_equal(a[k], c[k])
                if k not in ("transition", "innovation", "stationary"):
                    np.testing.assert_array_equal(a[k], b[k])
            pars = {k: c[k] if k in changed else b[k] for k in a.files}
        if not (out / "representation.npz").exists():
            os.link(paths_m[0] / "representation.npz", out / "representation.npz")
        save_npz(out / "parameters.npz", pars)
        meta.pop("id")
        meta["training_fingerprint"] = digest(
            [x["id"] for x in parents] + [spread, family, weather]
        )
        meta["files"] = dict(meta["files"], **{"parameters.npz": sha(out / "parameters.npz")})
        meta["id"] = digest(meta)
        save_json(out / "model.json", meta)
        fit = details[0].copy()
        fit.update(
            calibration=meta["weather"],
            id=meta["id"],
            background_postfit=spread,
            weather_postfit=weather,
            sources=dict(parent=referenced, weather=parents[1]["id"], ocean=parents[2]["id"]),
        )
        save_json(dest / "fit" / (month + ".json"), fit)
        test = MonthlyModel(dest / "weights", m)
        if test.base.rho >= 1 or test.base.residual > 1.0e-8:
            raise ValueError("Exported state lost stationarity")
    record(
        "calibrations_composed",
        parent=tag,
        weather_family=family,
        weather=weather,
        spread=spread,
        label=label,
    )
    return dest / "weights"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for flag in ("parent", "family", "weather", "spread", "label"):
        p.add_argument("--" + flag, required=True)
    a = p.parse_args()
    merge(a.parent, a.family, a.weather, a.spread, a.label)
