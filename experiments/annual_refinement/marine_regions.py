"""Physical regional ocean errors and upper-decile forecasts with common training thresholds."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joint import PhysicalValidation
from ocean import FIELDS, ocean_data, panel_cohort
from records import WORK, record
from scipy.special import ndtr
from spatial_check import field_loc

from annual_lim.layout import DEFAULT_WORK, read
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic

# Boxes identify grid subsets; they are not dynamical current boundaries.
BOXES = {
    "whole_basin": (0, 60, 260, 360),
    "TC_corridor": (10, 30, 280, 340),
    "northwest_Atlantic": (30, 45, 280, 315),
    "subpolar_Atlantic": (45, 60, 295, 340),
    "Caribbean_Gulf": (12, 30, 262, 295),
    "eastern_tropical_Atlantic": (5, 25, 315, 345),
}


def verify(tag, weights, month, role):
    dest = WORK / "marine_regions" / tag / role / f"{month:02}.parquet"
    if dest.is_file():
        return
    original = MonthlyModel(DEFAULT_WORK / "weights", month)
    model = MonthlyModel(weights, month)
    if role != "val":
        receipt = read(WORK / "FROZEN.json")
        if tag not in ("published_v1", receipt["stage"]) or (
            tag != "published_v1" and model.meta["id"] != receipt["months"][f"{month:02}"]
        ):
            raise ValueError("Regional scores require the frozen fit")
    cohort, split = panel_cohort(role)
    tr, meta = ocean_data(cohort, month, split)
    y, x = np.meshgrid(model.rep.lat, model.rep.lon, indexing="ij")
    v = PhysicalValidation(
        SimpleNamespace(geo=original.rep, month=month), tr, meta, np.empty((len(meta), 0))
    )
    v.regions = {
        name: (np.cos(np.deg2rad(y)) * ((y >= s) & (y <= n) & (x >= w) & (x <= e))).ravel()
        for name, (s, n, w, e) in BOXES.items()
    }
    prob = v.ocean_scores(model)
    with np.load(WORK / "verification/original_ocean_climatology" / f"{month:02}.npz") as f:
        threshold = f["tail"].reshape(3, -1)
        clim = f["clim"].reshape(3, -1)
    rows = []
    for fi, field in enumerate(FIELDS):
        group = "sst" if field == "sst" else "ocean"
        idx = int(field == "gamma")
        if field == "sst":
            maps = tr[:, :1]
        else:
            maps = tr[:, 1:].copy()
            maps[:, 0] = np.log(np.maximum(0.01, maps[:, 0]))
        block = model.rep.blocks[group]
        decoded = block.inverse(block.transform(maps))[:, idx].reshape(len(tr), -1)
        if field == "mld":
            decoded = np.maximum(0.01, np.exp(decoded))
        if field == "gamma":
            decoded = np.maximum(0.0, decoded)
        obs = tr[:, fi].reshape(len(tr), -1).astype(float)
        base = original.rep.blocks[group].valid[idx].ravel() & np.isfinite(threshold[fi])
        decoded = np.where(np.isfinite(decoded), decoded, clim[fi])
        cuts = np.log(np.maximum(threshold[fi], 0.01)) if field == "mld" else threshold[fi]
        for year, indices in meta.groupby("year").groups.items():
            select = np.asarray(indices)
            g = meta.loc[select[0], "g"]
            mu = model.bg.mean([g])[0]
            L = model.factor(g)
            loc, load, keep = field_loc(model, mu, L, field)
            loc = np.where(keep, loc, original.rep.blocks[group].mean[idx].ravel())
            sd = np.linalg.norm(load, axis=1)
            p = ndtr((loc - cuts) / np.maximum(sd, 1.0e-12))
            obs_year = obs[select]
            occ = obs_year > threshold[fi]
            d2 = (obs_year - decoded[select]) ** 2
            for region, area in v.regions.items():
                weight = np.isfinite(obs_year) * base * area
                wp = weight * occ
                brier = np.where(weight > 0, (p[None] - occ) ** 2, 0.0)
                blockinfo = meta.loc[select, ["member", "year", "g"]].copy()
                blockinfo = blockinfo.assign(
                    field=field,
                    region=region,
                    projection_mse=np.sum(np.where(weight > 0, d2, 0.0) * weight, axis=1)
                    / weight.sum(axis=1),
                    upper_tail_sse=np.sum(np.where(wp > 0, d2, 0.0) * wp, axis=1),
                    upper_tail_weight=wp.sum(axis=1),
                    upper_brier=np.sum(brier * weight, axis=1) / weight.sum(axis=1),
                    event_rate=wp.sum(axis=1) / weight.sum(axis=1),
                    predicted_rate=np.sum(np.where(weight > 0, p[None], 0.0) * weight, axis=1)
                    / weight.sum(axis=1),
                )
                rows.append(blockinfo)
    data = prob.merge(
        pd.concat(rows, ignore_index=True),
        on=["member", "year", "g", "region", "field"],
        validate="one_to_one",
    )
    with atomic(dest) as tmp:
        data.to_parquet(tmp, index=False, compression="zstd")
    record(
        "ocean_regional_verified",
        tag=tag,
        role=role,
        month=month,
        cases=len(meta),
        regions=len(BOXES),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, default=DEFAULT_WORK / "weights")
    p.add_argument("--month", required=True, type=int)
    p.add_argument("--role", choices=["val", "test", "fresh", "ssp245_year"], default="val")
    a = p.parse_args()
    verify(a.tag, a.weights, a.month, a.role)
