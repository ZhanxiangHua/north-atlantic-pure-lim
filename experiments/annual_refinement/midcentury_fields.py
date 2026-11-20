"""Check frozen EOF reconstruction and conditional monthly steering on the new MPI maps."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from annual_lim.layout import read
from annual_lim.model import MonthlyModel
from annual_lim.train import months
from midcentury_prepare import CFG, STUDY
from midcentury_sources import PLAN, REPORT
from paired_intervals import panel_weights
from score_storms import limits
from tc_emulator.common import ROOT, atomic
from tc_emulator.diagnostics import region_area, steering


def evaluate(month, base, research):
    plan = read(PLAN)
    manifest = read(ROOT / "checkpoints/annual_refinement/manifest.json")["months"][f"{month:02}"]
    models = {plan["baseline"]: MonthlyModel(base, month), plan["candidate"]: MonthlyModel(research, month)}
    if [model.meta["id"] for model in models.values()] != [manifest["base_id"], manifest["model_id"]]:
        raise ValueError("Unexpected monthly model")
    data, meta, _ = months(CFG, STUDY / "ssp245_year", month, "test")
    frame = pd.DataFrame(meta)
    warm = frame.g.to_numpy()
    old, new = models.values()
    np.testing.assert_equal(old.rep.lat, new.rep.lat)
    np.testing.assert_equal(old.rep.lon, new.rep.lon)
    area = region_area(old.rep.lat, old.rep.lon)
    target = np.stack([steering(d["wind"].astype(float)).mean(axis=0) for d in data])
    rows, maps = [], {}
    for tag, model in models.items():
        z = np.stack([model.rep.encode(d) for d in data])
        pc = model.mean.predict(model.bg, warm, z)
        pred = steering(model.rep.blocks["wind"].inverse(pc))
        month_mse = (np.square(pred - target) * area).sum(axis=(2, 3)).mean(axis=1)
        rows.append(frame.assign(month=month, model=tag, field="monthly_steering", mse=month_mse))
        maps[tag] = model.rep.decode(z)
    for field in ("sst", "mld", "gamma"):
        truth = np.stack([d[field] for d in data])
        arrays = {}
        for tag, vals in maps.items():
            arr = vals[field]
            if field in ("mld", "gamma"):
                arr = np.maximum(arr, .01 if field == "mld" else 0.)
            arrays[tag] = arr
        # Both fits are scored on exactly the same wet cells of each observed map.
        ok = np.isfinite(truth) & np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
        weights = ok * area
        total = weights.sum(axis=(1, 2))
        if (total <= 0).any():
            raise ValueError(f"No shared wet cells for {field}")
        for tag, arr in arrays.items():
            mse = (np.square(np.where(ok, arr - truth, 0).astype(float)) * weights).sum(axis=(1, 2)) / total
            rows.append(frame.assign(month=month, model=tag, field=field, mse=mse))
    out = pd.concat(rows)[["year", "member", "month", "g", "model", "field", "mse"]]
    with atomic(STUDY / "fields" / f"{month:02}.parquet") as tmp:
        out.to_parquet(tmp, index=False)


def summarize():
    p = read(PLAN)
    full = pd.concat([pd.read_parquet(STUDY / "fields" / f"{m:02}.parquet") for m in p["months"]])
    wide = full.pivot(index=["year", "member", "g", "month", "field"], columns="model", values="mse")
    if wide.isna().any().any() or len(wide) != 6 * 8 * 12 * 4:
        raise ValueError("Incomplete paired field panel")
    with atomic(REPORT / "field_cases.parquet") as tmp:
        full.sort_values(["year", "member", "month", "field", "model"]).to_parquet(tmp, index=False)
    scores = []
    for scope, period in [(str(y), d) for y, d in full.groupby("year")] + [("new_six", full)]:
        for field, data in period.groupby("field"):
            mse = data.groupby(["member", "year", "g", "model"]).mse.mean().unstack("model")
            m, y, g = [mse.index.get_level_values(k).to_numpy() for k in ("member", "year", "g")]
            w, _, _ = panel_weights(m, y, g, "all", p["bootstrap_draws"], p["bootstrap_seed"])
            base, research = [np.sqrt(w @ mse[tag].to_numpy()) for tag in (p["baseline"], p["candidate"])]
            change = research - base
            scores.append(dict(scope=scope, field=field, v1_rmse=base[0], research_rmse=research[0],
                               change=change[0], **limits(change)))
    with atomic(REPORT / "field_scores.csv") as tmp:
        pd.DataFrame(scores).to_csv(tmp, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", type=int, choices=range(1, 13))
    parser.add_argument("--base", type=Path)
    parser.add_argument("--research", type=Path)
    args = parser.parse_args()
    if args.month is None:
        summarize()
    elif not args.base or not args.research:
        parser.error("A single month requires both weight directories")
    else:
        evaluate(args.month, args.base, args.research)
