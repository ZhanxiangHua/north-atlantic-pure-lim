"""Independently retrain a selected model from prepared fields and compare physical laws across temperatures."""
import argparse
import time
from pathlib import Path

import numpy as np
from configure import setup
from records import WORK
from retrain import train
from spatial_check import field_loc
from weather import model_terms, projected

from annual_lim.layout import case_dir, design, read
from annual_lim.model import MonthlyModel
from tc_emulator.common import save_json


def link_input(destination, source):
    source = Path(source).resolve()
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Unexpected pre-existing input {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=True)


def qa(original, repeated):
    rows = []
    for m in range(1, 13):
        parent = MonthlyModel(original, m)
        other = MonthlyModel(repeated, m)
        row = dict(month=m, identical_model_id=parent.meta["id"] == other.meta["id"])
        gg = np.array([-0.18, 0.0, 1.5, 3.0, 4.34])
        laws = []
        for model in (parent, other):
            lag, bg, forced = model_terms(model, gg)
            raw, within, acs = projected(model.cal, lag, gg, np.full(len(gg), 30), bg)
            fields = []
            for g in gg:
                mu = model.bg.mean([g])[0]
                F = model.factor(g)
                for key in ("sst", "mld", "gamma"):
                    loc, load, valid = field_loc(model, mu, F, key)
                    idx = np.flatnonzero(valid)[::37]
                    fields.append(np.r_[loc[idx], (load[idx] @ load[idx].T).ravel()])
            laws.append((np.concatenate(fields), raw, within, acs))
        for key, a, b in zip(
            ("ocean_moments", "window_variances", "within_variances", "wind_autocorrelations"),
            laws[0],
            laws[1],
        ):
            delta = float(np.max(abs(a - b)))
            row[key + "_max_abs"] = delta
            row[key + "_allclose"] = bool(np.allclose(a, b, atol=3.0e-6, rtol=2.0e-5))
        rows.append(row)
    return rows


def run(campaign, spec, source, workers):
    root = Path(campaign).resolve()
    if root == WORK:
        raise ValueError("Reproduction requires an independent work tree")
    started = time.monotonic()
    setup()
    if spec["cohort"] in ("expanded", "extended"):
        for role, member, year, _ in design(WORK / "configs/additions")[1]:
            if role == "train":
                link_input(
                    case_dir(WORK / "data", member, year), case_dir(root / "data", member, year)
                )
    if spec["cohort"] == "extended":
        link_input(WORK / "infill", root / "infill")
    if spec["name"].startswith("rich_"):
        link_input(WORK / "ocean_only", root / "ocean_only")
    weights = train(spec, workers)
    rows = qa(source, weights)
    good = all(v for r in rows for k, v in r.items() if k.endswith("_allclose"))
    save_json(
        WORK / "reproduction.json",
        dict(
            recipe=spec,
            elapsed_seconds=round(time.monotonic() - started, 2),
            months=rows,
            all_physical_checks=good,
            identical_ids=sum(r["identical_model_id"] for r in rows),
            inputs="Prepared MPI fields plus published v1 reference. All new basis, regression, covariance and daily LIM parameters rebuilt.",
        ),
    )
    if not good:
        raise RuntimeError(
            "Independent numerical fit differs in physical units; see reproduction.json"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--recipe", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--workers", default=3, type=int)
    a = p.parse_args()
    run(a.campaign, read(a.recipe), a.source, a.workers)
