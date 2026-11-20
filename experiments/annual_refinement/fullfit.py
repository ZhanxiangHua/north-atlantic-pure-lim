"""Refit both the ocean and the wind, validate their coupled output and export selected models."""
import argparse
import os
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from backbone import Bank, Recipe, build
from joint import PhysicalValidation
from records import WORK, record
from weather import cache_read, model_terms, projected, train_cal, validation, year_table

from annual_lim.layout import DEFAULT_WORK, read
from annual_lim.model import MonthlyModel
from annual_lim.train import fit_dynamics
from tc_emulator.common import atomic, digest, lock, save_json, save_npz, sha
from tc_emulator.pure_lim_memory import MemoryLIM

BASE = Recipe()
DEPTH = Recipe(
    sst=48,
    ocean=72,
    sst_basis="sst",
    ocean_basis="ocean_d070",
    ridge=0.03,
    tail_penalty=125.0,
    mean_curve=0.5,
)
WIDE = replace(DEPTH, sst=72, ocean=96)
NESTED = Recipe(
    sst=48,
    ocean=72,
    sst_basis="nested_sst",
    ocean_basis="nested_ocean_depth",
    coarse_parent="release",
    tail_penalty=1.0,
    mean_curve=0.5,
    ridge=0.03,
)
CASES = {
    "released_design": BASE,
    "mean_curve_only": replace(BASE, ridge=0.03, mean_curve=0.5),
    "compact_depth": replace(DEPTH, sst=32, ocean=48),
    "medium_ocean": replace(DEPTH, ocean_basis="ocean"),
    "medium_depth": DEPTH,
    "medium_depth_rank48": replace(DEPTH, regression_rank=48),
    "medium_depth_pooled": replace(DEPTH, covariance="pooled"),
    "medium_depth_block": replace(DEPTH, covariance="block_active", covariance_slope_shrink=0.5),
    "wide_ocean": replace(WIDE, ocean_basis="ocean"),
    "wide_depth": WIDE,
    "wide_depth_rank48": replace(WIDE, regression_rank=48),
    "largest_depth": replace(WIDE, sst=96, ocean=128),
    "medium_wind160": replace(DEPTH, wind=160, wind_basis="wind"),
    "medium_wind192": replace(DEPTH, wind=192, wind_basis="wind"),
    "medium_slow120": replace(DEPTH, wind=120, wind_basis="mix_slow50"),
    "medium_slow160": replace(DEPTH, wind=160, wind_basis="mix_slow50"),
    "medium_shear160": replace(DEPTH, wind=160, wind_basis="mix_slow50_shear15"),
    "wide_slow160": replace(WIDE, wind=160, wind_basis="mix_slow50"),
    "wide_shear160": replace(WIDE, wind=160, wind_basis="mix_slow50_shear15"),
    "wide_refit_air": replace(WIDE, refit_other=True),
    "rich_basis": replace(WIDE, sst_basis="rich_sst", ocean_basis="rich_ocean_d070"),
    "rich_marginals": replace(
        WIDE, sst_basis="rich_sst", ocean_basis="rich_ocean_d070", ocean_marginals=True
    ),
    "rich_compact": replace(
        DEPTH,
        sst=32,
        ocean=48,
        sst_basis="rich_sst",
        ocean_basis="rich_ocean_d070",
        ocean_marginals=True,
    ),
    "rich_medium": replace(
        DEPTH, sst_basis="rich_sst", ocean_basis="rich_ocean_d070", ocean_marginals=True
    ),
    "rich_slow160": replace(
        WIDE,
        sst_basis="rich_sst",
        ocean_basis="rich_ocean_d070",
        ocean_marginals=True,
        wind_basis="mix_slow50",
        wind=160,
    ),
    "calendar_wide": replace(
        NESTED, sst=72, ocean=96, sst_basis="calendar_sst", ocean_basis="calendar_ocean_depth"
    ),
    "calendar_partial": replace(
        NESTED,
        coarse_parent="refit",
        tail_gain=0.25,
        sst_basis="calendar_sst",
        ocean_basis="calendar_ocean_depth",
    ),
    "nested_control": NESTED,
    "nested_wide": replace(NESTED, sst=72, ocean=96),
    "nested_unweighted": replace(NESTED, ocean_basis="nested_ocean"),
    "nested_refit": replace(NESTED, coarse_parent="refit"),
    "nested_partial": replace(NESTED, coarse_parent="refit", tail_gain=0.25),
    "nested_half": replace(NESTED, coarse_parent="refit", tail_gain=0.5),
    "nested_full": replace(NESTED, coarse_parent="refit", tail_gain=1.0),
    "nested_wide_partial": replace(NESTED, coarse_parent="refit", sst=72, ocean=96, tail_gain=0.25),
}
# Each calendar month uses exactly the same candidate recipes.
FILTERS = {
    "current_scale": (5, 20, 0.0, False, 1.0, 1.0),
    "centered": (5, 20, 0.5, False, 1.0, 1.0),
    "unbiased": (5, 20, 0.5, True, 1.0, 1.0),
    "short": (4, 16, 0.5, False, 1.0, 1.0),
    "month_weighted": (5, 20, 0.35, False, 1.0, 2.0),
}


def save_model(root, m, bank, r, cal, label, notes):
    dest = root / "weights" / f"{bank.month:02}"
    p = (
        bank.root
        / "export"
        / digest(
            [
                bank.info["input_digest"],
                [
                    r.sst_basis,
                    r.sst,
                    r.ocean_basis,
                    r.ocean,
                    r.split_mld,
                    r.wind_basis,
                    r.wind,
                    r.refit_other,
                ],
            ]
        )
        / "representation.npz"
    )
    with lock(p.parent / ".lock"):
        if not p.exists():
            m.rep.save(p)
    dest.mkdir(exist_ok=True, parents=True)
    representation = dest / "representation.npz"
    if not representation.exists():
        os.link(p, representation)
    if sha(representation) != sha(p):
        raise ValueError("Existing weights correspond to a different spatial fit")
    values = dict(
        center=m.bg.center,
        g_scale=m.bg.scale,
        beta=m.bg.beta,
        background_root=m.bg.lower0,
        directions=m.directions,
        background_response=[m.bg_amp, m.bg_slope],
        regression=m.mean.coefficients,
        z_scale=m.mean.scale,
        transition=m.base.a,
        innovation=m.base.q,
        stationary=m.base.sigma,
    )
    version = 2 if np.any(m.block_slope) else 1
    if version == 2:
        values.update(block_slope=m.block_slope, block_scale=m.block_scale)
    save_npz(dest / "parameters.npz", values)
    meta = dict(
        format=version,
        month=bank.month,
        calendar="proleptic_gregorian",
        training_fingerprint=digest([bank.info["input_digest"], asdict(r), label, notes]),
        weather=asdict(cal),
        files={k: sha(dest / k) for k in ("parameters.npz", "representation.npz")},
    )
    meta["id"] = digest(meta)
    save_json(dest / "model.json", meta)
    published = read(DEFAULT_WORK / "fit" / f"{bank.month:02}.json")
    save_json(
        root / "fit" / f"{bank.month:02}.json",
        dict(
            id=meta["id"],
            training=bank.info["train"],
            month=bank.month,
            bands=published["bands"],
            weak_cut=published["weak_cut"],
            spectral_radius=m.base.rho,
            lyapunov_relative_error=m.base.residual,
            background_model=asdict(r),
            calibration=asdict(cal),
            target="published bands and weak-shear threshold retained for matched comparisons",
        ),
    )
    return meta["id"]


def fit_one(bank, r, tr, va, val, name, export=None, chosen=None):
    month = bank.month
    m = build(bank, r)
    g = bank.scores["train_g"]
    nd = bank.scores["train_days"]
    pcs = bank.daily(r, "train")
    mean = m.mean.predict(m.bg, g, bank.z(r, "train"))
    m.base, _, _ = fit_dynamics(
        [pcs[i, :d] for i, d in enumerate(nd)],
        mean,
        {
            "lim_ridge_candidates": [r.lim_ridge]
            + [k for k in (0.001, 0.01, 0.1) if k > r.lim_ridge]
        },
    )
    published = MonthlyModel(DEFAULT_WORK / "weights", month)
    _, gt, _ = year_table(tr)
    _, gv, nv = year_table(va)
    m.cal = replace(
        published.cal, center=float(g.mean()), g_min=float(g.min()), g_max=float(g.max())
    )
    lags, bg, force = model_terms(m, np.r_[gt, gv])
    ocean = val.ocean_scores(m)
    wind = val.wind_errors(m, bank.z(r, "val"))
    rows = []
    methods = {}
    for filt, spec in FILTERS.items():
        if chosen and filt != chosen:
            continue
        cal, loss = train_cal(m, tr, lags, bg[: len(gt)], *spec)
        methods[filt] = {"calibration": asdict(cal), "training_loss": loss}
        scores = validation(
            cal, va, lags, bg[len(gt) :], force, -0.0115220616351514, 3.419637285227324
        )
        _, _, acf = projected(cal, lags, gv, nv, bg[len(gt) :])
        for row in scores:
            rows.append(dict(name=name, filter=filt, month=month, **row))
        methods[filt]["validation_acf_mean_abs_error"] = float(
            np.mean(np.abs(acf[:, :, 1:] - va["by_year_acf"][:, :, 1:]))
        )
        if export is not None:
            m.weather = MemoryLIM(m.base, cal)
            methods[filt]["export_id"] = save_model(export, m, bank, r, cal, filt, spec)
    details = dict(
        recipe=asdict(r),
        filters=methods,
        spectral_radius=m.base.rho,
        lyapunov_relative_residual=m.base.residual,
    )
    return pd.DataFrame(rows), ocean, wind, details


def run(cohort, month, stage, export_name=None, filter_name=None):
    if export_name:
        names = [export_name]
        target = WORK / "exports" / cohort / (export_name + "__" + filter_name)
    else:
        target = None

        def group(name, recipe):
            if name.startswith("calendar_"):
                return "calendar"
            if name.startswith("rich_"):
                return "rich"
            if name.startswith("nested_"):
                return "nested"
            return "basic" if recipe.wind_basis == "published_wind" else "wind"

        names = [n for n, r in CASES.items() if group(n, r) == stage]
        if cohort == "original":
            names = [n for n in names if n != "wide_refit_air"]
    bank = Bank(cohort, month)
    val = PhysicalValidation(bank)
    tr = cache_read(cohort, "train", month)
    va = cache_read("original", "val", month)
    for name in names:
        dest = WORK / "coupled" / cohort / name / f"{month:02}"
        if not export_name and (dest / "metrics.json").exists():
            continue
        if target is not None and (target / "fit" / f"{month:02}.json").exists():
            continue
        started = time.monotonic()
        record(
            "coupled_start", month=month, cohort=cohort, recipe=name, export=export_name is not None
        )
        scores, ocean, wind, detail = fit_one(
            bank, CASES[name], tr, va, val, name, target, filter_name
        )
        if not export_name:
            for label, frame in (("windows", scores), ("ocean", ocean), ("monthly_wind", wind)):
                with atomic(dest / (label + ".parquet")) as tmp:
                    frame.to_parquet(tmp, index=False, compression="zstd")
            save_json(dest / "metrics.json", detail)
        record(
            "coupled_done",
            month=month,
            cohort=cohort,
            recipe=name,
            export=export_name is not None,
            filters=len(detail["filters"]),
            seconds=round(time.monotonic() - started, 1),
        )
        print(month, cohort, name, round(time.monotonic() - started, 1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default="original")
    p.add_argument("--month", type=int, required=True)
    p.add_argument(
        "--stage", choices=["basic", "wind", "rich", "nested", "calendar"], default="basic"
    )
    p.add_argument("--export", choices=CASES)
    p.add_argument("--filter", choices=FILTERS)
    a = p.parse_args()
    if a.export and not a.filter:
        p.error("--export needs the selected filter")
    run(a.cohort, a.month, a.stage, a.export, a.filter)
