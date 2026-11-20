"""Calibrate the transfer of background variance to monthly wind jointly with the pure LIM."""
import argparse
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from backbone import Bank, build
from fullfit import CASES, save_model
from joint import PhysicalValidation
from records import WORK, record
from scipy.optimize import least_squares
from weather import cache_read, model_terms, projected, train_cal, validation, year_table

from annual_lim.layout import DEFAULT_WORK
from annual_lim.model import MonthlyModel
from annual_lim.statistics import trace_lags, variance
from annual_lim.train import fit_dynamics
from tc_emulator.common import atomic, save_json
from tc_emulator.pure_lim_memory import MemoryCalibration, response


def cross_validated_gain(bank, recipe, k=5):
    z = bank.z(recipe, "train")
    g = bank.scores["train_g"]
    w = bank.wind_monthly(recipe, "train")
    metric = bank.G(recipe)[0]
    members = bank.scores["train_member"]
    folds = np.array_split(np.random.default_rng(52337).permutation(np.unique(members)), k)
    num = []
    den = []
    for fold in folds:
        keep = np.isin(members, fold)
        model = build(bank, recipe, np.flatnonzero(~keep))
        a = model.mean.predict(model.bg, g[keep], model.bg.mean(g[keep]))
        shift = model.mean.predict(model.bg, g[keep], z[keep]) - a
        num.append(float(np.sum((shift @ metric) * (w[keep] - a))))
        den.append(float(np.sum((shift @ metric) * shift)))
    return float(np.clip(sum(num) / sum(den), 0.5, 1.4)), np.divide(num, den).tolist()


def fit_budget(tr, lags, bg, init, cvgain, mode, center_w, month_w, regularization):
    _, g, nd = year_table(tr)
    target = tr["by_year_raw"]
    ac = tr["by_year_acf"][:, :, 1:]

    def parameters(x):
        aa, bb, r, b = x[:4]
        norm = np.sqrt(lags[0, 0] / response(lags[0], r, 5, 20, b, 1)[0])
        return MemoryCalibration(
            "distributed_memory",
            float(g.mean()),
            float(g.min()),
            float(g.max()),
            1.25,
            float(aa),
            float(bb),
            float(r),
            delay=5,
            last_lag=20,
            feedback=float(b),
            normalization=float(norm),
        )

    def loss(x):
        scale = np.exp(x[-1])
        p = parameters(x)
        tot, within, acf = projected(p, lags, g, nd, bg * scale**2)
        err = np.log(tot / target)
        err[:, -1] *= np.sqrt(month_w)
        return np.r_[
            err.ravel() / np.sqrt(4 + month_w),
            np.sqrt(center_w / 4) * np.log(within[:, :4] / tr["by_year_within"][:, :4]).ravel(),
            ((acf[:, :, 1:] - ac) / np.sqrt(60)).ravel(),
            np.sqrt(len(g) * regularization) * np.log(scale / cvgain),
        ]

    start = [init.intercept, init.slope, init.retention, init.feedback, np.log(cvgain)]
    result = least_squares(
        loss,
        start,
        bounds=([-0.7, -0.35, 0.65, 0.0, np.log(0.4)], [0.7, 0.35, 1.0, 1.2, np.log(1.4)]),
        max_nfev=150,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(np.exp(result.x[-1])), parameters(result.x)


def finite_member_variance(data, mask):
    x = data["mu"][mask]
    q = data["second"][mask]
    m = data["member"][mask]
    per = np.stack([x[m == v].mean(axis=0) for v in np.unique(m)])
    n = len(per)
    return variance(x, q) + ((per - per.mean(axis=0)) ** 2).sum(axis=(0, 2)) / (n * (n - 1))


SETTINGS = {
    "crossfit": ("cv", 0.5, 1.0, 0.0),
    "budget": ("budget", 1.0, 2.0, 0.0),
    "budget_cv": ("budget", 1.0, 2.0, 0.05),
    "budget_strongcv": ("budget", 1.0, 2.0, 0.3),
    "budget_month": ("budget", 1.0, 4.0, 0.02),
    "budget_centered": ("budget", 2.0, 1.0, 0.02),
}


def fit_transfer(bank, tr, va, phys, name, setting, export=None):
    r = CASES[name]
    month = bank.month
    m = build(bank, r)
    zs = bank.z(r, "train")
    gs = bank.scores["train_g"]
    nd = bank.scores["train_days"]
    daily = bank.daily(r, "train")
    paths = [daily[i, :n] for i, n in enumerate(nd)]
    base_coeff = m.mean.coefficients.copy()
    original = MonthlyModel(DEFAULT_WORK / "weights", month).cal
    m.cal = replace(original, center=float(gs.mean()), g_min=float(gs.min()), g_max=float(gs.max()))
    gt = year_table(tr)[1]
    _, gv, nv = year_table(va)

    def set_dynamics(gain):
        m.mean.coefficients = base_coeff.copy()
        m.mean.coefficients[2:] *= gain
        mean = m.mean.predict(m.bg, gs, zs)
        m.base, _, _ = fit_dynamics(
            paths, mean, {"lim_ridge_candidates": [r.lim_ridge, 0.001, 0.01, 0.1]}
        )

    set_dynamics(1.0)
    lags, bg, forced = model_terms(m, np.r_[gt, gv])
    cvgain, folds = cross_validated_gain(bank, r)
    mode, center_w, month_w, lam = SETTINGS[setting]
    steps = []
    if mode == "cv":
        gain = cvgain
    else:
        for j in range(3):
            gain, _ = fit_budget(
                tr, lags, bg[: len(gt)], m.cal, cvgain, mode, center_w, month_w, lam
            )
            steps.append(gain)
            set_dynamics(gain)
            lags = trace_lags(m.base, bank.G(r), 51)
            if len(steps) > 1 and abs(gain - steps[-2]) < 0.002:
                break
    set_dynamics(gain)
    lags = trace_lags(m.base, bank.G(r), 51)
    cal, loss = train_cal(
        m, tr, lags, bg[: len(gt)] * gain**2, 5, 20, center_w, False, 1.0, month_w
    )
    rows = validation(
        cal, va, lags, bg[len(gt) :] * gain**2, forced, -0.0115220616351514, 3.419637285227324
    )
    for group in ("all", "inside", "cold", "warm"):
        gsval = va["g"]
        mask = {
            "all": np.ones(len(gsval), bool),
            "cold": gsval < -0.0115220616351514,
            "warm": gsval > 3.419637285227324,
            "inside": (gsval >= -0.0115220616351514) & (gsval <= 3.419637285227324),
        }[group]
        corr = finite_member_variance(va, mask)
        for item in rows:
            if item["group"] == group and item["target"] == "total":
                j = ("1", "5", "10", "15", "month").index(item["window"])
                item["unbiased_truth"] = corr[j]
                item["unbiased_ratio"] = item["model"] / corr[j]
    detail = dict(
        recipe=asdict(r),
        gain=gain,
        member_cv_gain=cvgain,
        fold_gains=folds,
        gain_iterations=steps,
        cal=asdict(cal),
        loss=loss,
        model_radius=m.base.rho,
        lyapunov_residual=m.base.residual,
    )
    if export is not None:
        detail["id"] = save_model(
            export, m, bank, r, cal, setting, {"gain": gain, "settings": SETTINGS[setting]}
        )
    return pd.DataFrame(rows), phys.wind_errors(m, bank.z(r, "val")), detail


def run(cohort, month, export_name=None, setting=None, rich=False):
    bank = Bank(cohort, month)
    phys = PhysicalValidation(bank)
    tr = cache_read(cohort, "train", month)
    va = cache_read("original", "val", month)
    names = (
        [export_name]
        if export_name
        else (
            ["rich_compact", "rich_medium", "rich_marginals", "rich_slow160"]
            if rich
            else [
                "released_design",
                "compact_depth",
                "medium_depth",
                "wide_ocean",
                "wide_depth",
                "wide_slow160",
            ]
        )
    )
    for name in names:
        root = WORK / "transfer_budget" / cohort / name / f"{month:02}"
        if export_name is None and (root / "fits.json").exists():
            continue
        export = WORK / "exports" / cohort / (name + "__" + setting) if export_name else None
        if export is not None and (export / "fit" / f"{month:02}.json").exists():
            continue
        t = time.monotonic()
        rows = []
        winds = []
        meta = {}
        for key in [setting] if setting else SETTINGS:
            f, w, d = fit_transfer(bank, tr, va, phys, name, key, export)
            rows.append(f.assign(month=month, recipe=name, filter=key))
            winds.append(w.assign(filter=key))
            meta[key] = d
        if export_name is None:
            for label, frames in (("windows", rows), ("monthly_wind", winds)):
                with atomic(root / (label + ".parquet")) as tmp:
                    pd.concat(frames, ignore_index=True).to_parquet(
                        tmp, index=False, compression="zstd"
                    )
            save_json(root / "fits.json", meta)
        record(
            "transfer_budget_done",
            month=month,
            cohort=cohort,
            recipe=name,
            export=export_name is not None,
            variants=len(meta),
            seconds=round(time.monotonic() - t, 1),
        )
        print(cohort, month, name, len(meta), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--cohort", default="original")
    p.add_argument("--export", choices=CASES)
    p.add_argument("--filter", choices=SETTINGS)
    p.add_argument("--rich", action="store_true")
    a = p.parse_args()
    if a.export and not a.filter:
        p.error("Select the budget calibration to export")
    run(a.cohort, a.month, a.export, a.filter, a.rich)
