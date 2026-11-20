"""Follow an equal-rank ocean-front change into monthly steering with common daily weather."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from backbone import Bank, build
from fullfit import CASES
from joint import PhysicalValidation
from ocean import GROUP_LIMITS, joint_input, ocean_data
from ocean_fronts import STRENGTHS, edges, front_basis, gradient_percentile
from paired_intervals import panel_weights, record_estimate
from records import WORK, record
from weather import cache_read, model_terms, projected, year_table

from annual_lim.layout import read
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic

HERE = WORK / "front_circulation"


def run(month):
    files = [HERE / f"{month:02}_{name}.parquet" for name in ("members", "years")]
    if all(f.is_file() for f in files):
        return
    bank = Bank("extended", month)
    tr, train = ocean_data("extended", month, "train")
    va, val = ocean_data("original", month, "val")
    if set(train.member) & set(val.member) or train.year.nunique() != 57 or len(val) != 42:
        raise ValueError("Unexpected development-validation partitions")
    for name, meta in (("train", train), ("val", val)):
        for key in ("member", "year", "g"):
            np.testing.assert_array_equal(bank.scores[name + "_" + key], meta[key].to_numpy())
    pts = edges(bank.geo.lat, bank.geo.lon)
    clim = np.nansum(tr, axis=0, dtype=float) / np.maximum(1, np.isfinite(tr).sum(axis=0))
    receipt = read(WORK / "FROZEN.json")
    frozen = MonthlyModel(receipt["weights"], month)
    if frozen.meta["id"] != receipt["months"][f"{month:02}"]:
        raise ValueError("Expected the frozen daily weather law")
    obs = cache_read("original", "val", month)
    np.testing.assert_array_equal(val.member, obs["member"])
    np.testing.assert_array_equal(val.year, obs["year"])
    years, gs, days = year_table(obs)
    validation = PhysicalValidation(bank, va, val, obs["mu"][:, -1])
    recipe = replace(CASES["medium_ocean"], sst_basis="front_sst", ocean_basis="front_ocean")
    np.testing.assert_allclose(
        bank.block(recipe.wind_basis).basis[:, : recipe.wind],
        frozen.rep.blocks["wind"].basis,
        atol=1.0e-12,
    )
    members, annual = [], []
    fixed_within = None
    for strength in STRENGTHS:
        for name, ids, k in (("sst", [0], 48), ("ocean", [1, 2], 72)):
            parent = bank.geo.blocks[name]
            pct = np.stack(
                [gradient_percentile(clim[i], parent.valid[j], pts) for j, i in enumerate(ids)]
            )
            train_fields = tr[:, :1] if name == "sst" else joint_input(tr)
            val_fields = va[:, :1] if name == "sst" else joint_input(va)
            block = front_basis(parent, train_fields, pct, k, strength)
            token = "front_" + name
            bank.bases[token] = block
            bank.scores["train_" + token] = block.transform(train_fields)
            bank.scores["val_" + token] = block.transform(val_fields)
        model = build(bank, recipe)
        model.base, model.cal = frozen.base, frozen.cal
        observed_z = bank.z(recipe, "val")
        w = validation.wind_errors(model, observed_z).assign(month=month, strength=strength)
        members.append(w)
        lags, bg, _ = model_terms(model, gs)
        total, within, _ = projected(model.cal, lags, gs, days, bg)
        if fixed_within is None:
            fixed_within = within.copy()
        else:
            np.testing.assert_allclose(within, fixed_within, rtol=1e-10, atol=1e-10)
        mu = model.mean.predict(model.bg, val.g.to_numpy(), observed_z)
        G = bank.G(recipe)[0]
        for i, (year, g) in enumerate(zip(years, gs)):
            pred = mu[val.year == year]
            delta = pred - pred.mean(axis=0)
            emp = float(np.sum((delta @ G) * delta) / (len(delta) - 1))
            target = float(obs["by_year_cov"][i, 3])
            if min(emp, target, bg[i, 0], total[i, 3]) <= 0:
                raise ValueError("Nonpositive fixed-climate variance")
            annual.append(
                dict(
                    month=month,
                    year=int(year),
                    g=float(g),
                    strength=strength,
                    modeled_background=bg[i, 0],
                    background_on_mpi=emp,
                    background_abs_log=abs(np.log(bg[i, 0] / emp)),
                    modeled_15day=total[i, 3],
                    mpi_15day=target,
                    wind15_abs_log=abs(np.log(total[i, 3] / target)),
                    within_15day=within[i, 3],
                )
            )
    for p, df in zip(files, (pd.concat(members, ignore_index=True), pd.DataFrame(annual))):
        with atomic(p) as tmp:
            df.to_parquet(tmp, index=False, compression="zstd")
    record(
        "front_circulation_validation_month",
        month=month,
        strengths=list(STRENGTHS),
        training_cases=len(tr),
    )


def summarize(out):
    paths = lambda name: [HERE / f"{m:02}_{name}.parquet" for m in range(1, 13)]
    cases = pd.concat([pd.read_parquet(p) for p in paths("members")], ignore_index=True)
    var = pd.concat([pd.read_parquet(p) for p in paths("years")], ignore_index=True)
    keys = ["member", "year", "g"]
    panel = cases.groupby(keys + ["strength"]).conditional_mse.mean().unstack("strength")
    meta = panel.index.to_frame(index=False)
    lo, hi = GROUP_LIMITS
    var["regime"] = np.where(var.g < lo, "cold", np.where(var.g > hi, "warm", "inside"))
    rows = []
    intervals = []
    for band in ("all", "cold", "inside", "warm"):
        w, nm, ny = panel_weights(
            meta.member.to_numpy(), meta.year.to_numpy(), meta.g.to_numpy(), band, 2000, 200305
        )
        observed = np.sqrt(w @ panel[0.0].to_numpy())
        for strength in STRENGTHS:
            selected = (
                var[var.strength.eq(strength)]
                if band == "all"
                else var[var.strength.eq(strength) & var.regime.eq(band)]
            )
            candidate = np.sqrt(w @ panel[strength].to_numpy())
            rows.append(
                dict(
                    regime=band,
                    strength=strength,
                    monthly_steering_rmse_ms=candidate[0],
                    background_abs_log=selected.background_abs_log.mean(),
                    wind15_abs_log=selected.wind15_abs_log.mean(),
                    members=nm,
                    years=ny,
                )
            )
            if strength:
                record_estimate(
                    intervals,
                    band,
                    "monthly_steering_rmse",
                    (str(strength),),
                    observed,
                    candidate,
                    nm,
                    ny,
                    "physical",
                )
    for name, df in (
        ("front_circulation_validation.csv", pd.DataFrame(rows)),
        ("front_circulation_wind_intervals.csv", pd.DataFrame(intervals)),
        ("front_circulation_years.csv", var),
    ):
        with atomic(out / name) as tmp:
            df.to_csv(tmp, index=False)
    with atomic(out / "front_circulation_cases.parquet") as tmp:
        cases.to_parquet(tmp, index=False, compression="zstd")
    record(
        "front_circulation_validation_complete",
        training_years=57,
        member_months=504,
        wind_rows=len(cases),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--month", type=int, choices=range(1, 13))
    mode.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.month is None:
        summarize(args.summary)
    else:
        run(args.month)
