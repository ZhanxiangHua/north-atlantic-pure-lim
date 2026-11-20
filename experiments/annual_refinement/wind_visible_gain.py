"""Test a seasonal background variance gain on the released wind regression and weather law."""
import argparse
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from backbone import Bank
from fullfit import CASES
from joint import PhysicalValidation
from ocean import GROUP_LIMITS, ocean_data
from records import WORK, record
from scipy.linalg import orth
from weather import cache_read, model_terms, projected, year_table

from annual_lim.layout import read
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic
from tc_emulator.warming_circulation import transfer_matrix

HERE = WORK / "wind_visible_gain"
METHODS = ("as_fitted", "constant", "monthly", "cyclic_one", "annual")


def linked_factor(factor, transfer, gain):
    """Scale the complete transmitted covariance by gain while preserving the wind-null directions."""
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("A variance gain must be positive")
    if gain == 1:
        return factor
    visible = orth(factor.T @ transfer, rcond=1e-12)
    return factor + (np.sqrt(gain) - 1) * (factor @ visible) @ visible.T


def moments(month):
    dest = HERE / f"moments_{month:02}.csv"
    if dest.is_file():
        return
    receipt = read(WORK / "FROZEN.json")
    spec = read(WORK / "recipe.json")
    model = MonthlyModel(receipt["weights"], month)
    bank = Bank("extended", month)
    recipe = CASES[spec["name"]]
    native = bank.representation(recipe)
    for key, eof in native.blocks.items():
        other = model.rep.blocks[key]
        np.testing.assert_array_equal(eof.valid, other.valid)
        for attr in ("mean", "weights", "basis"):
            np.testing.assert_allclose(
                getattr(eof, attr), getattr(other, attr), atol=1.0e-15, rtol=0
            )
    K = transfer_matrix(model.mean)
    G = bank.G(recipe)[0]
    rows = []
    for role in ("train", "val"):
        z = bank.z(recipe, role)
        years = bank.scores[role + "_year"]
        warming = bank.scores[role + "_g"]
        number = 35 if role == "train" else 7
        for year in np.unique(years):
            sel = years == year
            if sel.sum() != number:
                raise ValueError("Incomplete population for the climate year")
            g = float(warming[sel][0])
            placed = z[sel] @ K
            d = placed - placed.mean(axis=0)
            empirical = np.sum((d @ G) * d) / (number - 1)
            D = model.factor(g).T @ K
            predicted = np.sum((D @ G) * D)
            rows.append(
                dict(
                    month=month,
                    role=role,
                    year=int(year),
                    g=g,
                    members=number,
                    observed_transmitted=empirical,
                    model_transmitted=predicted,
                )
            )
    with atomic(dest) as tmp:
        pd.DataFrame(rows).to_csv(tmp, index=False)
    record("wind_visible_gain_training_moments", month=month, training_years=57, validation_years=6)


def fit_laws(frame):
    training = frame[frame.role.eq("train")]
    expected = training.pivot(index="month", columns="year", values="model_transmitted")
    observed = training.pivot(index="month", columns="year", values="observed_transmitted").loc[
        expected.index, expected.columns
    ]
    g = training.groupby("year").g.first().loc[expected.columns].to_numpy()
    old = expected.to_numpy()
    new = observed.to_numpy()
    x = g - g.mean()
    slope = np.log(new / old) @ x / (x @ x)
    D = np.eye(12) - np.roll(np.eye(12), 1, axis=1)
    b = np.stack(
        [
            np.zeros(12),
            np.zeros(12),
            slope,
            np.linalg.solve(np.eye(12) + D.T @ D, slope),
            np.full(12, slope.mean()),
        ]
    )
    a = new.sum(axis=1)[None, :] / (old[None, :, :] * np.exp(b[:, :, None] * x)).sum(axis=2)
    a[0] = 1.0
    np.testing.assert_allclose(
        (old[None, :, :] * a[1:, :, None] * np.exp(b[1:, :, None] * x)).mean(axis=2),
        np.broadcast_to(new.mean(axis=1), (4, 12)),
        rtol=1e-12,
    )
    return pd.DataFrame(
        [
            dict(
                method=method,
                month=m + 1,
                amplitude=a[j, m],
                slope_per_K=b[j, m],
                center_K=g.mean(),
                lower=g.min() - 1.25,
                upper=g.max() + 1.25,
            )
            for j, method in enumerate(METHODS)
            for m in range(12)
        ]
    )


def score(month, moments_frame, laws):
    receipt = read(WORK / "FROZEN.json")
    model = MonthlyModel(receipt["weights"], month)
    K = transfer_matrix(model.mean)
    data, meta = ocean_data("original", month, "val")
    obs = cache_read("original", "val", month)
    np.testing.assert_array_equal(meta.member, obs["member"])
    np.testing.assert_array_equal(meta.year, obs["year"])
    bank = SimpleNamespace(geo=model.rep, month=month)
    validation = PhysicalValidation(bank, data, meta, obs["mu"][:, -1])
    years, g, days = year_table(obs)
    target = (
        moments_frame[moments_frame.role.eq("val") & moments_frame.month.eq(month)]
        .set_index("year")
        .loc[years]
    )
    lags, baseline, _ = model_terms(model, g)
    np.testing.assert_allclose(baseline[:, 0], target.model_transmitted, rtol=1e-7)
    total, _, _ = projected(model.cal, lags, g, days, baseline)
    weather = total[:, 3] - baseline[:, 0]
    ocean = []
    wind = []
    for row in laws[laws.month.eq(month)].itertuples(index=False):
        gain = row.amplitude * np.exp(
            row.slope_per_K * (np.clip(g, row.lower, row.upper) - row.center_K)
        )
        lookup = {}
        for gi, si in zip(g, gain):
            F = model.factor(gi)
            revised = linked_factor(F, K, si)
            before = F.T @ K
            after = revised.T @ K
            np.testing.assert_allclose(
                after.T @ after, si * (before.T @ before), atol=1e-10, rtol=1e-9
            )
            lookup[float(gi)] = revised
        alt = copy(model)
        alt.factor = lookup.__getitem__
        scores = validation.ocean_scores(alt)
        ocean.append(scores[scores.region.eq("basin")].assign(method=row.method, month=month))
        bg = baseline[:, 0] * gain
        wind.append(
            pd.DataFrame(
                dict(
                    method=row.method,
                    year=years,
                    month=month,
                    g=g,
                    gain=gain,
                    transmitted=bg,
                    transmitted_target=target.observed_transmitted.to_numpy(),
                    background_abs_log=np.abs(np.log(bg / target.observed_transmitted.to_numpy())),
                    wind15_total=bg + weather,
                    wind15_mpi=obs["by_year_cov"][:, 3],
                    wind15_abs_log=np.abs(np.log((bg + weather) / obs["by_year_cov"][:, 3])),
                )
            )
        )
    return pd.concat(ocean, ignore_index=True), pd.concat(wind, ignore_index=True)


def summarize(out):
    frame = pd.concat(
        [pd.read_csv(HERE / f"moments_{m:02}.csv") for m in range(1, 13)], ignore_index=True
    )
    laws = fit_laws(frame)
    omega = []
    winds = []
    for m in range(1, 13):
        o, w = score(m, frame, laws)
        omega.append(o)
        winds.append(w)
    ocean = pd.concat(omega, ignore_index=True)
    wind = pd.concat(winds, ignore_index=True)

    def grouped(data, keys, values):
        lo, hi = GROUP_LIMITS
        groups = np.where(data.g < lo, "cold", np.where(data.g > hi, "warm", "inside"))
        parts = pd.concat(
            [data.assign(regime=groups), data.assign(regime="all")], ignore_index=True
        )
        return parts.groupby(["method", "regime", *keys])[values].mean().reset_index()

    for name, table in (
        ("fits", laws),
        ("yearly", wind),
        ("training_targets", frame[frame.role.eq("train")]),
        ("wind_validation", grouped(wind, [], ["background_abs_log", "wind15_abs_log"])),
        ("ocean_validation", grouped(ocean, ["field"], ["crps", "coverage90"])),
    ):
        with atomic(out / f"wind_visible_gain_{name}.csv") as tmp:
            table.to_csv(tmp, index=False)
    with atomic(out / "wind_visible_gain_ocean_cases.parquet") as tmp:
        ocean.to_parquet(tmp, index=False, compression="zstd")
    record(
        "wind_visible_gain_validation_complete",
        training_years=57,
        validation_cases=504,
        methods=list(METHODS),
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
        moments(args.month)
