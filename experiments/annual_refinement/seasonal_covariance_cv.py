"""Test calendar smoothing of the covariance transmitted to wind on 42 additional training years."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from covariance_laws import FIELDS
from records import WORK, record

from annual_lim.layout import CONFIG, DEFAULT_WORK, design
from annual_lim.model import MonthlyModel
from annual_lim.statistics import metrics
from tc_emulator.common import atomic
from tc_emulator.warming_circulation import transfer_matrix

HERE = WORK / "seasonal_covariance_cv"
TARGETS = ("all_background", "ocean")
METHODS = ("constant", "separate_months", "cyclic_0.25", "cyclic_1", "cyclic_4", "annual_slope")


def make_gram(month):
    dest = HERE / "grams" / f"{month:02}.npz"
    if dest.is_file():
        return
    baseline = MonthlyModel(DEFAULT_WORK / "weights", month)
    old = {int(y) for split, _, y, _ in design(CONFIG)[1] if split == "train"}
    source = WORK / "features/extended" / f"{month:02}/scores.npz"
    with np.load(source) as bank:
        use = ~np.isin(bank["train_year"], list(old))
        year, member, warming = (bank["train_" + k][use] for k in ("year", "member", "g"))
        z = np.column_stack([bank["train_published_" + k][use] for k in FIELDS])
    years, members = np.unique(year), np.unique(member)
    if (len(years), len(members)) != (42, 35):
        raise ValueError(
            "The training-only study requires all 42 added years and 35 training members"
        )
    order = np.lexsort((member, year))
    shape = (len(years), len(members))
    np.testing.assert_array_equal(
        year[order].reshape(shape), np.broadcast_to(years[:, None], shape)
    )
    np.testing.assert_array_equal(member[order].reshape(shape), np.broadcast_to(members, shape))
    g = warming[order].reshape(shape)
    np.testing.assert_allclose(g, np.broadcast_to(g[:, :1], shape))
    x = z[order].reshape(*shape, z.shape[1])
    wind = transfer_matrix(baseline.mean)
    geom = metrics(baseline.rep)[2][0]
    k_ocean = sum(baseline.rep.blocks[k].basis.shape[1] for k in ("sst", "ocean"))
    grams = []
    for end in (z.shape[1], k_ocean):
        signal = x[:, :, :end] @ wind[:end]
        # Group variance will subtract the mean of each particular member subset.
        gram = (signal @ geom) @ np.swapaxes(signal, 1, 2)
        subset = np.arange(3, 14)
        cmp = signal[0, subset]
        direct = np.sum((cmp - cmp.mean(axis=0)) @ geom * (cmp - cmp.mean(axis=0))) / (len(cmp) - 1)
        np.testing.assert_allclose(sample_variance(gram[0], subset), direct, atol=1e-10)
        grams.append(gram)
    with atomic(dest) as tmp:
        np.savez(tmp, gram=np.stack(grams, axis=1), year=years, member=members, g=g[:, 0])
    record("seasonal_variance_training_inputs", month=month, training_years=42, training_members=35)


def sample_variance(grams, members):
    n = len(members)
    chosen = np.take(np.take(grams, members, axis=-2), members, axis=-1)
    trace = np.trace(chosen, axis1=-2, axis2=-1)
    return (trace - chosen.sum(axis=(-2, -1)) / n) / (n - 1)


def laws(var, g, test_g):
    centered = g - g.mean()
    slopes = (np.log(var) * centered[None, :, None]).sum(axis=1) / (centered @ centered)
    difference = np.eye(12) - np.roll(np.eye(12), 1, axis=1)
    penalty = difference.T @ difference
    by_method = [np.zeros_like(slopes), slopes]
    by_method.extend(np.linalg.solve(np.eye(12) + k * penalty, slopes) for k in (0.25, 1, 4))
    by_method.append(np.broadcast_to(slopes.mean(axis=0), slopes.shape))
    b = np.stack(by_method)
    normalizer = np.exp(b[:, :, None, :] * centered[None, None, :, None]).mean(axis=2)
    mean = var.mean(axis=1)[None, :, :] / normalizer
    held = np.clip(test_g, g.min() - 1.25, g.max() + 1.25) - g.mean()
    return mean[:, :, None, :] * np.exp(b[:, :, None, :] * held[None, None, :, None]), b


def fit(lane):
    grams = []
    for month in range(1, 13):
        with np.load(HERE / "grams" / f"{month:02}.npz") as source:
            if not grams:
                years, g, members = (source[k] for k in ("year", "g", "member"))
            else:
                for expected, name in ((years, "year"), (g, "g"), (members, "member")):
                    np.testing.assert_array_equal(expected, source[name])
            grams.append(source["gram"])
    grams = np.stack(grams)
    ordered = np.argsort(g)
    middle = ordered[np.linspace(2, len(g) - 3, 10, dtype=int)]
    regimes = {
        "cold": (ordered[:10], ordered[10:]),
        "inside": (middle, ordered[~np.isin(ordered, middle)]),
        "warm": (ordered[-10:], ordered[:-10]),
    }
    axes = pd.MultiIndex.from_product(
        (METHODS, range(1, 13), TARGETS), names=["method", "month", "target"]
    )
    for rep in range(lane, 64, 4):
        dest = HERE / "splits" / f"{rep:02}.parquet"
        if dest.is_file():
            continue
        rng = np.random.default_rng(2026112000 + rep)
        member_order = rng.permutation(len(members))
        test_members, training_members = member_order[:7], member_order[7:]
        truth = sample_variance(grams, test_members)
        fits = {n: sample_variance(grams, training_members[:n]) for n in (14, 21, 28)}
        rows = []
        for regime, (outside, eligible) in regimes.items():
            test_years = rng.choice(outside, 5, replace=False)
            # Keep both fitting endpoints so the climate support is the same at all three data sizes.
            edges = eligible[[0, -1]]
            choices = rng.permutation(eligible[1:-1])
            for ny in (12, 20, 28):
                train = np.r_[edges, choices[: ny - 2]]
                if np.intersect1d(train, test_years).size:
                    raise ValueError("An evaluation year was reused in fitting")
                for nm, variance in fits.items():
                    pred, slopes = laws(variance[:, train], g[train], g[test_years])
                    held_truth = truth[:, test_years, :]
                    error = np.log(pred / held_truth[None, :, :, :])
                    results = axes.to_frame(index=False)
                    results["abs_log_error"] = np.abs(error).mean(axis=2).ravel()
                    results["signed_log_error"] = error.mean(axis=2).ravel()
                    results["log_variance_slope_K"] = slopes.ravel()
                    rows.append(
                        results.assign(replicate=rep, regime=regime, fit_years=ny, fit_members=nm)
                    )
        with atomic(dest) as tmp:
            pd.concat(rows, ignore_index=True).to_parquet(tmp, index=False, compression="zstd")
        record(
            "seasonal_variance_training_fold_done",
            replicate=rep,
            compared_training_splits=27,
            fit_methods=len(METHODS),
            held_members=7,
            held_years=5,
        )


def summarize(out):
    files = [HERE / "splits" / f"{rep:02}.parquet" for rep in range(64)]
    rows = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    identity = ["replicate", "regime", "fit_years", "fit_members", "target", "month"]
    ref = rows[rows.method.eq("separate_months")][identity + ["abs_log_error"]].rename(
        columns={"abs_log_error": "unpooled_error"}
    )
    rows = rows.merge(ref, on=identity, validate="many_to_one")
    rows["difference_from_separate"] = rows.abs_log_error - rows.unpooled_error
    cols = ["abs_log_error", "signed_log_error", "log_variance_slope_K", "difference_from_separate"]
    keys = ["regime", "fit_years", "fit_members", "target", "method"]
    split = rows.groupby(keys + ["replicate"], as_index=False)[cols].mean()
    split["fraction_beating_separate"] = split.difference_from_separate.lt(0).astype(float)
    report = split.groupby(keys, as_index=False)[cols + ["fraction_beating_separate"]].mean()
    out.mkdir(exist_ok=True, parents=True)
    with atomic(out / "seasonal_covariance_training_cv.csv") as tmp:
        report.assign(replicates=64).to_csv(tmp, index=False)
    with atomic(out / "seasonal_covariance_training_folds.parquet") as tmp:
        rows.to_parquet(tmp, index=False, compression="zstd")
    record("seasonal_variance_training_cv_complete", repeats=64, splits=1728, rows=len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("gram").add_argument("--month", type=int, required=True)
    sub.add_parser("fit").add_argument("--lane", type=int, required=True)
    sub.add_parser("summary").add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "gram":
        make_gram(args.month)
    elif args.mode == "fit":
        fit(args.lane)
    else:
        summarize(args.output)
