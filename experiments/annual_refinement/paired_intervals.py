"""Paired uncertainty from resampling whole MPI members and whole years across all 12 months."""
import argparse

import numpy as np
import pandas as pd
from ocean import GROUP_LIMITS
from records import WORK, record

from annual_lim.layout import read
from tc_emulator.common import atomic, save_json


def panel_weights(members, years, warming, group, draws, seed):
    """Independent bootstrap axes retain repeated months, paired drivers, and all weather windows."""
    unique_m, mi = np.unique(members, return_inverse=True)
    unique_y, idx, yi = np.unique(years, return_index=True, return_inverse=True)
    if len(unique_y) * len(unique_m) != len(years) or len(set(zip(mi, yi))) != len(years):
        raise ValueError("The crossed bootstrap requires a complete member by year grid")
    g = warming[idx]
    lo, hi = GROUP_LIMITS
    bins = np.where(g < lo, "cold", np.where(g > hi, "warm", "inside"))
    allowed = np.flatnonzero(np.ones(len(g), bool) if group == "all" else bins == group)
    if not len(allowed):
        raise ValueError(f"No years in {group}")
    rng = np.random.default_rng(seed)
    # Stratification keeps the original cold/inside/warm mix in the pooled score.
    mcount = rng.multinomial(len(unique_m), np.full(len(unique_m), 1 / len(unique_m)), size=draws)
    ycount = np.zeros((draws, len(unique_y)), dtype=int)
    for b in ("cold", "inside", "warm"):
        use = np.intersect1d(allowed, np.flatnonzero(bins == b))
        if len(use):
            ycount[:, use] = rng.multinomial(len(use), np.full(len(use), 1 / len(use)), size=draws)
    weights = mcount[:, mi] * ycount[:, yi]
    first = np.isin(yi, allowed).astype(float)
    weights = np.vstack((first, weights))
    return weights / weights.sum(axis=1, keepdims=True), len(unique_m), len(allowed)


def read_metric(root, name, months=range(1, 13)):
    parts = [
        pd.read_parquet(root / f"{m:02}" / (name + ".parquet")).assign(calendar_month=m)
        for m in months
    ]
    return pd.concat(parts, ignore_index=True)


def category_table(root, months):
    """Collapse the 12 months for individual additive scores before crossed resampling."""
    result = {}
    for file, keys, variables in (
        (
            "ocean",
            ["region", "field"],
            [
                ("crps", "crps", False),
                ("mean_mse", "mean_rmse", True),
                ("coverage90", "coverage90", False),
            ],
        ),
        (
            "projection",
            ["region", "field"],
            [("mse", "projection_rmse", True), ("bias", "projection_bias", False)],
        ),
        (
            "forecast",
            ["lead"],
            [
                ("lim_mse", "forecast_rmse", True),
                ("persistence_mse", "persistence_rmse", True),
                ("mean_mse", "forecast_mean_rmse", True),
            ],
        ),
        (
            "monthly_wind",
            [],
            [
                ("conditional_mse", "monthly_wind_rmse", True),
                ("forcing_only_mse", "forcing_only_rmse", True),
            ],
        ),
    ):
        data = read_metric(root, file, months)
        groups = data.groupby(keys) if keys else [((), data)]
        for item, frame in groups:
            if not isinstance(item, tuple):
                item = (item,)
            scores = frame.groupby(["member", "year"], sort=True)[[x[0] for x in variables]].mean()
            for col, metric, square in variables:
                result[(metric,) + tuple(map(str, item))] = (scores[col], square)
    return result


def record_estimate(rows, group, metric, dims, old, new, members, years, unit):
    delta = new - old
    pct = np.divide(100 * delta, np.where(np.abs(old) > 1.0e-10, old, np.nan))

    def interval(x):
        finite = x[1:][np.isfinite(x[1:])]
        return np.quantile(finite, [0.025, 0.975]) if len(finite) else [np.nan, np.nan]

    ci = interval(delta)
    rel = interval(pct) if unit != "signed" else [np.nan, np.nan]
    rows.append(
        dict(
            group=group,
            metric=metric,
            dimension="/".join(dims),
            members=members,
            years=years,
            baseline=old[0],
            candidate=new[0],
            difference=delta[0],
            ci_low=ci[0],
            ci_high=ci[1],
            basic_ci_low=2 * delta[0] - ci[1],
            basic_ci_high=2 * delta[0] - ci[0],
            change_percent=pct[0] if unit != "signed" else np.nan,
            pct_ci_low=rel[0],
            pct_ci_high=rel[1],
            valid_resamples=int(np.isfinite(delta[1:]).sum()),
        )
    )


def wind_estimates(paths, role, weights, months):
    """Use one Gram matrix per calendar month and window to avoid resampling grid cells."""
    windows = ("1", "5", "10", "15", "month")
    result = {}
    for month in months:
        with np.load(paths[0] / role / f"{month:02}/panel_moments.npz") as a, np.load(
            paths[1] / role / f"{month:02}/panel_moments.npz"
        ) as b:
            order = np.lexsort((a["year"], a["member"]))
            np.testing.assert_array_equal(a["member"], b["member"])
            np.testing.assert_array_equal(a["year"], b["year"])
            np.testing.assert_allclose(a["second"], b["second"], atol=0.0, rtol=0.0)
            mu = a["mu"][order]
            s2 = weights @ a["second"][order]
            actual_center = weights @ a["within"][order, :4]
            obs = np.empty_like(s2)
            for j in range(5):
                X = mu[:, j].astype(float)
                gram = X @ X.T
                obs[:, j] = s2[:, j] - np.einsum("bi,bi->b", weights @ gram, weights)
            g = a["g"][order]
            varg = weights @ (g * g) - (weights @ g) ** 2
            for k, model in enumerate((a, b)):
                total = weights @ model["predicted_total"][order] + varg[:, None] * float(
                    model["forced"]
                )
                within = weights @ model["predicted_within"][order, :4]
                for target, pred, truth in (
                    ("total", total, obs),
                    ("within", within, actual_center),
                ):
                    positive = truth > 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(pred))
                    ratio = np.divide(pred, truth, out=np.full_like(pred, np.nan), where=positive)
                    signed = np.log(ratio)
                    err = np.abs(signed)
                    for j in range(err.shape[1]):
                        result.setdefault((k, target, windows[j]), []).append(err[:, j])
                        result.setdefault((k, "ratio_" + target, windows[j]), []).append(
                            np.exp(signed[:, j])
                        )
                    result.setdefault((k, target, "multi"), []).append(err[:, 1:].mean(axis=1))
    return {k: np.mean(v, axis=0) for k, v in result.items()}


def compare(tag, role, draws, exclude_september=False):
    months = [m for m in range(1, 13) if not (exclude_september and m == 9)]
    suffix = "_excluding_september" if exclude_september else ""
    root = WORK / "verification"
    paths = [root / "published_v1", root / tag]
    receipts = [[read(p / role / f"{m:02}/complete.json") for m in range(1, 13)] for p in paths]
    if any(c["role"] != role for model in receipts for c in model):
        raise ValueError("Verification receipt does not match its role")
    result = [category_table(p / role, months) for p in paths]
    example = read_metric(paths[0] / role, "monthly_wind", months)
    grid = example.groupby(["member", "year"], sort=True).g.first()
    members, years = (
        grid.index.get_level_values(0).to_numpy(),
        grid.index.get_level_values(1).to_numpy(),
    )
    rows = []
    temps = grid.to_numpy()
    low, high = GROUP_LIMITS
    present = set(np.where(temps < low, "cold", np.where(temps > high, "warm", "inside")))
    for i, group in enumerate(("all", "cold", "inside", "warm")):
        if group != "all" and group not in present:
            continue
        weights, n_m, n_y = panel_weights(
            members, years, grid.to_numpy(), group, draws, 20261120 + i
        )
        for key, (old, sqrt) in result[0].items():
            new, sqrt_check = result[1][key]
            if (
                not old.index.equals(grid.index)
                or not new.index.equals(grid.index)
                or sqrt != sqrt_check
            ):
                raise ValueError(f"Member-years differ for {key}")
            x, y = weights @ old.to_numpy(), weights @ new.to_numpy()
            if sqrt:
                x, y = np.sqrt(x), np.sqrt(y)
            record_estimate(
                rows,
                group,
                key[0],
                key[1:],
                x,
                y,
                n_m,
                n_y,
                "signed" if "bias" in key[0] else "raw",
            )
        wind = wind_estimates(paths, role, weights, months)
        for k, target, d in wind:
            if k == 0:
                record_estimate(
                    rows,
                    group,
                    ("variance_ratio" if target.startswith("ratio_") else "variance_log_error"),
                    (target.removeprefix("ratio_"), d),
                    wind[0, target, d],
                    wind[1, target, d],
                    n_m,
                    n_y,
                    "raw",
                )
    dest = WORK / "comparison" / tag / role
    df = pd.DataFrame(rows)
    with atomic(dest / ("paired" + suffix + ".csv")) as tmp:
        df.to_csv(tmp, index=False)
    save_json(
        dest / ("method" + suffix + ".json"),
        dict(
            replicates=draws,
            calendar_months=months,
            interval=0.95,
            baseline="published_v1",
            candidate=tag,
            paired_axes=["member", "year"],
            strata=[name for name in ("cold", "inside", "warm") if name in present],
            same_draw_across_months=True,
            monthly_wind_variance="equal-weight annual mean of calendar-month absolute log errors",
            note="Intervals describe sensitivity on these MPI members and years. Existing labels are kept fixed; single-year temperature strata have no between-year uncertainty.",
        ),
    )
    record("paired_comparison_done", tag=tag, role=role, draws=draws, metrics=len(df))
    print(
        df[
            (df.group == "all")
            & df.metric.isin(["monthly_wind_rmse", "crps", "projection_rmse", "variance_log_error"])
        ][["metric", "dimension", "baseline", "candidate", "ci_low", "ci_high"]]
        .round(4)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--role", choices=["val", "test", "fresh", "ssp245_year"], required=True)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--exclude-september", action="store_true")
    args = p.parse_args()
    compare(args.candidate, args.role, args.draws, args.exclude_september)
