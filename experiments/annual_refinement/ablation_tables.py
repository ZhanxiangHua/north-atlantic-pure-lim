"""Summarize held-out ocean reconstruction at equal rank, field, area and warming."""
from pathlib import Path

import numpy as np
import pandas as pd
from ocean import GROUP_LIMITS
from records import WORK

from tc_emulator.common import atomic


def summarize(paths, dimensions):
    """Accumulate sums per month before aggregating so large screens fit in memory."""
    parts = []
    for source in sorted(paths):
        frame = pd.read_parquet(source)
        low, high = GROUP_LIMITS
        bands = np.where(frame.g < low, "cold", np.where(frame.g > high, "warm", "inside"))
        for warming in ("all", "cold", "inside", "warm"):
            frame_part = frame if warming == "all" else frame.loc[bands == warming]
            grouped = frame_part.groupby(dimensions, observed=True, dropna=False)
            sums = grouped[["mse", "bias", "tail_sse", "tail_weight", "coverage"]].sum()
            sums["cases"] = grouped.size()
            if "grid_crps" in frame_part:
                sums["grid_crps"] = grouped.grid_crps.sum()
            parts.append(sums.reset_index().assign(warming=warming))
    if not parts:
        raise ValueError("Missing ocean validation trials")
    data = pd.concat(parts, ignore_index=True)
    sums = data.groupby([*dimensions, "warming"], as_index=False, observed=True, dropna=False).sum(
        numeric_only=True
    )
    sums["rmse"] = np.sqrt(sums.pop("mse") / sums.cases)
    sums["upper_decile_rmse"] = np.sqrt(sums.pop("tail_sse") / sums.pop("tail_weight"))
    sums["bias"] /= sums.cases
    sums["grid_fraction"] = sums.pop("coverage") / sums.cases
    if "grid_crps" in sums:
        sums["grid_crps"] /= sums.cases
    return sums


def ablations(root):
    root = Path(root)
    matrices = [pd.read_csv(p) for p in sorted((WORK / "covariance_laws").glob("*.csv"))]
    if matrices:
        with atomic(root / "fixed_transfer_covariance_validation.csv") as tmp:
            pd.concat(matrices, ignore_index=True).to_csv(tmp, index=False)
    residuals = [pd.read_parquet(p) for p in sorted((WORK / "ocean_residual").glob("*/*.parquet"))]
    if residuals:
        residual = pd.concat(residuals, ignore_index=True)
        residual["warming"] = np.where(
            residual.g < GROUP_LIMITS[0],
            "cold",
            np.where(residual.g > GROUP_LIMITS[1], "warm", "inside"),
        )
        residual = pd.concat([residual, residual.assign(warming="all")], ignore_index=True)
        group = ["tag", "cohort", "arm", "field", "region", "warming"]
        summary = residual.groupby(group, as_index=False).agg(
            crps=("crps", "mean"),
            coverage90=("coverage90", "mean"),
            mean_mse=("mean_mse", "mean"),
            extra_fraction=("extra_fraction", "mean"),
            log_variance_slope=("training_log_variance_slope", "mean"),
            months=("month", "nunique"),
            member_month_years=("member", "size"),
        )
        summary["mean_rmse"] = np.sqrt(summary.pop("mean_mse"))
        with atomic(root / "unresolved_ocean_variance_validation.csv") as tmp:
            summary.to_csv(tmp, index=False)
    for name, paths, dimensions in [
        (
            "ocean_rank_validation",
            (WORK / "ocean").glob("*/*/val/*.parquet"),
            ["cohort", "layout", "parameter", "rank", "field", "region"],
        ),
        (
            "ocean_training_years",
            (WORK / "features/expanded").glob("*/rich_projection.parquet"),
            ["cohort", "train", "basis", "rank", "field", "region"],
        ),
        (
            "ocean_transform_validation",
            (WORK / "ocean_alternatives/original").glob("*/scores.parquet"),
            [
                "mld_power",
                "gamma_sqrt",
                "depth_weight",
                "local_fraction",
                "rank",
                "field",
                "region",
            ],
        ),
    ]:
        with atomic(root / (name + ".csv")) as tmp:
            summarize(paths, dimensions).to_csv(tmp, index=False)
    gates = []
    for p in sorted((WORK / "calendar_ocean").glob("*/validation.csv")):
        q = pd.read_csv(p)
        gates.append(
            q[
                [
                    "cohort",
                    "name",
                    "filter",
                    "comparison",
                    "eligible",
                    "difference_ocean_ratio",
                    "difference_mean_wind_ratio",
                    "difference_total_ratio",
                    "difference_within_ratio",
                ]
            ]
        )
    if gates:
        with atomic(root / "neighboring_months_validation.csv") as tmp:
            pd.concat(gates, ignore_index=True).to_csv(tmp, index=False)

    for name in ("windows", "direction"):
        frames = []
        for path in sorted((WORK / "lag_calibration").glob("*/*/" + name + ".parquet")):
            data = pd.read_parquet(path).assign(
                tag=path.parent.parent.name, month=int(path.parent.name)
            )
            frames.append(data)
        if frames:
            data = pd.concat(frames, ignore_index=True)
            keys = ["target", "window"] if name == "windows" else ["metric", "region", "lag"]
            value = "abs_log_error" if name == "windows" else "error"
            stats = data.groupby(["tag", "method", "group", *keys], as_index=False)[value].mean()
            with atomic(root / ("training_directional_loss_" + name + ".csv")) as tmp:
                stats.to_csv(tmp, index=False)
