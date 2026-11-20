"""Compare only complete, twelve-month validation recipes against the actual released model."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from records import WORK, record

from tc_emulator.common import atomic


def ocean_stats(frame):
    means = frame.groupby(["region", "field"]).crps.mean()
    return {f"{f}_{r}_crps": v for (r, f), v in means.items()}


def window_stats(frame):
    output = {}
    for task in ("total", "within"):
        d = frame[(frame.target == task) & (frame.window != "1")]
        for group in ("all", "cold", "inside", "warm"):
            w = d[d.group == group]
            output[f"{task}_{group}_logerr"] = w.abs_log_error.mean()
            if group == "all":
                output[f"{task}_15d_logerr"] = w[w.window == "15"].abs_log_error.mean()
                output[f"{task}_5d_logerr"] = w[w.window == "5"].abs_log_error.mean()
                if task == "total":
                    output["full_month_logerr"] = w[w.window == "month"].abs_log_error.mean()
    return output


def load_set(paths, filename, columns=None):
    return pd.concat(
        [
            pd.read_parquet(p.parent / filename, columns=columns).assign(
                calendar_month=int(p.parent.name)
            )
            for p in paths
        ],
        ignore_index=True,
    )


def acquire():
    rows = []
    for cohort in ("original", "expanded", "extended"):
        for path in sorted((WORK / "coupled" / cohort).glob("*")):
            complete = sorted(path.glob("*/metrics.json"))
            if len(complete) != 12:
                continue
            ocean = ocean_stats(load_set(complete, "ocean.parquet", ["region", "field", "crps"]))
            w = load_set(complete, "monthly_wind.parquet", ["conditional_mse"])
            wind = float(np.sqrt(w.conditional_mse.mean()))
            windows = load_set(complete, "windows.parquet")
            for key, df in windows.groupby("filter"):
                rows.append(
                    dict(
                        cohort=cohort,
                        name=path.name,
                        filter=key,
                        stage="coupled",
                        conditional_wind_rmse=wind,
                        **ocean,
                        **window_stats(df),
                    )
                )
            if cohort == "original" and path.name == "released_design":
                ref = pd.concat(
                    [
                        pd.read_parquet(p)
                        for p in sorted(
                            (WORK / "weather/published_backbone/screen").glob("*.parquet")
                        )
                    ]
                )
                rows.append(
                    dict(
                        cohort="release",
                        name="published_v1",
                        filter="as_fitted",
                        stage="frozen",
                        conditional_wind_rmse=wind,
                        **ocean,
                        **window_stats(ref[ref.recipe == "as_fitted"]),
                    )
                )
            bp = WORK / "transfer_budget" / cohort / path.name
            budget = sorted(bp.glob("*/fits.json"))
            if len(budget) == 12:
                bw = load_set(budget, "monthly_wind.parquet", ["filter", "conditional_mse"])
                wf = np.sqrt(bw.groupby("filter").conditional_mse.mean())
                for key, df in load_set(budget, "windows.parquet").groupby("filter"):
                    rows.append(
                        dict(
                            cohort=cohort,
                            name=path.name,
                            filter=key,
                            stage="budget",
                            conditional_wind_rmse=wf.loc[key],
                            **ocean,
                            **window_stats(df),
                        )
                    )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if not (frame.cohort == "release").any():
        return pd.DataFrame()
    base = frame[frame.cohort == "release"].iloc[0]
    ocean = [c for c in frame if c.endswith("_crps")]
    frame["ocean_ratio"] = frame[ocean].div(base[ocean].astype(float)).mean(axis=1)
    for name, key in (
        ("mean_wind_ratio", "conditional_wind_rmse"),
        ("total_ratio", "total_all_logerr"),
        ("within_ratio", "within_all_logerr"),
    ):
        frame[name] = frame[key] / base[key]
    frame["relative_composite"] = frame[
        ["ocean_ratio", "mean_wind_ratio", "total_ratio", "within_ratio"]
    ].mean(axis=1)
    frame["all_four_better"] = (
        frame[["ocean_ratio", "mean_wind_ratio", "total_ratio", "within_ratio"]] < 1.0
    ).all(axis=1)
    return frame.sort_values(["relative_composite", "name", "filter"])


def main(out=None):
    frame = acquire()
    if frame.empty:
        return
    out = Path(out) if out else WORK / "scoreboard.csv"
    with atomic(out) as tmp:
        frame.to_csv(tmp, index=False)
    record(
        "scoreboard_written",
        recipes=len(frame),
        best=frame.iloc[0][["cohort", "name", "filter", "relative_composite"]].to_dict(),
        all_four_improve=int(frame.all_four_better.sum()),
        file=str(out),
    )
    print(
        frame[
            [
                "cohort",
                "name",
                "filter",
                "ocean_ratio",
                "mean_wind_ratio",
                "total_ratio",
                "within_ratio",
            ]
        ]
        .head(8)
        .round(3)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output")
    main(p.parse_args().output)
