"""Measure paired CPU time of the full and compact NumPy weather propagation."""
import argparse
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from records import WORK, record
from threadpoolctl import threadpool_info, threadpool_limits

from annual_lim.layout import read
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic


def time_month(month, full, compact):
    models = {
        "primary_240": MonthlyModel(full, month),
        "compact_192": MonthlyModel(compact, month),
    }
    if (len(models["primary_240"].base.a), len(models["compact_192"].base.a)) != (
        240,
        192,
    ):
        raise ValueError("Timing requires the two verified wind sizes")
    g = np.full(64, 2.0)
    rows = []
    with threadpool_limits(limits=1):
        libraries = ",".join(
            sorted(
                {
                    t["internal_api"]
                    for t in threadpool_info()
                    if t["user_api"] == "blas"
                }
            )
        )
        rng = np.random.default_rng(73571 + month)
        for repeat in range(-1, 7):
            names = list(models)
            if (month + repeat) % 2:
                names.reverse()
            for position, name in enumerate(names):
                model = models[name]
                noise = rng.normal(size=(64, 31, len(model.base.a)))
                history = np.random.default_rng(1183 + repeat)
                start = perf_counter()
                weather = model.weather.propagate(g, noise, history)
                elapsed = perf_counter() - start
                if weather.shape != (64, 31, 120) or not np.isfinite(weather).all():
                    raise ValueError(
                        "The timing run did not produce complete daily winds"
                    )
                if repeat >= 0:
                    rows.append(
                        dict(
                            month=month,
                            repeat=repeat,
                            order=position,
                            model=name,
                            seconds=elapsed,
                            paths=len(g),
                            days=31,
                            filter_history_days=model.cal.last_lag,
                            state=len(model.base.a),
                            blas_threads=1,
                            blas=libraries,
                            node=platform.node(),
                            matrix_bytes=sum(
                                x.nbytes
                                for x in (model.base.a, model.base.q, model.base.sigma)
                            ),
                        )
                    )
    with atomic(WORK / "compact_timing" / f"{month:02}.csv") as tmp:
        pd.DataFrame(rows).to_csv(tmp, index=False)
    record("compact_runtime_timing_month", month=month, pairs=7, paths_per_call=64)


def summarize(dest):
    raw = pd.concat(
        [pd.read_csv(WORK / "compact_timing" / f"{mo:02}.csv") for mo in range(1, 13)],
        ignore_index=True,
    )
    pair = raw.pivot(
        index=["month", "repeat", "node"], columns="model", values="seconds"
    )
    pair["compact_over_full"] = pair.compact_192 / pair.primary_240
    summary = (
        pair.reset_index()
        .groupby("month")
        .agg(
            full_median_s=("primary_240", "median"),
            compact_median_s=("compact_192", "median"),
            median_ratio=("compact_over_full", "median"),
            paired_runs=("repeat", "count"),
        )
    )
    raw.to_csv(dest / "compact_runtime_measurements.csv", index=False)
    summary.to_csv(dest / "compact_runtime_by_month.csv")
    record(
        "compact_runtime_timing_complete",
        pairs=len(pair),
        median_ratio=float(pair.compact_over_full.median()),
    )
    print(
        pair.compact_over_full.describe(percentiles=[0.25, 0.5, 0.75]).to_string(),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--month", type=int, choices=range(1, 13))
    group.add_argument("--summary", type=Path)
    p.add_argument("--full", type=Path)
    p.add_argument(
        "--compact",
        type=Path,
        default=WORK / "compact_ocean_confirmation/deployment/weights",
    )
    args = p.parse_args()
    if args.month:
        time_month(
            args.month,
            args.full or Path(read(WORK / "FROZEN.json")["weights"]),
            args.compact,
        )
    else:
        summarize(args.summary)
