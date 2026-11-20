"""Summarize paired training-member/year sensitivity runs, keeping every complete result on disk."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from records import WORK, record

from tc_emulator.common import atomic


def summarize(out):
    rows = []
    for path in sorted((WORK / "crossfit").glob("lane*/*.json")):
        d = json.loads(path.read_text())
        a, b = d["results"]
        oc = [k for k in a if k.endswith("_crps")]
        rows.append(
            dict(
                lane=int(path.parent.name[4:]),
                trial=d["index"],
                cohort=d["cohort"],
                month=d["month"],
                arm=b["arm"],
                regime=d["scheme"],
                train_members=d["member_size"],
                seed=d["seed"],
                ocean_pct=100 * np.mean([b[k] / a[k] - 1 for k in oc]),
                steering_mean_pct=100
                * (b["conditional_wind_rmse"] / a["conditional_wind_rmse"] - 1),
                total_error_change=b["wind_total"] - a["wind_total"],
                within_error_change=b["wind_within"] - a["wind_within"],
                mld_crps_difference=b["mld_basin_crps"] - a["mld_basin_crps"],
                gamma_crps_difference=b["gamma_basin_crps"] - a["gamma_basin_crps"],
            )
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Missing completed refits")
    out = Path(out)
    with atomic(out / "training_cv.parquet") as tmp:
        frame.to_parquet(tmp, index=False, compression="zstd")
    fields = [
        "ocean_pct",
        "steering_mean_pct",
        "total_error_change",
        "within_error_change",
        "mld_crps_difference",
        "gamma_crps_difference",
    ]
    for name, groups in (
        ("overall", ["cohort", "arm"]),
        ("bands", ["cohort", "arm", "regime"]),
        ("size", ["cohort", "arm", "train_members"]),
    ):
        grouped = frame.groupby(groups)
        stat = grouped[fields].mean()
        stat["trials"] = grouped.size()
        for f in fields[:4]:
            stat[f + "_win_fraction"] = grouped[f].agg(lambda x: (x < 0).mean())
        with atomic(out / (name + ".csv")) as tmp:
            stat.to_csv(tmp)
    record(
        "training_cv_summary",
        trials=len(frame),
        cohorts=sorted(frame.cohort.unique()),
        path=str(out),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=WORK / "crossfit_summary")
    a = p.parse_args()
    summarize(a.output)
