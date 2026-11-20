"""Export paired training-only fits together with their exact member and year lists."""
import argparse
from pathlib import Path

import pandas as pd
from records import WORK, record

from annual_lim.layout import read
from tc_emulator.common import atomic


def export(out, require_complete=False):
    rows = []
    totals = {}
    for study in ("crossfit", "learning_curve"):
        count = 0
        for path in sorted((WORK / study).glob("lane*/*.json")):
            data = read(path)
            fits = data["results"].values() if study == "learning_curve" else [data]
            for fit in fits:
                split = fit["split"]
                for axis in ("members", "years"):
                    if set(split["training_" + axis]) & set(split["held_" + axis]):
                        raise ValueError(f"Training and held {axis} overlap in {path}")
                if fit["train_cases"] != len(split["training_members"]) * len(
                    split["training_years"]
                ) or fit["held_cases"] != len(split["held_members"]) * len(split["held_years"]):
                    raise ValueError(f"Member-year rectangle is incomplete in {path}")
                if len(fit["results"]) != 2 or fit["results"][0]["arm"] != "reference":
                    raise ValueError(f"Missing paired reference in {path}")
                key = dict(
                    study=study,
                    partition=path.parent.name + "/" + path.stem,
                    cohort=fit["cohort"],
                    candidate=fit["results"][1]["arm"],
                    regime=fit["scheme"],
                    month=fit["month"],
                    seed=fit["seed"],
                    train_cases=fit["train_cases"],
                    held_cases=fit["held_cases"],
                    **{
                        k: split[k]
                        for k in (
                            "training_members",
                            "training_years",
                            "held_members",
                            "held_years",
                        )
                    },
                )
                for result in fit["results"]:
                    rows.append(dict(**key, **result))
                count += 1
        totals[study] = count
    frame = pd.DataFrame(rows)
    if require_complete:
        grid = frame[frame.study.eq("learning_curve") & frame.arm.eq("reference")]
        if set(grid.groupby("cohort").size()) != {6048} or grid.cohort.nunique() != 3:
            raise ValueError("At least one of the three 6,048-partition cohorts is incomplete")
        if not grid.groupby("partition").cohort.nunique().eq(3).all():
            raise ValueError("The three training cohorts must share every partition")
    with atomic(out / "training_partition_scores.parquet") as tmp:
        frame.to_parquet(tmp, index=False, compression="zstd")
    record(
        "training_partition_archive",
        paired_architecture_trials=totals["crossfit"],
        paired_learning_trials=totals["learning_curve"],
        individual_fit_rows=len(frame),
        complete_learning_grid=require_complete,
    )
    print(totals, "individual model fits", len(frame), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    export(args.output, args.require_complete)
