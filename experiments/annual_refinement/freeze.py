"""Freeze exactly one deployable candidate using only paired, twelve-month validation results."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from records import WORK, record

from annual_lim.layout import read
from annual_lim.model import MonthlyModel
from tc_emulator.common import ROOT, lock, save_json, sha


def freeze(tag, weights, cohort, rationale):
    weights = Path(weights).resolve()
    output = WORK / "FROZEN.json"
    with lock(WORK / "experiment_records/.freeze.lock"):
        if output.is_file():
            old = read(output)
            if old["stage"] != tag or Path(old["weights"]) != weights:
                raise ValueError(
                    "The independent test receipt is already sealed for a different fit"
                )
            return old
        comparison = WORK / "comparison" / tag / "val" / "paired.csv"
        table = pd.read_csv(comparison)
        # A scalar score has no field or horizon label. Preserve the CSV's empty label in JSON.
        table["dimension"] = table["dimension"].fillna("")
        ids = {}
        for m in range(1, 13):
            month = f"{m:02}"
            # Loading verifies the file and metadata hashes before signing the release.
            model = MonthlyModel(weights, m)
            score = read(WORK / "verification" / tag / "val" / month / "complete.json")
            pub = read(WORK / "verification/published_v1/val" / month / "complete.json")
            if (
                score["model_id"] != model.meta["id"]
                or score["members"] != pub["members"]
                or score["years"] != pub["years"]
                or score["role"] != "val"
                or model.base.rho >= 1
            ):
                raise ValueError(f"Candidate does not match the validation receipt in month {m}")
            ids[month] = model.meta["id"]
        chosen = table[
            (table.group == "all")
            & (
                (table.metric == "crps")
                | (table.metric == "monthly_wind_rmse")
                | (
                    (table.metric == "variance_log_error")
                    & table.dimension.isin(["total/multi", "within/multi", "total/15", "within/15"])
                )
            )
        ]
        evidence = chosen[
            ["metric", "dimension", "baseline", "candidate", "difference", "ci_low", "ci_high"]
        ].to_dict("records")
        receipt = dict(
            stage=tag,
            weights=str(weights),
            cohort=cohort,
            months=ids,
            utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            selection=rationale,
            baseline="published_v1",
            validation=comparison.relative_to(WORK).as_posix(),
            validation_sha256=sha(comparison),
            validation_evidence=evidence,
            code_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
            runtime_sha256={
                str(p.relative_to(ROOT)): sha(p)
                for p in (
                    *sorted((ROOT / "src/annual_lim").glob("*.py")),
                    *(
                        ROOT / "src/tc_emulator" / (module + ".py")
                        for module in (
                            "background",
                            "pure_lim",
                            "pure_lim_memory",
                            "representation",
                            "wind_lim",
                            "warming_circulation",
                            "fourier_weather",
                            "tc_adapter",
                        )
                    ),
                    *(
                        ROOT / "notebooks" / (module + ".py")
                        for module in (
                            "tc_benchmark",
                            "benchmark_mpi2050",
                            "prepare_mpi2050",
                            "score_mpi2050",
                        )
                    ),
                )
            },
            independent_years=read(Path(__file__).with_name("design.json"))["fresh_test_years"],
        )
        save_json(output, receipt)
    record("primary_frozen", tag=tag, cohort=cohort, months=len(ids), receipt_sha256=sha(output))
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--rationale", required=True)
    a = p.parse_args()
    freeze(a.tag, a.weights, a.cohort, a.rationale)
