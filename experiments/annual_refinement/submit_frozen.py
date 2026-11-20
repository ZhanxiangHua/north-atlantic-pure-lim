"""Launch both independent test panels, their paired comparisons, and the fixed 2050 replay."""
import argparse
import subprocess
from pathlib import Path

from records import WORK, record

from annual_lim.layout import read
from tc_emulator.common import ROOT, lock, save_json


def submit(script, args, dependencies, account):
    cmd = ["qsub", "-A", account, "-o", str(WORK / "logs")]
    if dependencies:
        cmd.extend(["-W", "depend=afterok:" + ":".join(dependencies)])
    if args:
        cmd.extend(["-v", ",".join(f"{k}={v}" for k, v in args.items())])
    cmd.append(str(Path(__file__).parent / (script + ".pbs")))
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def run(account):
    receipt = read(WORK / "FROZEN.json")
    done = WORK / "experiment_records/holdout_jobs.json"
    with lock(done.with_suffix(".lock")):
        ids = read(done) if done.is_file() else {}

        def add(key, script, args, dependencies=()):
            if key not in ids:
                ids[key] = submit(script, args, dependencies, account)
                save_json(done, ids)
                record("holdout_job_submitted", job=ids[key], stage=key, primary=receipt["stage"])

        for role in ("test", "fresh", "ssp245_year"):
            add(
                "released_regions_" + role,
                "marine",
                {"MODEL_TAG": "published_v1", "EVAL_ROLE": role},
            )
            add(
                "selected_regions_" + role,
                "marine",
                {
                    "MODEL_TAG": receipt["stage"],
                    "MODEL_WEIGHTS": receipt["weights"],
                    "EVAL_ROLE": role,
                },
            )
            add(
                "released_" + role,
                "verify",
                {"MODEL_TAG": "published_v1", "EVAL_ROLE": role, "N_DRAWS": 32},
            )
            add(
                "selected_" + role,
                "verify",
                {
                    "MODEL_TAG": receipt["stage"],
                    "MODEL_WEIGHTS": receipt["weights"],
                    "COHORT": receipt["cohort"],
                    "EVAL_ROLE": role,
                    "N_DRAWS": 32,
                },
            )
            add(
                "uncertainty_" + role,
                "intervals",
                {"MODEL_TAG": receipt["stage"], "EVAL_ROLE": role},
                [ids["released_" + role], ids["selected_" + role]],
            )
            add(
                "released_structure_" + role,
                "spatial",
                {"MODEL_TAG": "published_v1", "EVAL_ROLE": role},
            )
            add(
                "selected_structure_" + role,
                "spatial",
                {
                    "MODEL_TAG": receipt["stage"],
                    "MODEL_WEIGHTS": receipt["weights"],
                    "EVAL_ROLE": role,
                },
            )
            add(
                "native_controls_" + role,
                "controls",
                {
                    "MODEL_TAG": receipt["stage"],
                    "MODEL_WEIGHTS": receipt["weights"],
                    "EVAL_ROLE": role,
                },
                [ids["released_" + role]],
            )
            add(
                "released_budget_" + role,
                "attribution",
                {"MODEL_TAG": "published_v1", "EVAL_ROLE": role},
                [ids["released_" + role]],
            )
            add(
                "selected_budget_" + role,
                "attribution",
                {
                    "MODEL_TAG": receipt["stage"],
                    "MODEL_WEIGHTS": receipt["weights"],
                    "EVAL_ROLE": role,
                },
                [ids["released_" + role]],
            )
        for role in ("val", "test", "fresh", "ssp245_year"):
            settings = {
                "MODEL_TAG": receipt["stage"],
                "MODEL_WEIGHTS": receipt["weights"],
                "COHORT": receipt["cohort"],
                "EVAL_ROLE": role,
            }
            add("leadlag_" + role, "leadlag", settings)
            add("leadlag_summary_" + role, "leadlag_summary", settings, [ids["leadlag_" + role]])
            add(
                "two_day_forecasts_" + role,
                "forecast_controls",
                settings,
                [] if role == "val" else [ids["selected_" + role]],
            )
        annual_drivers = {
            "mpi_daily": "MPI",
            "published_v1": "LIM",
            "native_fourier": "Fourier",
            receipt["stage"]: "LIM",
            "selected_fourier": "Fourier",
        }
        for tag, driver in annual_drivers.items():
            settings = {"MODEL_TAG": tag, "WIND_DRIVER": driver, "EVAL_ROLE": "ssp245_year"}
            if tag in (receipt["stage"], "selected_fourier"):
                settings["MODEL_WEIGHTS"] = receipt["weights"]
            add("annual_storms_" + tag, "storms", settings)
        add(
            "annual_storm_scores",
            "storm_summary",
            {"MODEL_TAG": receipt["stage"], "EVAL_ROLE": "ssp245_year"},
            [ids["annual_storms_" + tag] for tag in annual_drivers],
        )
        add("ssp245", "external", {})
        val_settings = {
            "MODEL_TAG": receipt["stage"],
            "MODEL_WEIGHTS": receipt["weights"],
            "EVAL_ROLE": "val",
        }
        add("validated_regions", "marine", val_settings)
        add("validated_structure", "spatial", val_settings)
        add("validated_budget", "attribution", val_settings)
        add("validated_fourier", "controls", val_settings)
        add("checkpoint", "pack", {})
        add("independent_retraining", "reproduce", {})
        add(
            "matched_seasons",
            "storms",
            {
                "MODEL_TAG": receipt["stage"],
                "MODEL_WEIGHTS": receipt["weights"],
                "WIND_DRIVER": "LIM",
            },
        )
        add(
            "seasonal_storms",
            "storm_summary",
            {"MODEL_TAG": receipt["stage"]},
            [ids["matched_seasons"]],
        )
        terminal = [
            "independent_retraining",
            "checkpoint",
            "seasonal_storms",
            "annual_storm_scores",
            "ssp245",
            "validated_regions",
            "validated_structure",
            "validated_budget",
            "validated_fourier",
            "leadlag_summary_val",
            "two_day_forecasts_val",
        ]
        for role in ("test", "fresh", "ssp245_year"):
            for key in (
                "uncertainty",
                "native_controls",
                "released_regions",
                "selected_regions",
                "released_structure",
                "selected_structure",
                "released_budget",
                "selected_budget",
                "leadlag_summary",
                "two_day_forecasts",
            ):
                terminal.append(key + "_" + role)
        if not (WORK / "experiment_records/infill_audit.json").is_file():
            add("raw_infill_audit", "audit_infill", {})
            terminal.append("raw_infill_audit")
        add("portable_report", "report", {}, [ids[k] for k in terminal])
        add("executed_tutorials", "tutorial", {}, [ids["portable_report"]])
    return ids


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--account", required=True)
    print(run(p.parse_args().account))
