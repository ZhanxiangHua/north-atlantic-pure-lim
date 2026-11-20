"""Collect portable scientific tables and the timestamped ledger after independent verification."""
import argparse
import gzip
import json
import shutil
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from ablation_tables import ablations
from forecast_controls import intervals as forecast_intervals
from ocean import GROUP_LIMITS, data_paths
from records import WORK
from scoreboard import acquire
from storm_archive import archive as archive_storms

from annual_lim.layout import case_dir, design, read
from tc_emulator.common import ROOT, atomic, save_json


def write(frame, path):
    with atomic(path) as tmp:
        frame.to_csv(tmp, index=False)


def copy_table(source, dest):
    if not source.is_file():
        raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def source_manifest(root):
    rows = []
    members = {}
    for label, cohort, split in (
        ("15 paired training years", "original", "train"),
        ("30 paired training years", "expanded", "train"),
        ("57 paired training years", "extended", "train"),
        ("27 added ocean years", "wind_infill", "train"),
        ("validation", "original", "val"),
        ("original holdout", "original", "test"),
        ("new annual holdout", "fresh", "test"),
        ("SSP2-4.5 annual replay", "ssp245_year", "test"),
    ):
        cfg, store = data_paths(cohort)
        grid = pd.DataFrame(
            [c for c in design(cfg)[1] if c[0] == split], columns=["split", "member", "year", "g"]
        )
        members[label] = sorted(grid.member.unique().tolist())
        for year, frame in grid.groupby("year"):
            g = float(frame.g.iloc[0])
            if frame.g.nunique() != 1:
                raise ValueError("Members disagree on forcing")
            lo, hi = GROUP_LIMITS
            rows.append(
                dict(
                    panel=label,
                    year=year,
                    members=len(frame),
                    warming_K=g,
                    regime="cold" if g < lo else "warm" if g > hi else "inside",
                    experiment="ssp245"
                    if cohort == "ssp245_year"
                    else "historical"
                    if year < 2015
                    else "ssp585",
                )
            )
        if cohort == "ssp245_year":
            checks = []
            for r in grid.itertuples():
                mark = read(case_dir(store, r.member, r.year) / "complete.json")
                if (
                    len(mark["hashes"]) != 12
                    or len(mark["sources"]) != 9
                    or sum(mark["days"]) != 365
                ):
                    raise ValueError("Incomplete annual SSP2-4.5 reference")
                checks.append(
                    dict(
                        member=r.member,
                        year=r.year,
                        calendar=mark["calendar"],
                        month_days=mark["days"],
                        prior_september_sha256=mark["september_reference_sha256"],
                        monthly_regional_sha256=mark["hashes"],
                        source_checksums={
                            k: v["source_checksum"] for k, v in mark["sources"].items()
                        },
                    )
                )
            save_json(root / "ssp245_annual_source_audit.json", checks)
    write(pd.DataFrame(rows), root / "climate_panels.csv")
    save_json(root / "climate_members.json", members)


def gather_method_data(tag, root, roles=("val", "test", "fresh", "ssp245_year")):
    regional = []
    spatial = []
    physical = []
    transfer = []
    blocks = []
    bands = []
    for role in roles:
        for mtag, label in (("published_v1", "published"), (tag, "research")):
            verification = WORK / "verification" / mtag / role
            for month in range(1, 13):
                mm = f"{month:02}"
                p = pd.read_parquet(verification / mm / "physics.parquet")
                p["month"] = month
                physical.append(p.assign(panel=role, model=label))
            rp = sorted((WORK / "marine_regions" / mtag / role).glob("*.parquet"))
            sp = sorted((WORK / "spatial_check" / mtag / role).glob("*.parquet"))
            tp = sorted((WORK / "attribution" / mtag / role).glob("*_years.csv"))
            if len(rp) != 12 or len(sp) != 12 or len(tp) != 12:
                raise ValueError(f"Missing {role} regional or dependence verification for {mtag}")
            rc = pd.concat(
                [pd.read_parquet(i).assign(month=int(i.stem)) for i in rp], ignore_index=True
            )
            rc["season"] = np.choose((rc.month % 12) // 3, ["DJF", "MAM", "JJA", "SON"])
            rc = (
                rc.groupby(["region", "field", "season"])
                .agg(
                    crps=("crps", "mean"),
                    coverage90=("coverage90", "mean"),
                    projection_mse=("projection_mse", "mean"),
                    upper_sse=("upper_tail_sse", "sum"),
                    upper_weight=("upper_tail_weight", "sum"),
                    upper_brier=("upper_brier", "mean"),
                    pred_event_rate=("predicted_rate", "mean"),
                    MPI_event_rate=("event_rate", "mean"),
                )
                .reset_index()
            )
            rc["projection_rmse"] = np.sqrt(rc.pop("projection_mse"))
            rc["upper_tail_rmse"] = np.sqrt(rc.pop("upper_sse") / rc.pop("upper_weight"))
            regional.append(rc.assign(panel=role, model=label))
            sd = pd.concat(
                [pd.read_parquet(i).assign(month=int(i.stem)) for i in sp], ignore_index=True
            )
            sd = (
                sd.groupby(["region", "metric", "field", "distance_steps"])
                .value.mean()
                .reset_index()
            )
            spatial.append(sd.assign(panel=role, model=label))
            tb = pd.concat([pd.read_csv(i) for i in tp], ignore_index=True)
            transfer.append(tb.assign(panel=role, model=label))
            parts = sorted((WORK / "attribution" / mtag / role).glob("*_blocks.csv"))
            if len(parts) != 12:
                raise ValueError(f"Missing ocean–atmosphere variance attribution for {mtag}/{role}")
            blocks.append(
                pd.concat([pd.read_csv(i) for i in parts]).assign(panel=role, model=label)
            )
        controls = pd.read_csv(WORK / "comparison" / tag / role / "weather_controls.csv")
        controls["moment_source"] = controls.model.str.split("_", n=1).str[0]
        k = ["month", "group", "window", "target", "moment_source"]
        iod = controls[controls.model.str.endswith("_IID")][k + ["ratio"]].rename(
            columns={"ratio": "iid_ratio"}
        )
        controls = controls.merge(iod, on=k, validate="many_to_one")
        controls["closer_than_iid"] = (controls.ratio - 1).abs() < (controls.iid_ratio - 1).abs()
        bands.append(controls.assign(panel=role))
    write(pd.concat(regional), root / "regional_ocean.csv")
    write(pd.concat(spatial), root / "ocean_dependence.csv")
    ps = (
        pd.concat(physical)
        .groupby(["panel", "model", "group", "month"])[
            [
                "mean_wind_ms",
                "shear_quantile_ms",
                "weak_spells_per_month",
                "lag_bias",
                "q95_shear",
                "ref_q95_shear",
            ]
        ]
        .mean()
        .reset_index()
    )
    write(ps, root / "wind_physics.csv")
    write(pd.concat(transfer), root / "background_to_wind.csv")
    write(pd.concat(blocks), root / "background_to_wind_by_field.csv")
    configs = [read(WORK / "tuned_fourier" / tag / f"{m:02}.json") for m in range(1, 13)]
    write(
        pd.DataFrame(configs).drop(columns="training_years"),
        root / "training_calibrated_fourier.csv",
    )
    weather = pd.concat(bands, ignore_index=True)
    write(weather, root / "wind_windows_and_fourier.csv")
    multi = weather[(weather.target == "total") & weather.window.isin(["5", "10", "15"])].copy()
    multi["in_release_band"] = multi.in_release_band.astype(bool)
    multi["passed"] = multi.in_release_band & multi.closer_than_iid
    grouping = ["panel", "model", "group"]
    flags = ["passed", "in_release_band", "closer_than_iid"]
    collective = multi.groupby(grouping + ["month"], as_index=False)[flags].all()
    table = pd.concat([multi, collective.assign(window="5+10+15")], ignore_index=True)
    counts = table.groupby(grouping + ["window"], as_index=False).agg(
        months_tested=("month", "nunique"),
        months_in_band=("in_release_band", "sum"),
        months_closer_to_mpi_than_iid=("closer_than_iid", "sum"),
        months_passing=("passed", "sum"),
    )
    write(counts, root / "multiday_pass_counts.csv")


def learning_tables(root):
    allrows = []
    steps = []
    for p in (WORK / "learning_curve").glob("lane*/*.json"):
        data = read(p)
        same = data["results"]
        if "original" not in same or len(same) < 2:
            continue
        for old, new in (
            ("original", "expanded"),
            ("original", "extended"),
            ("expanded", "extended"),
        ):
            if old not in same or new not in same:
                continue
            before, after = (same[k]["results"][0] for k in (old, new))
            if before["arm"] != "reference" or after["arm"] != "reference":
                raise ValueError("Learning-step comparison expects a common reference architecture")
            crps = [after[k] / before[k] - 1 for k in before if k.endswith("_crps")]
            delta = after["conditional_wind_rmse"] - before["conditional_wind_rmse"]
            steps.append(
                dict(
                    from_cohort=old,
                    to_cohort=new,
                    regime=data["scheme"],
                    wind_percent=100 * delta / before["conditional_wind_rmse"],
                    wind_difference_ms=delta,
                    ocean_percent=100 * np.mean(crps),
                    wind_improved=float(delta < 0),
                )
            )
        for cohort, x in same.items():
            if cohort == "original":
                continue
            for i, row in enumerate(x["results"]):
                ref = same["original"]["results"][i]
                keys = [k for k in row if k.endswith("_crps")]
                allrows.append(
                    dict(
                        cohort=cohort,
                        regime=data["scheme"],
                        members=data["members"],
                        arm=row["arm"],
                        training_cases=x["train_cases"],
                        base_cases=same["original"]["train_cases"],
                        ocean_percent=100 * np.mean([row[k] / ref[k] - 1 for k in keys]),
                        conditional_wind_percent=100
                        * (row["conditional_wind_rmse"] / ref["conditional_wind_rmse"] - 1),
                        total_log_error_difference=row["wind_total"] - ref["wind_total"],
                        within_log_error_difference=row["wind_within"] - ref["wind_within"],
                    )
                )
    d = pd.DataFrame(allrows)
    if not d.empty:
        grp = d.groupby(["cohort", "regime", "members", "arm"])
        means = grp.mean(numeric_only=True)
        means["matched_splits"] = grp.size()
        write(means.reset_index(), root / "matched_learning_curve.csv")
    if steps:
        step = pd.DataFrame(steps)
        step = pd.concat([step, step.assign(regime="all")], ignore_index=True)
        groups = step.groupby(["from_cohort", "to_cohort", "regime"])
        summary = groups.mean(numeric_only=True)
        summary["matched_splits"] = groups.size()
        write(summary.reset_index(), root / "matched_training_steps.csv")


def make(out):
    chosen = read(WORK / "FROZEN.json")
    tag = chosen["stage"]
    root = out / "data"
    root.mkdir(exist_ok=True, parents=True)
    for role in ("val", "test", "fresh", "ssp245_year"):
        write(forecast_intervals(tag, role), root / ("two_day_forecasts_" + role + ".csv"))
        copy_table(
            WORK / "comparison" / tag / role / "paired.csv", root / ("paired_" + role + ".csv")
        )
        copy_table(
            WORK / "comparison" / tag / role / "fourier_intervals.csv",
            root / ("paired_fourier_" + role + ".csv"),
        )
        for table in ("leadlag_intervals", "leadlag_matrices"):
            copy_table(
                WORK / "comparison" / tag / role / (table + ".csv"),
                root / (table + "_" + role + ".csv"),
            )
    cp = WORK / "comparison" / tag
    copy_table(
        cp / "fresh/paired_excluding_september.csv", root / "paired_fresh_excluding_september.csv"
    )
    copy_table(cp / "val/storms.csv", root / "paired_seasonal_storms.csv")
    copy_table(cp / "ssp245/paired.csv", root / "paired_mpi2050.csv")
    copy_table(cp / "ssp245_year/storms.csv", root / "annual_mpi2050_storms.csv")
    copy_table(cp / "ssp245/ssp245_tracks.png", out / "figures/ssp245_tracks.png")
    archives = {panel: archive_storms(tag, root, panel) for panel in ("val", "ssp245_year")}
    save_json(root / "synthetic_storm_archives.json", archives)
    for name in ("attempts.parquet", "tc_tracks_6h.parquet"):
        copy_table(WORK / "mpi2050" / tag / name, root / ("september_mpi2050_" + name))
    gather_method_data(tag, root)
    write(
        pd.read_csv(WORK / "selection.csv").drop(columns=["weights"]), root / "model_selection.csv"
    )
    frame = acquire()
    write(frame, root / "coupled_validation.csv")
    copy_table(WORK / "experiment_records/infill_audit.json", root / "ocean_input_agreement.json")
    copy_table(
        WORK / "reproduction" / tag / "reproduction.json", root / "independent_retraining.json"
    )
    for alternative in ("original_nested_refined", "extended_balanced_refined"):
        path = WORK / "reproduction" / alternative / "reproduction.json"
        if path.is_file():
            copy_table(path, root / ("validation_retraining_" + alternative + ".json"))
    copy_table(WORK / "packages" / tag / "verified.json", root / "checkpoint_checks.json")
    copy_table(WORK / "crossfit_summary/overall.csv", root / "training_cv.csv")
    copy_table(WORK / "crossfit_summary/bands.csv", root / "training_cv_by_temperature.csv")
    source_manifest(root)
    learning_tables(root)
    ablations(root)
    copy_table(WORK / "experiment_records/progress.csv", root / "history.csv")
    copy_table(WORK / "experiment_records/extended_progress.csv", root / "expanded_history.csv")
    journal = (
        (WORK / "JOURNAL.md")
        .read_text()
        .replace("(experiment_records/events.jsonl)", "(data/events.jsonl.gz)")
    )
    with atomic(out / "JOURNAL.zh.md") as tmp:
        tmp.write_text(journal)
    copy_table(WORK / "DECISIONS.md", out / "DECISIONS.md")
    copy_table(WORK / "recipe.json", out / "recipe.json")
    public = dict(chosen)
    public.pop("weights")
    public["validation"] = "data/paired_val.csv"
    save_json(out / "freeze_receipt.json", public)
    tally = Counter()
    times = []
    design_window = read(Path(__file__).with_name("design.json"))
    start = datetime.fromisoformat(design_window["start_utc"])
    end = datetime.fromisoformat(design_window["end_utc"])
    step = timedelta(minutes=10)
    slots = []
    for i in range(int(np.ceil((end - start) / step))):
        left = start + i * step
        slots.append(
            dict(
                start_utc=left.isoformat(),
                end_utc=min(left + step, end).isoformat(),
                recorded_events=0,
                completed_training_fits=0,
                completed_storm_replays=0,
            )
        )
    dest = root / "events.jsonl.gz"
    with open(WORK / "experiment_records/events.jsonl") as source, open(dest, "wb") as target:
        with gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as zipped:
            for line in source:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tally[row["event"]] += 1
                times.append(row["utc"])
                when = datetime.fromisoformat(row["utc"])
                if start <= when < end:
                    slot = slots[(when - start) // step]
                    slot["recorded_events"] += 1
                    slot["completed_training_fits"] += row["event"] in {
                        "paired_refit_done",
                        "learning_curve_refit",
                        "coupled_done",
                    }
                    slot["completed_storm_replays"] += row["event"] == "storm_case_done"
                zipped.write(
                    (
                        json.dumps(row, ensure_ascii=False, sort_keys=True).replace(str(ROOT), ".")
                        + "\n"
                    ).encode()
                )
    save_json(
        root / "experiment_counts.json",
        dict(
            requested_start=start.isoformat(),
            requested_end=end.isoformat(),
            first_event=min(times),
            latest_event=max(times),
            counts=dict(tally),
        ),
    )
    write(pd.DataFrame(slots), root / "activity_by_10min.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    make(p.parse_args().output)
