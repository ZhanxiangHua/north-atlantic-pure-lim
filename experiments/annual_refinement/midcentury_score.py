"""Archive matched replays and score the registered six-year test and its 2050 sensitivity."""
import argparse
import sys
from itertools import product

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from annual_lim.layout import CONFIG, case_dir, read
from midcentury_prepare import STUDY
from midcentury_sources import PLAN, REPORT
from paired_intervals import panel_weights
from score_storms import RATES, SHAPES, distance, limits, rates
from storm_archive import KEYS
from tc_emulator.common import ROOT, atomic, save_json, sha
from tc_emulator.physical import upstream

PROPOSAL = ["year", "month", "member", "draw", "origin", "seed_id"]
FIXED = "MPI background"
PRIOR = ROOT / "reports/annual_refinement/data"


def drivers(plan):
    return {plan["reference_driver"]: ["MPI daily"], plan["baseline"]: [FIXED, "warming only"],
            plan["candidate"]: [FIXED, "warming only"], **{f: [FIXED] for f in plan["fourier_controls"]}}


def archive(plan, members):
    months, years = plan["months"], plan["new_years"]
    prepared = { (m, y): read(case_dir(STUDY / "ssp245_year", m, y) / "complete.json")
                 for m, y in product(members, years) }
    inputs = []
    for (m, y), mark in prepared.items():
        for n, days in zip(months, mark["days"], strict=True):
            inputs.append(dict(member=m, year=y, month=n, days=days, g=mark["g"],
                               sha256=mark["hashes"][f"{n:02}.npz"]))
    inputs = pd.DataFrame(inputs).sort_values(["year", "member", "month"])
    with atomic(REPORT / "prepared_months.csv") as tmp:
        inputs.to_csv(tmp, index=False)
    folders = []
    for tag, protocols in drivers(plan).items():
        for m, y, n in product(members, years, months):
            folder = STUDY / "storm_ssp245_year" / tag / f"{n:02}" / str(y) / m
            done = read(folder / "complete.json")
            if (done["member"], done["year"], done["month"], done["tag"], done["origins"], done["seeds"],
                    done["g"], done["input_sha256"]) != (
                    m, y, n, tag, plan["start_days"], plan["proposals_per_start"], prepared[m,y]["g"],
                    prepared[m,y]["hashes"][f"{n:02}.npz"]):
                raise ValueError(f"Simulation receipt does not match design: {folder}")
            if done["attempts"] != len(protocols) * len(plan["start_days"]) * plan["proposals_per_start"]:
                raise ValueError(f"Missing a storm panel: {folder}")
            folders.append(folder)
    pa.set_cpu_count(4)
    for src, dest, keys in [("attempts", "attempts", KEYS), ("fixes", "tracks_6h", [*KEYS, "elapsed_h"])]:
        paths = [str(p / f"{src}.parquet") for p in folders]
        frame = ds.dataset(paths, format="parquet").to_table().to_pandas()
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate key in {src}")
        with atomic(REPORT / f"{dest}.parquet") as tmp:
            frame.sort_values(keys).to_parquet(tmp, index=False, compression="zstd")


def paired_inputs(attempts, plan, members):
    reference = attempts.loc[attempts.model == plan["reference_driver"]]
    cols = [*PROPOSAL, "g", "lon", "lat", "filter_draw", "v_init"]
    ref = reference[cols].sort_values(PROPOSAL).reset_index(drop=True)
    pp = plan["proposals_per_start"]
    if ref.duplicated(PROPOSAL).any():
        raise ValueError("Duplicate MPI disturbance")
    groups = ref.groupby(["year", "member", "month", "draw", "origin"]).size()
    years = sorted([*plan["new_years"], plan["reference_year"]])
    expected = {(y, m, n, d, o) for y, m, n in product(years, members, plan["months"])
                for d, o in enumerate(plan["start_days"])}
    if set(groups.index) != expected or not groups.eq(pp).all():
        raise ValueError("MPI lacks the complete member-by-year design")
    protocol_set = {(tag, protocol) for tag, protocols in drivers(plan).items() for protocol in protocols}
    if set(attempts.groupby(["model", "protocol"]).groups) != protocol_set:
        raise ValueError("Unexpected model or protocol")
    for (tag, protocol), frame in attempts.groupby(["model", "protocol"]):
        other = frame.sort_values(PROPOSAL)[cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(other, ref, check_exact=True, check_dtype=False,
                                      obj=f"disturbances for {tag}/{protocol}")


def check_tracks(attempts, fixes):
    sys.path.insert(0, str(ROOT / "notebooks"))
    from tc_benchmark import aligned_window
    # aligned_window uses the same radius from the TC solver as the original integration.
    upstream()
    rows = []
    for k, tr in fixes.sort_values([*KEYS, "elapsed_h"]).groupby(KEYS, sort=False):
        hours = tr.elapsed_h.to_numpy()
        if hours[0] != 0 or (np.diff(hours) != 6).any():
            raise ValueError(f"Discontinuous six-hourly track: {k}")
        segment = aligned_window(tr.lon_e.to_numpy(), tr.lat_n.to_numpy(), tr.vmax_ms.to_numpy(),
                                 hours, np.ones(len(hours), dtype=bool))
        rows.append((*k, *(segment[1:] if segment else (np.nan, np.nan))))
    rebuilt = pd.DataFrame(rows, columns=[*KEYS, "peak_48h_ms", "displacement_48h_km"]).set_index(KEYS).sort_index()
    saved = attempts[attempts.tc].set_index(KEYS).sort_index()
    pd.testing.assert_index_equal(rebuilt.index, saved.index)
    for metric, tolerance in [("displacement_48h_km", .005), ("peak_48h_ms", .001)]:
        np.testing.assert_allclose(rebuilt[metric], saved[metric], atol=tolerance, rtol=0, equal_nan=True)
    return dict(accepted_storms=len(rebuilt), six_hourly_fixes=len(fixes),
                eligible_48h=int(rebuilt.displacement_48h_km.notna().sum()),
                max_rebuilt_distance_difference_km=(rebuilt.displacement_48h_km-saved.displacement_48h_km).abs().max())


def score_scope(scope, rows, plan):
    ref = rows[rows.model == plan["reference_driver"]]
    grid = ref.groupby(["member", "year"], sort=True).g.first()
    mem, yrs = (grid.index.get_level_values(i).to_numpy() for i in (0, 1))
    weights, _, _ = panel_weights(mem, yrs, grid.to_numpy(), "all", plan["bootstrap_draws"], plan["bootstrap_seed"])
    scores, contrasts = [], []
    for protocol in (FIXED, "warming only"):
        group = rows[rows.protocol == protocol]
        for metric in (*SHAPES, *RATES):
            error = {}
            for tag, model in group.groupby("model", sort=False):
                if metric in RATES:
                    obs, pred = rates(ref, metric, grid.index, weights), rates(model, metric, grid.index, weights)
                    val, unit = np.abs(pred - obs), "per 100 proposed disturbances"
                else:
                    obs = pred = None
                    val, unit = distance(ref, model, metric, grid.index, weights), SHAPES[metric]
                if not np.isfinite(val[0]):
                    raise ValueError(f"Undefined primary or secondary point in {scope}/{tag}/{metric}")
                error[tag] = val
                scores.append(dict(scope=scope, protocol=protocol, model=tag, metric=metric, error=val[0],
                                   reference_rate=obs[0] if obs is not None else np.nan,
                                   model_rate=pred[0] if pred is not None else np.nan, **limits(val), unit=unit))
            pairs = [(plan["candidate"], tag) for tag in [plan["baseline"], *plan["fourier_controls"]]]
            pairs.append((plan["baseline"], "native_fourier"))
            for model, tag in pairs:
                if tag not in error:
                    continue
                diff = error[model] - error[tag]
                contrasts.append(dict(scope=scope, protocol=protocol, model=model, comparator=tag,
                                      metric=metric, change=diff[0], **limits(diff), unit=unit))
    return scores, contrasts


def summarize(do_archive):
    plan, split = read(PLAN), read(CONFIG / "members.json")
    reg = read(REPORT / "registration.json")
    if (sha(PLAN), sorted(split["test"]), plan["new_years"]) != (
        reg["plan_sha256"], sorted(reg["held_out_members"]), reg["primary_years"]
    ):
        raise ValueError("Study design or held-out members differ from registration")
    if do_archive:
        archive(plan, split["test"])
    new = pd.read_parquet(REPORT / "attempts.parquet")
    if set(new.year.unique()) != set(plan["new_years"]):
        raise ValueError("Archive does not cover the registered six years")
    checks = check_tracks(new, pd.read_parquet(REPORT / "tracks_6h.parquet"))
    old = pd.read_parquet(PRIOR / "annual_mpi2050_attempts.parquet")
    full = pd.concat([new, old], ignore_index=True)
    paired_inputs(full, plan, split["test"])
    counts = full.assign(eligible=full.tc & full.displacement_48h_km.notna()).groupby(
        ["year", "g", "model", "protocol"]).agg(attempts=("tc", "size"), tc=("tc", "sum"),
        eligible_48h=("eligible", "sum"), landfall=("landfall", "sum")).reset_index()
    scores, contrasts = [], []
    for scope, rows in [(str(y), d) for y, d in full.groupby("year")] + [("new_six", new), ("all_seven", full)]:
        s, c = score_scope(scope, rows, plan)
        scores += s
        contrasts += c
        print(scope, flush=True)
    scores, contrasts = pd.DataFrame(scores), pd.DataFrame(contrasts)
    old_scores = pd.read_csv(PRIOR / "annual_mpi2050_storms.csv")
    keys = ["protocol", "driver", "metric"]
    original = old_scores[old_scores.month.eq("all")].set_index(keys)
    back = scores[scores.scope.eq(str(plan["reference_year"]))].rename(columns={"model": "driver"}).set_index(keys)
    np.testing.assert_allclose(back.error, original.loc[back.index, "error"], rtol=1e-10, atol=1e-10)
    for name, frame in [("counts", counts), ("scores", scores), ("contrasts", contrasts)]:
        with atomic(REPORT / (name + ".csv")) as tmp:
            frame.to_csv(tmp, index=False)
    checks.update(new_attempt_rows=len(new), previous_attempt_rows=len(old),
                  shared_proposals=int(new.model.eq(plan["reference_driver"]).sum()),
                  registered_plan_sha256=sha(PLAN),
                  source_files_sha256=sha(REPORT / "cmip6_files.csv"),
                  new_archives_sha256={f: sha(REPORT / f) for f in ("attempts.parquet", "tracks_6h.parquet")})
    save_json(REPORT / "checks.json", checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true", help="Rebuild raw public archives from the completed FAST jobs first.")
    summarize(parser.parse_args().archive)
