"""Prospectively confirm compact wind memory and the physical MLD ceiling on six new climates."""
import argparse
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from deep_mixed_layer import exceedance, quantile
from bathymetric_mld import marine_masks
from ocean import GROUP_LIMITS
from order_reduction import steering_error
from paired_intervals import panel_weights, record_estimate
from partial_wind_memory import past_directions, project_state
from records import WORK, record
from spatial_check import field_loc
from verify_model import forecast_coefficients
from weather import model_terms, projected, year_table
from wind_moments import observed_arrays

from annual_lim.layout import CONFIG, case_dir, design, read
from annual_lim.model import MonthlyModel
from annual_lim.statistics import metrics
from annual_lim.train import months
from tc_emulator.common import ROOT, atomic, save_json, sha


SCIENCE = Path(__file__).resolve().parent
PLAN = SCIENCE / "compact_ocean_confirmation_plan.json"
PUBLIC = ROOT / "reports/annual_refinement/data"
CFG = WORK / "configs/compact_ocean_confirmation"
HERE = WORK / "compact_ocean_confirmation"
SOURCE = HERE / "input"


def setup():
    plan = read(PLAN)
    CFG.mkdir(parents=True, exist_ok=True)
    for name in ("members.json", "forcing.csv", "catalog.parquet"):
        dest = CFG / name
        if not dest.exists():
            dest.symlink_to(CONFIG / name)
    save_json(
        CFG / "settings.json",
        dict(
            read(CONFIG / "settings.json"),
            train_years=[],
            validation_years=[],
            test_years=sorted(map(int, plan["years_K"])),
        ),
    )


def checked():
    plan = read(PLAN)
    if (
        sha(SCIENCE / "partial_wind_memory.py") != plan["partial_source_sha256"]
        or sha(PUBLIC / "mpi_bathymetry_grid.npz") != plan["bathymetry_sha256"]
    ):
        raise ValueError("A registered numerical input has changed")
    target = {
        (m, int(y), g) for y, g in plan["years_K"].items() for m in plan["members"]
    }
    if {
        (m, y, g) for role, m, y, g in design(CFG)[1] if role == "test"
    } != target or set(map(int, plan["years_K"])) & set(plan["excluded_annual_years"]):
        raise ValueError("Confirmation climates differ from the registered plan")
    return plan


def ocean_scores(model, fields, meta, depth, month):
    sea = model.rep.blocks["ocean"].valid[0].ravel()
    wet = sea & np.isfinite(depth) & (depth > 0)
    bottom = depth[wet]
    yy, xx = np.meshgrid(model.rep.lat, model.rep.lon, indexing="ij")
    y, x = yy.ravel()[wet], xx.ravel()[wet]
    regions = marine_masks(y, x, bottom)
    truth = np.stack([d["mld"].ravel()[wet] for d in fields])
    scores = []
    for _, panel in meta.groupby("year"):
        g = panel.g.iloc[0]
        loc, basis, mask = field_loc(
            model, model.bg.mean(np.array([g]))[0], model.factor(g), "mld"
        )
        np.testing.assert_array_equal(mask, sea)
        loc, sd = loc[wet], np.linalg.norm(basis[wet], axis=1)
        obs = truth[panel.index]
        for method, cap in (("research_log", np.inf), ("sea_floor", bottom)):
            q = quantile(loc, sd, 0.99, 1.0, cap)
            p = exceedance(loc, sd, bottom, 1.0, cap)
            for region, zone in regions.items():
                weights = np.isfinite(obs) * zone * np.cos(np.deg2rad(y))
                weights /= weights.sum(axis=1, keepdims=True)

                def avg(z):
                    return np.sum(weights * np.where(np.isfinite(z), z, 0.0), axis=1)

                diff = obs - q
                table = panel[["member", "year", "g"]].assign(
                    month=month,
                    method=method,
                    region=region,
                    q99_loss_m=avg(np.maximum(0.99 * diff, -0.01 * diff)),
                    q99_exceedance=avg(obs > q),
                    model_below_floor=avg(p),
                    actual_below_floor=avg(obs > bottom),
                    actual_over_by_1m=avg(obs > bottom + 1.0),
                )
                for cut in (2000, 3000):
                    event = obs > cut
                    prob = exceedance(loc, sd, cut, 1.0, cap)
                    table[f"expected_over_{cut}m"] = avg(prob)
                    table[f"observed_over_{cut}m"] = avg(event)
                    table[f"brier_{cut}m"] = avg((prob - event) ** 2)
                scores.append(table)
    return pd.concat(scores, ignore_index=True)


def run(month, weights):
    plan = checked()
    if month not in plan["months"]:
        raise ValueError("Month outside the registered panel")
    base = MonthlyModel(weights, month)
    if base.meta["id"] != plan["frozen_primary"][f"{month:02}"]:
        raise ValueError("Unexpected frozen primary")
    fields, rows, grid = months(CFG, SOURCE, month, "test")
    meta = pd.DataFrame(rows)
    np.testing.assert_array_equal(grid[0], base.rep.lat)
    np.testing.assert_array_equal(grid[1], base.rep.lon)
    area, wet, _ = metrics(base.rep)
    winds = [d["wind"] for d in fields]
    observations = observed_arrays(winds, rows, area, wet)
    years, g, lens = year_table(observations)
    block = base.rep.blocks["wind"]
    pcs = [block.transform(w) for w in winds]
    means = [base.wind_mean(gg, base.rep.encode(d)) for gg, d in zip(meta.g, fields)]
    short = copy(base)
    back, axes, _ = past_directions(base.base)
    short.base = project_state(base.base, back, axes, plan["compact_past_rank"])
    scores, windows = [], []
    for name, model in (("primary_240", base), ("compact_192", short)):
        coef = forecast_coefficients(model, [1])[1]
        mse = [
            steering_error(w, pc - mu, mu, block, area, 1, coef)
            for w, pc, mu in zip(winds, pcs, means)
        ]
        scores.append(
            meta[["member", "year", "g", "sha256"]].assign(
                month=month, method=name, mse=mse
            )
        )
        lags, bg, _ = model_terms(model, g)
        total, center, _ = projected(model.cal, lags, g, lens, bg)
        for target, pred, truth in (
            ("total", total, observations["by_year_cov"]),
            ("within", center, observations["by_year_within"]),
        ):
            windows.append(
                pd.DataFrame(
                    dict(
                        month=month,
                        year=years,
                        g=g,
                        method=name,
                        target=target,
                        predicted=pred[:, 3],
                        observed=truth[:, 3],
                        abs_log=np.abs(np.log(pred[:, 3] / truth[:, 3])),
                    )
                )
            )
    with np.load(PUBLIC / "mpi_bathymetry_grid.npz") as depth:
        np.testing.assert_array_equal(depth["lat"], base.rep.lat)
        np.testing.assert_array_equal(depth["lon"], base.rep.lon)
        ocean = ocean_scores(base, fields, meta, depth["depth_m"].ravel(), month)
    dest = HERE / "scores" / f"{month:02}"
    for key, frame in (
        ("day_one", pd.concat(scores)),
        ("windows", pd.concat(windows)),
        ("ocean", ocean),
    ):
        with atomic(dest / (key + ".parquet")) as temp:
            frame.to_parquet(temp, index=False, compression="zstd")
    record(
        "compact_ocean_confirmation_month",
        month=month,
        member_months=len(meta),
        plan_sha256=sha(PLAN),
    )


def summarize(output):
    plan = checked()
    day, windows, ocean = [
        pd.concat(
            [
                pd.read_parquet(HERE / "scores" / f"{m:02}" / (key + ".parquet"))
                for m in plan["months"]
            ],
            ignore_index=True,
        )
        for key in ("day_one", "windows", "ocean")
    ]
    if len(day) != 1056 or len(windows) != 264 or len(ocean) != 3168:
        raise ValueError("The confirmation is incomplete")
    rows = []
    for metric, dim, frame, ref, cand in [
        (
            "day_one_rmse_ms",
            "wind",
            day.rename(columns={"mse": "score"}),
            "primary_240",
            "compact_192",
        ),
        *[
            (
                "q99_loss_m",
                region,
                part.rename(columns={"q99_loss_m": "score"}),
                "research_log",
                "sea_floor",
            )
            for region, part in ocean.groupby("region")
        ],
    ]:
        panel = (
            frame.groupby(["member", "year", "g", "method"])
            .score.mean()
            .unstack("method")
        )
        member, yr, g = [panel.index.get_level_values(k).to_numpy() for k in range(3)]
        for ii, regime in enumerate(("all", "cold", "inside", "warm")):
            w, nm, ny = panel_weights(
                member, yr, g, regime, plan["replicates"], 601774 + ii
            )
            a, b = w @ panel[ref].to_numpy(), w @ panel[cand].to_numpy()
            if metric == "day_one_rmse_ms":
                a, b = np.sqrt(a), np.sqrt(b)
            record_estimate(rows, regime, metric, [dim], a, b, nm, ny, "raw")
    paired = pd.DataFrame(rows)
    windows["group"] = np.select(
        [windows.g < GROUP_LIMITS[0], windows.g > GROUP_LIMITS[1]],
        ["cold", "warm"],
        default="inside",
    )
    summaries = (
        pd.concat([windows, windows.assign(group="all")])
        .groupby(["group", "method", "target"], as_index=False)
        .abs_log.mean()
    )
    obs_cols = [
        c
        for c in ocean
        if c.startswith(
            ("q99", "model_", "actual_", "expected_", "observed_", "brier_")
        )
    ]
    sea = ocean.groupby(["region", "method"], as_index=False)[obs_cols].mean()
    src = day[day.method == "primary_240"][["member", "year", "g", "month", "sha256"]]
    assert len(src.drop_duplicates(["member", "year", "month"])) == 528
    inputs = []
    for m, y in (
        src[["member", "year"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        mark = read(case_dir(SOURCE, m, int(y)) / "complete.json")
        assert len(mark["sources"]) == 9
        for key, val in mark["sources"].items():
            inputs.append(
                dict(
                    member=m,
                    year=y,
                    field=key,
                    version=val["version"],
                    checksum=val["source_checksum"],
                    location=val["url"],
                )
            )
    limit = plan["criteria"]
    wind = paired[(paired.group == "all") & (paired.metric == "day_one_rmse_ms")].iloc[
        0
    ]
    vs = summaries[summaries.group == "all"].pivot(
        index="target", columns="method", values="abs_log"
    )
    diff = vs.compact_192 - vs.primary_240
    meets = bool(
        wind.ci_high < limit["wind_day_one_paired_upper_margin_ms"]
        and diff["total"] <= limit["fixed_g_total_15day_max_increase"]
        and diff["within"] <= limit["fixed_g_within_15day_max_increase"]
    )
    decision = dict(
        plan_sha256=sha(PLAN),
        years=plan["years_K"],
        independent_member_months=528,
        day_one_increase_ms=wind.difference,
        day_one_upper95_ms=wind.ci_high,
        total_15day_error_increase=diff["total"],
        within_15day_error_increase=diff["within"],
        compact_prespecified_criteria_met=meets,
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, frm in (
        ("paired", paired),
        ("window_summary", summaries),
        ("ocean_summary", sea),
        ("source_hashes", src),
        ("source_datasets", pd.DataFrame(inputs)),
    ):
        frm.to_csv(
            output / ("compact_ocean_confirmation_" + name + ".csv"), index=False
        )
    for name, frm in (
        ("forecast_cases", day),
        ("window_cases", windows),
        ("ocean_cases", ocean),
    ):
        frm.to_parquet(
            output / ("compact_ocean_confirmation_" + name + ".parquet"),
            index=False,
            compression="zstd",
        )
    save_json(output / "compact_ocean_confirmation_result.json", decision)
    record("compact_ocean_confirmation_result", **decision)
    print(decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--setup", action="store_true")
    mode.add_argument("--month", type=int)
    mode.add_argument("--summary", type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    if args.setup:
        setup()
    elif args.month:
        path = args.weights or Path(read(WORK / "FROZEN.json")["weights"])
        run(args.month, path)
    else:
        summarize(args.summary)
