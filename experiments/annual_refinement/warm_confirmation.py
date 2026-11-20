"""Run the registered constant-gain confirmation on five additional MPI warm years."""
import argparse
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joint import PhysicalValidation
from paired_intervals import panel_weights, record_estimate
from records import WORK, record
from tuned_fourier import harmonic_lags
from weather import kernels, model_terms, projected
from weather_bootstrap import WINDOWS, accumulate
from weather_controls import daily_fourier_lags
from wind_visible_gain import linked_factor

from annual_lim.layout import CONFIG, case_dir, design, read
from annual_lim.model import MonthlyModel
from annual_lim.statistics import metrics, variance, window_moments
from annual_lim.train import months
from tc_emulator.common import ROOT, atomic, save_json, sha
from tc_emulator.warming_circulation import transfer_matrix

PLAN = Path(__file__).with_name("warm_confirmation_plan.json")
HERE = WORK / "warm_confirmation"
CFG = WORK / "configs/warm_confirmation"
INPUT = HERE / "input"
PUBLIC = ROOT / "reports/annual_refinement"
NAMES = (
    "primary_LIM",
    "gain_LIM",
    "primary_IID",
    "gain_IID",
    "primary_native_Fourier",
    "gain_native_Fourier",
    "primary_calibrated_Fourier",
    "gain_calibrated_Fourier",
)


def setup():
    plan = read(PLAN)
    CFG.mkdir(parents=True, exist_ok=True)
    for name in ("members.json", "forcing.csv", "catalog.parquet"):
        dest = CFG / name
        if not dest.exists():
            dest.symlink_to(CONFIG / name)
    source = dict(
        read(CONFIG / "settings.json"),
        train_years=[],
        validation_years=[],
        test_years=sorted(map(int, plan["years_K"])),
    )
    save_json(CFG / "settings.json", source)
    checked_cases(plan)


def checked_cases(plan):
    rows = design(CFG)[1]
    wanted = {(m, int(y), g) for y, g in plan["years_K"].items() for m in plan["members"]}
    if {(m, y, g) for role, m, y, g in rows if role == "test"} != wanted or len(rows) != len(
        wanted
    ):
        raise ValueError("Climate cases differ from the registered warm panel")
    return rows


def run(month, weights):
    plan = read(PLAN)
    if month not in plan["months"]:
        raise ValueError("Month outside the sealed confirmation")
    checked_cases(plan)
    if sha(PUBLIC / "data/wind_visible_gain_fits.csv") != plan["source_fit_sha256"]:
        raise ValueError("Gain source differs from the original sealed fit")
    if weights is None:
        marker = WORK / "FROZEN.json"
        weights = read(marker)["weights"] if marker.is_file() else ROOT / "work/refined_weights"
    model = MonthlyModel(weights, month)
    key = f"{month:02}"
    if model.meta["id"] != plan["primary_ids"][key]:
        raise ValueError("Wrong monthly checkpoint")
    out = HERE / "scores" / key
    if (out / "complete.json").is_file():
        return
    data, records, grid = months(CFG, INPUT, month, "test")
    for coord, value in zip(("lat", "lon", "plev"), grid):
        np.testing.assert_array_equal(value, getattr(model.rep, coord))
    meta = pd.DataFrame(records)
    g = meta.g.to_numpy()
    years, first, ii = np.unique(meta.year.to_numpy(), return_index=True, return_inverse=True)
    gs = g[first]
    days = meta.days.to_numpy()[first]
    area, _, G = metrics(model.rep)
    mu, sec = window_moments([d["wind"] for d in data], area)
    monthly = mu[:, -1]
    within = (
        sec
        - 2 * np.einsum("iwf,if->iw", mu, monthly)
        + np.einsum("if,if->i", monthly, monthly)[:, None]
    )
    obs = dict(
        member=meta.member.to_numpy(),
        year=meta.year.to_numpy(),
        g=g,
        mu=mu,
        second=sec,
        within=within,
    )
    lags, bg, forced = model_terms(model, gs)
    K = transfer_matrix(model.mean)
    gain = plan["amplitudes"][key]
    views = {}
    power = pd.read_csv(PUBLIC / "data/training_calibrated_fourier.csv")
    spectra = power[(power.month == month) & (power.model_id == model.meta["id"])]
    if len(spectra) != 1:
        raise ValueError("Missing training-calibrated Fourier parameters")
    sp = spectra.iloc[0]
    lags_native = daily_fourier_lags()
    lags_tuned = harmonic_lags(sp.period_days, sp.power_exponent)
    ops = [np.stack([kernels(int(n))[j] for n in days]) for j in (0, 1)]
    power = lags[0, 0] * model.cal.amplitude(gs) ** 2
    for label, s in (("primary", 1.0), ("gain", gain)):
        hbg = bg * s
        total, cent, _ = projected(model.cal, lags, gs, days, hbg)
        views[label + "_LIM"] = (total, cent, forced)
        for name, lag in (
            ("IID", np.r_[1.0, np.zeros(30)]),
            ("native_Fourier", lags_native),
            ("calibrated_Fourier", lags_tuned),
        ):
            views[label + "_" + name] = (
                hbg[:, :1] + power[:, None] * (ops[0] @ lag),
                power[:, None] * (ops[1] @ lag),
                forced,
            )
    hist = {}
    accumulate(obs, views, ii, hist, plan["resamples"])
    # Each replicate keeps the same member/year weights for all eleven calendar months.
    bootstrap = {}
    rows = []
    bands = read(Path(__file__).with_name("v1_fit_reference.json"))[key]["bands"]
    raw = variance(mu, sec)
    centered = within.mean(axis=0)
    for driver, (total, cent, forced) in views.items():
        for j, window in enumerate(WINDOWS):
            for target, pred, truth in (
                ("total", total[ii, j].mean() + np.var(g) * forced, raw[j]),
                ("within", cent[ii, j].mean(), centered[j]),
            ):
                if target == "within" and window == "month":
                    continue
                arr = hist[("all", target, window, driver, len(plan["members"]), len(years))][0]
                ratio = pred / truth
                np.testing.assert_allclose(arr[0], abs(np.log(ratio)), atol=2e-8)
                bootstrap["/".join((driver, target, window))] = arr
                lo, hi = bands[j] if target == "total" else (np.nan, np.nan)
                rows.append(
                    dict(
                        driver=driver,
                        month=month,
                        window=window,
                        target=target,
                        model_variance=pred,
                        mpi_variance=truth,
                        ratio=ratio,
                        abs_log_error=arr[0],
                        band_low=lo,
                        band_high=hi,
                    )
                )
    with atomic(out / "bootstrap.npz") as tmp:
        np.savez_compressed(tmp, **bootstrap)
    with atomic(out / "windows.csv") as tmp:
        pd.DataFrame(rows).to_csv(tmp, index=False)
    truth = np.stack([np.stack([d[k] for k in ("sst", "mld", "gamma")]) for d in data])
    valid = PhysicalValidation(SimpleNamespace(geo=model.rep, month=month), truth, meta, monthly)
    alt = copy(model)
    factors = {gi: linked_factor(model.factor(gi), K, gain) for gi in gs}
    alt.factor = factors.__getitem__
    zs = np.stack([model.rep.encode(d) for d in data])
    z_wind = zs @ K
    actual = []
    for year in years:
        sub = z_wind[meta.year == year]
        d = sub - sub.mean(axis=0)
        actual.append(np.sum((d @ G[0]) * d) / (len(sub) - 1))
    budget = pd.DataFrame(
        dict(
            month=month,
            year=years,
            g=gs,
            gain=gain,
            mpi_background=actual,
            primary_transmitted=bg[:, 0],
            gain_transmitted=gain * bg[:, 0],
        )
    )
    with atomic(out / "budget.csv") as tmp:
        budget.to_csv(tmp, index=False)
    sea = []
    for name, m in (("primary_LIM", model), ("gain_LIM", alt)):
        s = valid.ocean_scores(m)
        sea.append(s[s.region.eq("basin")].assign(method=name, month=month))
    with atomic(out / "ocean.parquet") as tmp:
        pd.concat(sea, ignore_index=True).to_parquet(tmp, index=False, compression="zstd")
    with atomic(out / "sources.csv") as tmp:
        meta.assign(month=month).to_csv(tmp, index=False)
    save_json(
        out / "complete.json",
        dict(plan=sha(PLAN), month=month, inputs=len(meta), model_id=model.meta["id"]),
    )
    record("warm_confirmation_scored", month=month, cases=len(meta), plan_sha256=sha(PLAN))


def summarize(out):
    plan = read(PLAN)
    ms = plan["months"]
    folders = [HERE / "scores" / f"{m:02}" for m in ms]
    for d in folders:
        if read(d / "complete.json")["plan"] != sha(PLAN):
            raise ValueError("Warm confirmation plan changed")
    win = pd.concat(
        [pd.read_csv(d / "windows.csv", dtype={"window": str}) for d in folders], ignore_index=True
    )
    total = win[(win.target == "total") & (win.window == "15")].copy()
    iid = total.set_index(["driver", "month"]).ratio
    total["iid_ratio"] = [
        iid.loc[(name.split("_")[0] + "_IID", m)] for name, m in zip(total.driver, total.month)
    ]
    total["passed"] = total.ratio.between(total.band_low, total.band_high) & (
        (total.ratio - 1).abs() < (total.iid_ratio - 1).abs()
    )
    counts = total.groupby("driver").passed.sum()
    histories = {}
    for d in folders:
        with np.load(d / "bootstrap.npz") as f:
            for name in f.files:
                histories.setdefault(name, []).append(f[name])
    hist = {k: np.stack(v).mean(axis=0) for k, v in histories.items()}
    rows = []
    for target in ("total", "within"):
        for window in (
            ("1", "5", "10", "15", "month") if target == "total" else ("1", "5", "10", "15")
        ):
            for method in NAMES:
                if method == "gain_LIM":
                    continue
                primary = hist["/".join((method, target, window))]
                challenger = hist["/".join(("gain_LIM", target, window))]
                record_estimate(
                    rows,
                    "warm",
                    "variance_log_error",
                    (target, window),
                    primary,
                    challenger,
                    8,
                    5,
                    "raw",
                )
                rows[-1]["comparator"] = method
    primary_controls = []
    for target in ("total", "within"):
        for window in (
            ("1", "5", "10", "15", "month") if target == "total" else ("1", "5", "10", "15")
        ):
            primary = hist["/".join(("primary_LIM", target, window))]
            for other in ("primary_native_Fourier", "primary_calibrated_Fourier", "primary_IID"):
                reference = hist["/".join((other, target, window))]
                record_estimate(
                    primary_controls,
                    "warm",
                    "variance_log_error",
                    (target, window),
                    reference,
                    primary,
                    8,
                    5,
                    "raw",
                )
                primary_controls[-1]["comparator"] = other
    intervals = pd.DataFrame(rows)
    criterion = intervals[
        (intervals.dimension == "total/15") & (intervals.comparator == "primary_LIM")
    ].iloc[0]
    strict = bool(criterion.ci_high < 0 and counts["gain_LIM"] >= counts["primary_LIM"])
    sources = pd.concat(
        [pd.read_csv(d / "sources.csv").assign(file_sha_verified=True) for d in folders],
        ignore_index=True,
    )
    budget = pd.concat([pd.read_csv(d / "budget.csv") for d in folders], ignore_index=True)
    sea = pd.concat([pd.read_parquet(d / "ocean.parquet") for d in folders], ignore_index=True)
    ocean = []
    for field, part in sea.groupby("field"):
        for metric in ("crps", "coverage90"):
            p = part.groupby(["member", "year", "g", "method"])[metric].mean().unstack("method")
            axes = p.index.to_frame(index=False)
            w, nm, ny = panel_weights(
                axes.member.to_numpy(),
                axes.year.to_numpy(),
                axes.g.to_numpy(),
                "warm",
                2000,
                112519,
            )
            record_estimate(
                ocean,
                "warm",
                metric,
                (field,),
                w @ p.primary_LIM.to_numpy(),
                w @ p.gain_LIM.to_numpy(),
                nm,
                ny,
                "physical",
            )
    if len(sources) != 440 or sources[["member", "year", "month"]].duplicated().any():
        raise ValueError("Expected 440 distinct, non-September member-months")
    references = []
    for role, member, year, g in checked_cases(plan):
        mark = read(case_dir(INPUT, member, year) / "complete.json")
        for field, source in mark["sources"].items():
            references.append(
                dict(
                    member=member,
                    year=year,
                    warming_K=g,
                    field=field,
                    version=source["version"],
                    checksum=source["source_checksum"],
                    location=source["url"],
                )
            )
    for name, v in (
        ("windows", win),
        ("pass15", total),
        ("paired", intervals),
        ("primary_controls", pd.DataFrame(primary_controls)),
        ("ocean_paired", pd.DataFrame(ocean)),
        ("variance_budget", budget),
        ("source_hashes", sources),
        ("source_datasets", pd.DataFrame(references)),
    ):
        with atomic(out / f"warm_confirmation_{name}.csv") as tmp:
            v.to_csv(tmp, index=False)
    with atomic(out / "warm_confirmation_ocean_cases.parquet") as tmp:
        sea.to_parquet(tmp, index=False, compression="zstd")
    result = dict(
        plan_sha256=sha(PLAN),
        cases=len(sources),
        months=ms,
        training_unchanged=True,
        primary_error=float(criterion.baseline),
        challenger_error=float(criterion.candidate),
        paired_change=float(criterion.difference),
        ci95=[float(criterion.ci_low), float(criterion.ci_high)],
        passes={k: int(v) for k, v in counts.items()},
        registered_improvement_met=strict,
    )
    save_json(out / "warm_confirmation_result.json", result)
    record("warm_confirmation_complete", **result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--setup", action="store_true")
    mode.add_argument("--month", type=int)
    mode.add_argument("--summary", type=Path)
    p.add_argument("--weights", type=Path)
    args = p.parse_args()
    if args.setup:
        setup()
    elif args.month is not None:
        run(args.month, args.weights)
    else:
        summarize(args.summary)
