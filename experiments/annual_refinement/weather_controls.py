"""Compare the native TC Fourier spectrum and i.i.d. weather at matched instantaneous variance."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ocean import GROUP_LIMITS
from records import WORK, record
from tuned_fourier import calibrated, harmonic_lags
from weather import kernels, model_terms, projected
from weather_bootstrap import accumulate, summarize

from annual_lim.layout import DEFAULT_WORK, read
from annual_lim.model import MonthlyModel
from annual_lim.statistics import variance
from annual_lim.validate import fourier_power
from tc_emulator.common import atomic


def daily_fourier_lags():
    lag = harmonic_lags(20.0, 3.0)
    for n in (28, 29, 30, 31):
        direct = [fourier_power(int(k)) for k in (1, 5, 10, 15, n)]
        np.testing.assert_allclose(kernels(n)[0] @ lag, direct, atol=5.0e-15)
    return lag


def compare(tag, weights, role, draws=2000, cohort=None):
    if role != "val":
        receipt = read(WORK / "FROZEN.json")
        if tag != receipt["stage"]:
            raise ValueError("Only the frozen candidate can open the external controls")
        cohort = receipt["cohort"]
    if cohort is None:
        if tag == "published_v1":
            cohort = "original"
        else:
            cohort = read(WORK / "parents" / (tag.removesuffix("_refined") + ".json"))["cohort"]
    rows = []
    history = {}
    fourier = daily_fourier_lags()
    for m in range(1, 13):
        month = f"{m:02}"
        released = MonthlyModel(DEFAULT_WORK / "weights", m)
        candidate = MonthlyModel(weights, m)
        if role != "val" and candidate.meta["id"] != receipt["months"][month]:
            raise ValueError("Weight mismatch")
        tuned = calibrated(candidate, tag, cohort, m)
        bands = read(DEFAULT_WORK / "fit" / (month + ".json"))["bands"]
        with np.load(
            WORK / "verification/published_v1" / role / month / "panel_moments.npz"
        ) as data:
            gs = data["g"]
            ys = data["year"]
            yr, first, inv = np.unique(ys, return_index=True, return_inverse=True)
            g = gs[first]
            import calendar

            nd = np.array([calendar.monthrange(y, m)[1] for y in yr])
            ops = [np.stack([kernels(int(k))[i] for k in nd]) for i in (0, 1)]
            views = {}
            for model, name in ((released, "v1"), (candidate, "candidate")):
                lags, bg, forced = model_terms(model, g)
                pure, center, _ = projected(model.cal, lags, g, nd, bg)
                views[name + "_LIM"] = (pure, center, forced)
                controls = [("Fourier", fourier), ("IID", np.r_[1.0, np.zeros(30)])]
                if name == "candidate":
                    controls.append(("tuned_Fourier", tuned))
                for method, h in controls:
                    amplitude = lags[0, 0] * model.cal.amplitude(g) ** 2
                    raw = bg[:, :1] + amplitude[:, None] * (ops[0] @ h)
                    cen = amplitude[:, None] * (ops[1] @ h)
                    views[name + "_" + method] = (raw, cen, forced)
            accumulate(data, views, inv, history, draws)
            low, high = GROUP_LIMITS
            for category, mask in (
                ("all", np.ones(len(gs), bool)),
                ("cold", gs < low),
                ("inside", (gs >= low) & (gs <= high)),
                ("warm", gs > high),
            ):
                if not mask.any():
                    continue
                raw = variance(data["mu"][mask], data["second"][mask])
                within = data["within"][mask].mean(axis=0)
                for method, (total, center, forced) in views.items():
                    ii = inv[mask]
                    simulated = total[ii].mean(axis=0) + np.var(gs[mask]) * forced
                    simcenter = center[ii].mean(axis=0)
                    for j, window in enumerate(("1", "5", "10", "15", "month")):
                        lo, hi = bands[j]
                        for target, p, o in (
                            ("total", simulated[j], raw[j]),
                            ("within", simcenter[j], within[j]),
                        ):
                            if target == "within" and window == "month":
                                continue
                            ratio = p / o
                            rows.append(
                                dict(
                                    model=method,
                                    month=m,
                                    group=category,
                                    window=window,
                                    target=target,
                                    ratio=ratio,
                                    reference=o,
                                    model_variance=p,
                                    abs_log_error=abs(np.log(ratio)),
                                    band_low=lo if target == "total" else np.nan,
                                    band_high=hi if target == "total" else np.nan,
                                    in_release_band=bool(lo <= ratio <= hi)
                                    if target == "total"
                                    else None,
                                )
                            )
    out = pd.DataFrame(rows)
    dest = WORK / "comparison" / tag / role / "weather_controls.csv"
    with atomic(dest) as tmp:
        out.to_csv(tmp, index=False)
    paired = summarize(history, out)
    with atomic(dest.with_name("fourier_intervals.csv")) as tmp:
        paired.to_csv(tmp, index=False)
    record("fourier_controls_done", tag=tag, role=role, metrics=len(out), paired_draws=draws)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--role", choices=["val", "test", "fresh", "ssp245_year"], default="val")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--cohort", choices=["original", "expanded", "extended"])
    a = p.parse_args()
    compare(a.tag, a.weights, a.role, a.draws, a.cohort)
