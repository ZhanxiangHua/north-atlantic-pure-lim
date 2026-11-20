"""Score the upper tail of local physical mixed-layer depth on development-validation cases."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from marine_regions import BOXES
from ocean import ocean_data
from paired_intervals import panel_weights, record_estimate
from records import WORK, record
from scipy.special import ndtr, ndtri
from spatial_check import field_loc

from annual_lim.layout import DEFAULT_WORK, read
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic

HERE = WORK / "deep_mixed_layer"
PROBS = (0.90, 0.95, 0.975, 0.99, 0.995)
DEPTHS = (1000, 2000, 3000)
TURN = ndtri(0.95)


def tail_map(z, shape):
    return z + (shape - 1) * np.maximum(z - TURN, 0.0)


def quantile(loc, sd, tau, shape, ceiling):
    return np.minimum(ceiling, np.maximum(0.01, np.exp(loc + sd * tail_map(ndtri(tau), shape))))


def exceedance(loc, sd, cut, shape, ceiling):
    z = (np.log(cut) - loc) / np.maximum(sd, 1e-12)
    p = np.where(sd > 1.0e-12, ndtr(-tail_map(z, 1 / shape)), (np.exp(loc) > cut).astype(float))
    return np.where(ceiling <= cut, 0.0, p)


def run(month):
    dest = HERE / f"{month:02}.parquet"
    if dest.is_file():
        return
    base = MonthlyModel(DEFAULT_WORK / "weights", month)
    receipt = read(WORK / "FROZEN.json")
    research = MonthlyModel(receipt["weights"], month)
    if research.meta["id"] != receipt["months"][f"{month:02}"]:
        raise ValueError("Unexpected research checkpoint")
    train, tm = ocean_data("extended", month, "train")
    held, meta = ocean_data("original", month, "val")
    if (
        tm.year.nunique() != 57
        or tm.member.nunique() != 35
        or meta.year.nunique() != 6
        or meta.member.nunique() != 7
        or set(tm.member) & set(meta.member)
    ):
        raise ValueError("Expected the declared training and development panels")
    sea = base.rep.blocks["ocean"].valid[0].ravel()
    training_mld = train[:, 1].reshape(len(train), -1)[:, sea]
    cap = np.where(np.isfinite(training_mld), training_mld, -np.inf).max(axis=0)
    if not np.all(np.isfinite(cap)) or not (cap > 0).all():
        raise ValueError("Missing training maximum on the fixed wet grid")
    del train, training_mld
    truth = held[:, 1].reshape(len(meta), -1)[:, sea].astype(float)
    yy, xx = np.meshgrid(base.rep.lat, base.rep.lon, indexing="ij")
    y, x = yy.ravel()[sea], xx.ravel()[sea]
    region = {}
    for key in ("whole_basin", "subpolar_Atlantic"):
        s, n, w, e = BOXES[key]
        region[key] = np.cos(np.deg2rad(y)) * ((y >= s) & (y <= n) & (x >= w) & (x <= e))
    rows = []
    for year in sorted(meta.year.unique()):
        sub = meta.year == year
        submeta = meta.loc[sub, ["year", "member", "g"]]
        g = float(submeta.g.iloc[0])
        obs = truth[sub]
        for model, methods in (
            (base, [("published_log", 1.0, np.inf)]),
            (
                research,
                [
                    ("research_log", 1.0, np.inf),
                    ("upper_075", 0.75, np.inf),
                    ("upper_050", 0.5, np.inf),
                    ("training_max", 1.0, cap),
                ],
            ),
        ):
            loc, F, mask = field_loc(model, model.bg.mean(np.array([g]))[0], model.factor(g), "mld")
            np.testing.assert_array_equal(mask, sea)
            loc, sd = loc[sea], np.linalg.norm(F[sea], axis=1)
            for name, shape, ceiling in methods:
                for box, ar in region.items():
                    weights = np.isfinite(obs) * ar
                    weights = weights / weights.sum(axis=1, keepdims=True)

                    def average(v):
                        return np.sum(weights * np.where(np.isfinite(v), v, 0.0), axis=1)

                    for tau in PROBS:
                        q = quantile(loc, sd, tau, shape, ceiling)
                        error = obs - q
                        score = average(np.maximum(tau * error, (tau - 1) * error))
                        rows.append(
                            submeta.assign(
                                month=month,
                                model=name,
                                region=box,
                                diagnostic="quantile",
                                level=tau,
                                score=score,
                                observed_rate=average(obs > q),
                                expected_rate=1 - tau,
                            )
                        )
                    for cut in DEPTHS:
                        p = exceedance(loc, sd, cut, shape, ceiling)
                        event = obs > cut
                        rows.append(
                            submeta.assign(
                                month=month,
                                model=name,
                                region=box,
                                diagnostic="threshold",
                                level=float(cut),
                                score=average((p - event) ** 2),
                                observed_rate=average(event),
                                expected_rate=average(p),
                            )
                        )
    with atomic(dest) as tmp:
        pd.concat(rows, ignore_index=True).to_parquet(tmp, index=False, compression="zstd")
    record("deep_ocean_tail_validation_month", month=month, validation_cases=len(meta), methods=5)


def summarize(out):
    frame = pd.concat(
        [pd.read_parquet(HERE / f"{m:02}.parquet") for m in range(1, 13)], ignore_index=True
    )
    full = frame.groupby(["region", "model", "diagnostic", "level"])[
        ["score", "expected_rate", "observed_rate"]
    ].mean()
    with atomic(out / "deep_mixed_layer_validation.csv") as tmp:
        full.reset_index().to_csv(tmp, index=False)
    with atomic(out / "deep_mixed_layer_validation_cases.parquet") as tmp:
        frame.to_parquet(tmp, index=False, compression="zstd")
    intervals = []
    for (box, kind, level), data in frame.groupby(["region", "diagnostic", "level"]):
        panels = data.groupby(["member", "year", "g", "model"]).score.mean().unstack("model")
        meta = panels.index.to_frame(index=False)
        for band in ("all", "cold", "inside", "warm"):
            w, nm, ny = panel_weights(
                meta.member.to_numpy(), meta.year.to_numpy(), meta.g.to_numpy(), band, 2000, 120077
            )
            ref = w @ panels["research_log"].to_numpy()
            for name in panels.columns:
                if name == "research_log":
                    continue
                row = []
                record_estimate(
                    row,
                    band,
                    kind,
                    (box, str(level), name),
                    ref,
                    w @ panels[name].to_numpy(),
                    nm,
                    ny,
                    "physical",
                )
                intervals.extend(row)
    with atomic(out / "deep_mixed_layer_validation_intervals.csv") as tmp:
        pd.DataFrame(intervals).to_csv(tmp, index=False)
    record(
        "deep_ocean_tail_validation_complete", validation_cases=504, individual_scores=len(frame)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--month", type=int, choices=range(1, 13))
    mode.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.month is None:
        summarize(args.summary)
    else:
        run(args.month)
