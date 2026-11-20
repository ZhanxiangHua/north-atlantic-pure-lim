"""Matched 15/30/57-year learning curves at fixed training members, temperature support and evaluation maps."""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from crossfit import Crossfit, choose_rectangle
from records import WORK, record

from annual_lim.layout import read
from tc_emulator.common import save_json


def ready(month):
    cohorts = ["original"]
    for tag in ("expanded", "extended"):
        if (WORK / "features" / tag / f"{month:02}/complete.json").is_file() and (
            WORK / "wind_moments" / tag / "train" / f"{month:02}.npz"
        ).is_file():
            cohorts.append(tag)
    return cohorts


def rectangle(engine, reference, detail, lo, hi):
    m = engine.meta
    tm = np.array(detail["training_members"])
    hm = np.array(detail["held_members"])
    hy = np.array(detail["held_years"])
    orig_y = set(reference.meta.year)
    # The exact same held-out members, years and warming support are used for every rung.
    base_years = set(detail["training_years"])
    added = m[(m.g >= lo) & (m.g <= hi) & ~m.year.isin(orig_y)].year.unique()
    use = sorted(base_years | set(map(int, added)))
    train = np.flatnonzero(m.member.isin(tm) & m.year.isin(use))
    held = np.flatnonzero(m.member.isin(hm) & m.year.isin(hy))
    audit = dict(detail, training_years=use, original_training_range=[float(lo), float(hi)])
    if np.intersect1d(use, hy).size or not len(held):
        raise ValueError("The learning curve leaked a held-out year")
    return train, held, audit


def run(lane, lanes):
    end = datetime.fromisoformat(
        read(Path(__file__).with_name("design.json"))["end_utc"].replace("Z", "+00:00")
    ).timestamp()
    root = WORK / "learning_curve" / f"lane{lane}"
    root.mkdir(exist_ok=True, parents=True)
    arms = (
        "depth_regularized",
        "depth_rank24",
        "compact",
        "more_ocean",
        "wide",
        "block_covariance",
        "depth",
    )
    schemes = ("cold", "inside", "warm")
    sizes = (14, 21, 28)
    # A finite, registered split grid prevents repeated unchanged experiments while each new data rung is ingesting.
    # New cohorts reopen previously completed splits so every rung is compared on the same held-out maps.
    engines = {}
    old_month = None
    while time.time() < end:
        ran = 0
        for epoch in range(8):
            for slot in range(lane, 12 * 3 * 3 * len(arms), lanes):
                if time.time() >= end:
                    break
                month = 1 + ((slot // (3 * 3 * len(arms)) + epoch * 5) % 12)
                regime = schemes[(slot // (3 * len(arms))) % 3]
                size = sizes[(slot // len(arms)) % 3]
                arm = arms[slot % len(arms)]
                complete = ready(month)
                if len(complete) == 1:
                    continue
                name = f"e{epoch:02}-s{slot:04}"
                path = root / (name + ".json")
                prev = read(path) if path.is_file() else dict(results={})
                absent = [c for c in complete if c not in prev["results"]]
                if not absent:
                    continue
                if old_month != month:
                    engines = {}
                    old_month = month
                if "original" not in engines:
                    engines["original"] = Crossfit("original", month)
                engine = engines["original"]
                seed = 932026 + epoch * 100003 + slot * 109
                tr, held, detail = choose_rectangle(engine.meta, regime, size, seed)
                lo, hi = engine.meta.g.iloc[tr].min(), engine.meta.g.iloc[tr].max()
                for c in absent:
                    if c not in engines:
                        engines[c] = Crossfit(c, month)
                    e = engines[c]
                    rr = rectangle(e, engine, detail, lo, hi)
                    outcome = e.round(slot, arm, regime, size, seed, True, rectangle=rr)
                    prev["results"][c] = outcome
                    prev.update(
                        month=month,
                        scheme=regime,
                        members=size,
                        arm=arm,
                        seed=seed,
                        epoch=epoch,
                        slot=slot,
                    )
                    prev["last_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    save_json(path, prev)
                    record(
                        "learning_curve_refit",
                        cohort=c,
                        lane=lane,
                        month=month,
                        scheme=regime,
                        members=size,
                        arm=arm,
                        cases=outcome["train_cases"],
                        eval_cases=outcome["held_cases"],
                        split=name,
                    )
                    ran += 1
                # Refresh the cohort list at a predictable cadence as new archives become available.
            if time.time() >= end:
                break
        if ran == 0:
            time.sleep(min(45.0, max(0.0, end - time.time())))
    record("learning_curve_end", lane=lane, requested_end=end)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lane", type=int, required=True)
    p.add_argument("--lanes", type=int, default=3)
    a = p.parse_args()
    run(a.lane, a.lanes)
