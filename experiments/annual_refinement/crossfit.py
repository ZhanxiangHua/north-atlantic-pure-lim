"""Run time-bounded, paired member-and-year refits. Every trial uses a new held-out rectangle."""
import argparse
import copy
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from backbone import Bank, Recipe, build
from joint import PhysicalValidation
from ocean import joint_input, ocean_data, solve_basis
from records import WORK, record
from weather import cache_read, model_terms, projected, train_cal, year_table

from annual_lim.layout import DEFAULT_WORK, read
from annual_lim.model import MonthlyModel
from annual_lim.statistics import variance
from annual_lim.train import fit_dynamics
from tc_emulator.common import save_json

# Candidate controls share an unchanged 120-dimensional wind basis in this arm.
ARMS = {
    "reference": Recipe(),
    "more_ocean": Recipe(sst=48, ocean=72, sst_basis="cv_sst", ocean_basis="cv_ocean"),
    "depth": Recipe(sst=48, ocean=72, sst_basis="cv_sst", ocean_basis="cv_depth", tail_penalty=1.0),
    "depth_regularized": Recipe(
        sst=48,
        ocean=72,
        sst_basis="cv_sst",
        ocean_basis="cv_depth",
        ridge=0.3,
        tail_penalty=25.0,
        mean_curve=0.5,
        warming_penalty=0.0,
    ),
    "depth_rank24": Recipe(
        sst=48,
        ocean=72,
        sst_basis="cv_sst",
        ocean_basis="cv_depth",
        ridge=0.3,
        tail_penalty=25.0,
        mean_curve=0.5,
        warming_penalty=0.0,
        regression_rank=24,
    ),
    "block_covariance": Recipe(
        sst=48,
        ocean=72,
        sst_basis="cv_sst",
        ocean_basis="cv_depth",
        ridge=0.3,
        tail_penalty=25.0,
        mean_curve=0.5,
        warming_penalty=0.0,
        covariance="block_active",
        covariance_slope_shrink=0.5,
    ),
    "compact": Recipe(
        sst=32,
        ocean=48,
        sst_basis="cv_sst",
        ocean_basis="cv_depth",
        ridge=0.3,
        tail_penalty=25.0,
        mean_curve=0.5,
        warming_penalty=0.0,
    ),
    "wide": Recipe(
        sst=72,
        ocean=96,
        sst_basis="cv_sst",
        ocean_basis="cv_depth",
        ridge=0.3,
        tail_penalty=25.0,
        mean_curve=0.5,
        warming_penalty=0.0,
    ),
}
REFERENCE = replace(ARMS["reference"], sst_basis="cv_sst", ocean_basis="cv_ocean")


def window_panel(all_obs, ix):
    out = {k: all_obs[k][ix] for k in ("year", "g", "member", "mu", "second", "within", "length")}
    out["unique"] = np.unique(out["year"])
    raw = []
    unbias = []
    centered = []
    for y in out["unique"]:
        ii = out["year"] == y
        u = out["mu"][ii]
        v = variance(u, out["second"][ii])
        n = ii.sum()
        raw.append(v)
        unbias.append(v + np.square(u - u.mean(axis=0)).sum(axis=(0, 2)) / (n * (n - 1)))
        centered.append(out["within"][ii].mean(axis=0))
    out["by_year_raw"] = np.stack(raw)
    out["by_year_cov"] = np.stack(unbias)
    out["by_year_within"] = np.stack(centered)
    # These member-held-out controls intentionally fit window targets alone. Full cohort fits also use all four wind ACFs.
    out["by_year_acf"] = np.zeros((len(raw), 4, 16))
    return out


def choose_rectangle(meta, scheme, ntrain, seed):
    rng = np.random.default_rng(seed)
    years, g = np.unique(meta.year, return_index=True)
    ys = years[np.argsort(meta.g.to_numpy()[g])]
    members = rng.permutation(sorted(meta.member.unique()))
    hm = members[:7]
    tm = members[7 : 7 + ntrain]
    if scheme == "inside":
        groups = np.array_split(ys[1:-1], 3)
        hy = np.array([rng.choice(part) for part in groups])
        ty = ys[~np.isin(ys, hy)]
    elif scheme == "cold":
        n = max(3, len(ys) // 5)
        hy = rng.choice(ys[:n], size=min(3, n), replace=False)
        ty = ys[n:]
    else:
        n = max(3, len(ys) // 5)
        hy = rng.choice(ys[-n:], size=min(3, n), replace=False)
        ty = ys[:-n]
    # Refit on a different subset even when the outer temperature analogue chooses the same boundary.
    count = max(8, round(0.83 * len(ty)))
    ty = np.sort(rng.choice(ty, size=count, replace=False))
    train = np.flatnonzero(meta.member.isin(tm) & meta.year.isin(ty))
    held = np.flatnonzero(meta.member.isin(hm) & meta.year.isin(hy))
    assert set(tm).isdisjoint(hm) and set(ty).isdisjoint(hy) and len(train) >= 100
    return (
        train,
        held,
        dict(
            training_members=sorted(tm.tolist()),
            held_members=sorted(hm.tolist()),
            training_years=sorted(map(int, ty)),
            held_years=sorted(map(int, hy)),
        ),
    )


class Crossfit:
    def __init__(self, cohort, month):
        self.bank = Bank(cohort, month)
        self.ocean, self.meta = ocean_data(cohort, month, "train")
        self.obs = cache_read(cohort, "train", month)
        self.published = MonthlyModel(DEFAULT_WORK / "weights", month)
        assert np.array_equal(self.obs["member"], self.meta.member) and np.array_equal(
            self.obs["year"], self.meta.year
        )
        y = self.bank.geo.lat
        self.area = np.broadcast_to(np.cos(np.deg2rad(y))[:, None], self.ocean.shape[-2:])

    def round(self, index, arm, scheme, member_size, seed, temporal, rectangle=None):
        train, held, split = (
            choose_rectangle(self.meta, scheme, member_size, seed)
            if rectangle is None
            else rectangle
        )
        bank = copy.copy(self.bank)
        bank.bases = bank.bases.copy()
        bank.scores = bank.scores.copy()
        # Every ocean EOF, conditional transfer, covariance and autoregression is fitted anew here.
        # The shared atmospheric spatial basis is fixed for this conditional sensitivity experiment.
        selected = [("reference", REFERENCE), (arm, ARMS[arm])]
        for name, data, vert, depth in [
            ("cv_sst", self.ocean[:, :1], [1.0], 0.0),
            ("cv_ocean", joint_input(self.ocean), [0.5, 0.5], 0.0),
            ("cv_depth", joint_input(self.ocean), [0.5, 0.5], 0.7),
        ]:
            n = max(
                [
                    r.sst if name == "cv_sst" else r.ocean
                    for _, r in selected
                    if name in (r.sst_basis, r.ocean_basis)
                ]
                or [0]
            )
            if not n:
                continue
            b = solve_basis(data[train], self.area, vert, n, depth_power=depth)
            bank.bases[name] = b
            bank.scores["train_" + name] = b.transform(data)
        val = PhysicalValidation(
            bank, self.ocean[held], self.meta.iloc[held], self.obs["mu"][held, -1]
        )
        train_panel = window_panel(self.obs, train) if temporal else None
        held_panel = window_panel(self.obs, held) if temporal else None
        rows = []
        for name, r in selected:
            m = build(bank, r, train)
            zs = bank.z(r, "train")
            ws = val.wind_errors(m, zs[held])
            scored = val.ocean_scores(m)
            stats = {
                field + "_" + region + "_crps": float(group.crps.mean())
                for (field, region), group in scored.groupby(["field", "region"])
            }
            stats["conditional_wind_rmse"] = float(np.sqrt(ws.conditional_mse.mean()))
            stats["forcing_wind_rmse"] = float(np.sqrt(ws.forcing_only_mse.mean()))
            if temporal:
                daily = bank.daily(r, "train")
                nd = bank.scores["train_days"][train]
                expected = m.mean.predict(m.bg, bank.scores["train_g"][train], zs[train])
                paths = [daily[i, :l] for i, l in zip(train, nd)]
                m.base, _, _ = fit_dynamics(
                    paths, expected, {"lim_ridge_candidates": [0.0, 0.001, 0.01, 0.1]}
                )
                _, gt, _ = year_table(train_panel)
                _, gh, nh = year_table(held_panel)
                m.cal = replace(
                    self.published.cal,
                    center=float(bank.scores["train_g"][train].mean()),
                    g_min=float(gt.min()),
                    g_max=float(gt.max()),
                )
                lags, bg, forced = model_terms(m, np.r_[gt, gh])
                # Matched training losses supply a fair covariance/EOF comparison on the held-out rectangle.
                cal, _ = train_cal(
                    m, train_panel, lags, bg[: len(gt)], 5, 20, 0.5, False, lag_weight=0.0
                )
                p, c, _ = projected(cal, lags, gh, nh, bg[len(gt) :])
                raw = variance(held_panel["mu"], held_panel["second"])
                ratio = (p.mean(axis=0) + np.var(gh) * forced) / raw
                wratio = c.mean(axis=0)[:4] / held_panel["within"].mean(axis=0)[:4]
                stats.update(
                    wind_total=float(np.abs(np.log(ratio[1:])).mean()),
                    wind_within=float(np.abs(np.log(wratio[1:])).mean()),
                    wind15_ratio=float(ratio[3]),
                    effective_radius=float(cal.retention * m.base.rho),
                )
            rows.append(dict(arm=name, **stats))
        return dict(
            index=index,
            cohort=bank.cohort,
            month=bank.month,
            scheme=scheme,
            seed=seed,
            member_size=member_size,
            temporal=temporal,
            split=split,
            results=rows,
            train_cases=len(train),
            held_cases=len(held),
        )


def run(lane):
    spec = read(Path(__file__).with_name("design.json"))
    deadline = datetime.fromisoformat(spec["end_utc"].replace("Z", "+00:00")).timestamp()
    root = WORK / "crossfit" / f"lane{lane}"
    root.mkdir(exist_ok=True, parents=True)
    index = 0
    while (root / f"{index:05}.json").is_file():
        index += 1
    previous = None
    engine = None
    while time.time() < deadline:
        # Calendar months and regimes rotate before any member-year split is repeated.
        # Each lane covers multiple seasons. Expanded training enters as soon as its prepared moments exist.
        batch = index // 4
        month = 1 + (batch * 5 + lane * 3) % 12
        expanded = (WORK / "features/expanded" / f"{month:02}" / "complete.json").is_file() and (
            WORK / "wind_moments/expanded/train" / f"{month:02}.npz"
        ).is_file()
        cohort = (
            "expanded" if expanded and (batch // 12 + index % 4 + lane) % 3 != 0 else "original"
        )
        engine_key = (cohort, month)
        seed = 20261120 + lane * 10000000 + index * 853
        regime = ("inside", "cold", "warm")[(index + batch // 12 + lane) % 3]
        size = (14, 21, 28)[(index // 3 + lane) % 3]
        candidates = [k for k in ARMS if k != "reference"]
        arm = candidates[(index + lane * 2 + batch // 12) % len(candidates)]
        began = time.monotonic()
        try:
            if previous != engine_key:
                engine = None
                engine = Crossfit(cohort, month)
                previous = engine_key
            outcome = engine.round(index, arm, regime, size, seed, temporal=True)
            outcome["elapsed_s"] = round(time.monotonic() - began, 2)
            outcome["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            save_json(root / f"{index:05}.json", outcome)
            record(
                "paired_refit_done",
                lane=lane,
                index=index,
                month=month,
                cohort=cohort,
                regime=regime,
                arm=arm,
                seconds=outcome["elapsed_s"],
            )
            print(outcome["utc"], index, cohort, month, arm, regime, flush=True)
        except Exception as exc:
            record(
                "paired_refit_error",
                lane=lane,
                index=index,
                month=month,
                cohort=cohort,
                error=repr(exc),
            )
            traceback.print_exc()
            raise
        index += 1
    record("paired_refit_end", lane=lane, rounds=index, requested_end=spec["end_utc"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lane", type=int, required=True)
    run(p.parse_args().lane)
