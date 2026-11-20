"""Fit training-only spatial bases once, then cache aligned training and validation coordinates."""
import argparse
import time
from dataclasses import fields

import numpy as np
import scipy.linalg as la
from ocean import solve_basis, split_basis
from records import WORK, record

from annual_lim.layout import CONFIG, DEFAULT_WORK
from annual_lim.model import MonthlyModel
from annual_lim.train import months
from tc_emulator.common import atomic, digest, save_json
from tc_emulator.representation import BLOCKS, EOF, eof, stack_fields, vertical_weights

MAX_OCEAN = 128
MAX_WIND = 192


def save_block(path, obj):
    with atomic(path) as tmp:
        np.savez(tmp, **{f.name: getattr(obj, f.name) for f in fields(EOF)})


def load_block(path):
    with np.load(path, allow_pickle=False) as f:
        return EOF(**{k: f[k] for k in f.files})


def forced_basis(base, values, gs, count):
    """Reserve space for a linear and quadratic warming map, then fit the orthogonal remainder."""
    valid = base.valid
    x = values.reshape(len(values), -1)[:, valid.ravel()].astype(float)
    x = np.where(np.isfinite(x), x - base.mean[valid], 0.0) * base.weights[valid]
    g = (gs - gs.mean()) / gs.std()
    coef = la.lstsq(np.column_stack((np.ones(len(g)), g, g * g)), x, check_finite=False)[0][1:]
    protected = la.orth(coef.T)
    rem = x - (x @ protected) @ protected.T
    k = count - protected.shape[1]
    vals, vec = la.eigh(
        rem @ rem.T, subset_by_index=[len(rem) - k, len(rem) - 1], check_finite=False
    )
    tail = (rem.T @ vec[:, ::-1]) / np.sqrt(vals[::-1])
    u = np.column_stack((protected, tail))
    var = np.var(x @ u, axis=0, ddof=1)
    return EOF(base.mean, base.weights, valid, u, var, base.total_variance)


def make(cohort, month):
    root = WORK / "features" / cohort / f"{month:02}"
    if (root / "complete.json").is_file():
        return
    start = time.monotonic()
    record("feature_bank_start", month=month, cohort=cohort)
    cfg = CONFIG if cohort == "original" else WORK / "configs" / cohort
    train, tr, grid = months(cfg, WORK / "data", month, "train")
    val, vr, _ = months(CONFIG, WORK / "data", month, "val")
    rep = MonthlyModel(DEFAULT_WORK / "weights", month).rep
    area = np.broadcast_to(np.cos(np.deg2rad(rep.lat))[:, None], (len(rep.lat), len(rep.lon)))
    gs = np.array([r["g"] for r in tr])
    panels = {"train": train, "val": val}
    root.mkdir(parents=True, exist_ok=True)
    state = {}
    for role, rs in [("train", tr), ("val", vr)]:
        for name in ("member", "year", "g", "days"):
            state[f"{role}_{name}"] = np.array([r[name] for r in rs])
    for block, names in BLOCKS.items():
        values = {
            role: np.stack([stack_fields(d, names, True) for d in data])
            for role, data in panels.items()
        }
        bases = {"published_" + block: rep.blocks[block]}
        if block in ("sst", "ocean"):
            vert = vertical_weights(names, rep.plev)
            std = solve_basis(values["train"], area, vert, MAX_OCEAN)
            bases[block] = std
            bases[block + "_forced"] = forced_basis(std, values["train"], gs, MAX_OCEAN)
            if block == "ocean":
                for p, token in ((0.35, "035"), (0.7, "070"), (1.0, "100")):
                    weighted = solve_basis(values["train"], area, vert, MAX_OCEAN, depth_power=p)
                    bases["ocean_d" + token] = weighted
                    if p == 0.7:
                        bases["ocean_d070_forced"] = forced_basis(
                            weighted, values["train"], gs, MAX_OCEAN
                        )
                bases["ocean_split070"] = split_basis(
                    values["train"], area, MAX_OCEAN, depth_power=0.7
                )
        elif cohort != "original":
            bases["new_" + block] = eof(
                values["train"],
                area,
                vertical_weights(names, rep.plev),
                rep.blocks[block].basis.shape[1],
                9152026,
            )
        for name, b in bases.items():
            save_block(root / (name + ".npz"), b)
            for role, v in values.items():
                state[role + "_" + name] = b.transform(v)
        del values, bases
    winds = [d["wind"] for d in train]
    maxbasis = eof(np.concatenate(winds), area, np.ones(4) / 4, MAX_WIND, 9152026)
    for key, w in (("published_wind", rep.blocks["wind"]), ("wind", maxbasis)):
        save_block(root / (key + ".npz"), w)
        for role, panel in panels.items():
            nd = max(len(d["wind"]) for d in panel)
            scores = np.full((len(panel), nd, w.basis.shape[1]), np.nan)
            for i, d in enumerate(panel):
                scores[i, : len(d["wind"])] = w.transform(d["wind"])
            # Separating large daily arrays avoids opening them for monthly-only transfer studies.
            state[role + "_" + key + "_monthly"] = np.nanmean(scores, axis=1)
            with atomic(root / (role + "_" + key + "_daily.npy")) as tmp:
                np.save(tmp, scores, allow_pickle=False)
    with atomic(root / "scores.npz") as tmp:
        np.savez(tmp, **state)
    save_json(
        root / "complete.json",
        dict(
            cohort=cohort,
            month=month,
            train=tr,
            val=vr,
            rank_max=MAX_OCEAN,
            max_wind=MAX_WIND,
            input_digest=digest([tr, vr]),
            grid=list(map(len, grid)),
        ),
    )
    record(
        "feature_bank_done",
        month=month,
        cohort=cohort,
        seconds=round(time.monotonic() - start, 2),
        train=len(tr),
        val=len(vr),
    )
    print("Features ready", cohort, month, len(tr), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default="original")
    p.add_argument("--month", type=int, required=True)
    a = p.parse_args()
    make(a.cohort, a.month)
