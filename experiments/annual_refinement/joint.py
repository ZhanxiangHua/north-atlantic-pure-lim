"""Score physical ocean marginals and conditional monthly steering on identical validation cases."""
import argparse
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from backbone import Bank, Recipe, build
from ocean import ocean_data
from records import WORK, record
from scipy.special import ndtr, ndtri

from tc_emulator.common import atomic, save_json
from tc_emulator.diagnostics import region_area, steering

PROBS = (np.arange(20) + 0.5) / 20
QUANTILES = ndtri(PROBS)


def mapped_quantiles(name, loc, scale, points):
    q = loc[None] + points[:, None] * scale
    if name == "mld":
        return np.maximum(0.01, np.exp(q))
    if name == "gamma":
        return np.maximum(0.0, q)
    return q


def physical_moments(name, loc, scale):
    if name == "sst":
        return loc, scale**2
    sd = np.maximum(scale, 1e-12)
    if name == "gamma":
        z = loc / sd
        density = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        mean = loc * ndtr(z) + scale * density
        second = (loc * loc + scale**2) * ndtr(z) + loc * scale * density
    else:
        floor = 0.01
        p = ndtr((np.log(floor) - loc) / sd)

        def moment(k):
            return floor**k * p + np.exp(k * loc + 0.5 * k * k * scale**2) * ndtr(
                (loc + k * scale**2 - np.log(floor)) / sd
            )

        mean, second = moment(1), moment(2)
    return mean, np.maximum(second - mean * mean, 0.0)


class PhysicalValidation:
    def __init__(self, bank, truth=None, meta=None, wind=None):
        self.bank = bank
        if truth is None:
            self.truth, self.meta = ocean_data("original", bank.month, "val")
            with np.load(WORK / "wind_moments/original/val" / f"{bank.month:02}.npz") as f:
                self.wind = f["mu"][:, -1]
                assert np.array_equal(f["member"], self.meta.member) and np.array_equal(
                    f["year"], self.meta.year
                )
        else:
            self.truth, self.meta, self.wind = truth, meta, wind
        y, x = np.meshgrid(bank.geo.lat, bank.geo.lon, indexing="ij")
        area = region_area(bank.geo.lat, bank.geo.lon)
        self.area = area
        self.regions = {
            "basin": area.ravel(),
            "corridor": (area * ((y >= 10) & (y <= 30) & (x >= 280) & (x <= 340))).ravel(),
        }
        self.log = []
        self.reference_mapping = {"sst": ("sst", 0), "mld": ("ocean", 0), "gamma": ("ocean", 1)}

    def wind_errors(self, model, z, winds=None, meta=None):
        meta = self.meta if meta is None else meta
        winds = self.wind if winds is None else winds
        g = meta.g.to_numpy()
        target = model.mean.predict(model.bg, g, z)
        forced = model.mean.predict(model.bg, g, model.bg.mean(g))
        arrays = meta[["member", "year", "g"]].copy()
        for name, pc in (("conditional_mse", target), ("forcing_only_mse", forced)):
            mapped = steering(model.rep.blocks["wind"].inverse(pc)) * np.sqrt(self.area / 2)
            diff = mapped.reshape(len(pc), -1) - winds
            arrays[name] = np.einsum("ij,ij->i", diff, diff)
        return arrays

    def ocean_scores(self, model, residual_variance=None):
        data, meta = self.truth, self.meta
        years, idx = np.unique(meta.year, return_index=True)
        gs = meta.g.to_numpy()[idx]
        mus = model.bg.mean(gs)
        factors = [model.factor(g) for g in gs]
        out = []
        for field_id, (field, (bname, channel)) in enumerate(self.reference_mapping.items()):
            b = model.rep.blocks[bname]
            base = self.bank.geo.blocks[bname]
            fixed = base.valid[channel].ravel()
            valid = b.valid[channel].ravel()
            start = sum(
                model.rep.blocks[key].basis.shape[1]
                for key in (("sst",) if bname == "ocean" else ())
            )
            end = start + b.basis.shape[1]
            valid_offset = int(b.valid[:channel].sum())
            J = (
                b.basis[valid_offset : valid_offset + valid.sum()]
                / b.weights[channel].ravel()[valid, None]
            )
            for t, year in enumerate(years):
                selected = meta.year.to_numpy() == year
                obs = data[selected, field_id].reshape(selected.sum(), -1)[:, fixed].astype(float)
                finite = np.isfinite(obs)
                # All candidates are evaluated on the original wet mask. A missing fitted feature gets the published climatology.
                loc = base.mean[channel].ravel().copy()
                sd = np.zeros_like(loc)
                loc[valid] = b.mean[channel].ravel()[valid] + J @ mus[t, start:end]
                sd[valid] = np.linalg.norm(J @ factors[t][start:end], axis=1)
                if residual_variance is not None:
                    sd[valid] = np.sqrt(sd[valid] ** 2 + residual_variance(field, gs[t])[valid])
                loc, sd = loc[fixed], sd[fixed]
                mean, var = physical_moments(field, loc, sd)
                quant = mapped_quantiles(field, loc, sd, QUANTILES)
                lo, hi = mapped_quantiles(field, loc, sd, ndtri([0.05, 0.95]))
                centered = np.where(finite, obs - mean, 0.0)
                score = np.zeros_like(obs)
                for tau, q in zip(PROBS, quant):
                    diff = np.where(finite, obs - q, 0.0)
                    score += 2 * (tau - (diff < 0)) * diff / len(PROBS)
                n = finite.sum(axis=0)
                true_mean = np.where(finite, obs, 0.0).sum(axis=0) / np.maximum(n, 1)
                true_var = np.where(finite, (obs - true_mean) ** 2, 0.0).sum(axis=0) / np.maximum(
                    n - 1, 1
                )
                for region, area in self.regions.items():
                    a = area[fixed]
                    w = finite * a
                    sw = w.sum(axis=1)
                    coverage = (obs >= lo) & (obs <= hi) & finite
                    wv = a * (n > 1)
                    block = meta.loc[selected, ["member", "year", "g"]].copy()
                    block["field"] = field
                    block["region"] = region
                    block["crps"] = (w * score).sum(axis=1) / sw
                    block["mean_mse"] = (w * centered**2).sum(axis=1) / sw
                    block["bias"] = -(w * centered).sum(axis=1) / sw
                    block["coverage90"] = (w * coverage).sum(axis=1) / sw
                    block["mask_coverage"] = w[:, valid[fixed]].sum(axis=1) / sw
                    block["model_var"] = (wv @ var) / wv.sum()
                    block["observed_var"] = (wv @ true_var) / wv.sum()
                    out.append(block)
        return pd.concat(out, ignore_index=True)


def recipe_set(phase):
    base = Recipe()
    variants = {"published": base}
    if phase == "spatial":
        for s, o in (
            (18, 18),
            (32, 32),
            (32, 48),
            (48, 48),
            (48, 72),
            (48, 96),
            (72, 96),
            (72, 128),
            (96, 128),
        ):
            for layout in ("ocean", "ocean_d070", "ocean_d100", "ocean_split070"):
                name = f"s{s}_o{o}_{layout}"
                variants[name] = replace(base, sst=s, ocean=o, sst_basis="sst", ocean_basis=layout)
        for s, o in ((32, 48), (48, 72), (72, 96)):
            variants[f"s{s}_o{o}_trend"] = replace(
                base, sst=s, ocean=o, sst_basis="sst_forced", ocean_basis="ocean_d070_forced"
            )
    elif phase == "covariance":
        for s, o in ((18, 18), (32, 48), (48, 72), (72, 96)):
            for law in ("pooled", "active", "block", "block_active"):
                for curve in (0.0, 0.5, 1.0):
                    r = replace(
                        base,
                        sst=s,
                        ocean=o,
                        sst_basis="sst",
                        ocean_basis="ocean_d070",
                        covariance=law,
                        mean_curve=curve,
                    )
                    variants[f"s{s}_o{o}_{law}_curve{curve:g}"] = r
    return variants


def run(cohort, month, phase):
    root = WORK / "joint" / cohort / phase / f"{month:02}"
    if (root / "complete.json").exists():
        return
    start = time.monotonic()
    record("joint_start", month=month, cohort=cohort, phase=phase)
    bank = Bank(cohort, month)
    val = PhysicalValidation(bank)
    candidates = recipe_set(phase)
    oceans = []
    winds = []
    for name, r in candidates.items():
        m = build(bank, r)
        oceans.append(val.ocean_scores(m).assign(recipe=name))
        winds.append(val.wind_errors(m, bank.z(r, "val")).assign(recipe=name))
    for label, items in (("oceans", oceans), ("winds", winds)):
        with atomic(root / (label + ".parquet")) as tmp:
            pd.concat(items, ignore_index=True).to_parquet(tmp, index=False, compression="zstd")
    save_json(
        root / "complete.json",
        dict(
            month=month,
            cohort=cohort,
            phase=phase,
            recipes={k: asdict(v) for k, v in candidates.items()},
        ),
    )
    record(
        "joint_done",
        month=month,
        cohort=cohort,
        phase=phase,
        recipes=len(candidates),
        seconds=round(time.monotonic() - start, 1),
    )
    print(month, cohort, phase, len(candidates), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default="original")
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--phase", choices=["spatial", "covariance"], default="spatial")
    a = p.parse_args()
    run(a.cohort, a.month, a.phase)
