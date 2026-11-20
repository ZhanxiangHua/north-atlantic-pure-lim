"""Validate ocean EOF compression in physical units, at matched and higher ranks."""
import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import scipy.linalg as la
from records import WORK, record

from annual_lim.layout import CONFIG, DEFAULT_WORK, case_dir, design, read
from annual_lim.model import MonthlyModel
from tc_emulator.common import atomic
from tc_emulator.representation import EOF, training_stats

FIELDS = ("sst", "mld", "gamma")
RANKS = (12, 18, 32, 48, 72, 96, 128)
GROUP_LIMITS = (-0.0115220616351514, 3.419637285227324)


def panel_cohort(role):
    """Map the evaluation label to its climate experiment and saved member split."""
    return (role, "test") if role in ("fresh", "ssp245_year") else ("original", role)


def data_paths(cohort):
    cfg = CONFIG if cohort == "original" else WORK / "configs" / cohort
    # The same member and calendar year can have different fields under two pathways.
    data = WORK / ("ssp245_year" if cohort == "ssp245_year" else "data")
    return cfg, data


def ocean_data(cohort, month, role):
    cfg, work = data_paths(cohort)
    cases = [c for c in design(cfg)[1] if c[0] == role]
    fields, rows = [], []
    for _, member, year, g in cases:
        folder = case_dir(work, member, year)
        mark = read(folder / "complete.json")
        if (mark["role"], mark["g"]) != (role, g):
            raise ValueError(f"Wrong identity at {folder}")
        with np.load(folder / f"{month:02}.npz", allow_pickle=False) as part:
            if (str(part["member"]), int(part["year"]), int(part["month"])) != (
                member,
                year,
                month,
            ):
                raise ValueError(f"Wrong file inside {folder}")
            fields.append(np.stack([part[f] for f in FIELDS]))
            rows.append((member, year, g, mark["hashes"][f"{month:02}.npz"]))
    return np.stack(fields), pd.DataFrame(rows, columns=["member", "year", "g", "source_sha256"])


def joint_input(ocean):
    values = ocean[:, 1:].copy()
    values[:, 0] = np.log(np.maximum(values[:, 0], 0.01))
    return values


def solve_basis(values, area, vertical, rank, *, depth_power=0.0, equalize=0.0, segments=()):
    """Truncated SVD in the training metric. Disjoint subspaces fit in the same EOF file."""
    mean, weights, valid = training_stats(values, area, np.asarray(vertical))
    if depth_power:
        # Linearizing h=exp(log h) gives dh ≈ h d(log h). Temper the weight to balance deep and shallow cells.
        weight = np.exp(np.clip(mean[0], np.log(1.0), np.log(1000.0))) ** depth_power
        weights[0] *= weight / np.sqrt(np.average(weight[valid[0]] ** 2, weights=area[valid[0]]))
    if equalize:
        filled = np.where(np.isfinite(values), values - mean, 0.0)
        sd = np.sqrt(
            np.sum(filled.astype(float) ** 2, axis=0)
            / np.maximum(np.isfinite(values).sum(axis=0), 1)
        )
        for j, keep in enumerate(valid):
            scale = np.maximum(sd[j], 0.3 * np.median(sd[j][keep]))
            rel = (np.median(sd[j][keep]) / np.maximum(scale, 1e-8)) ** equalize
            weights[j] *= rel
    x = values.reshape(len(values), -1)[:, valid.ravel()].astype(float)
    x = np.where(np.isfinite(x), x - mean[valid], 0.0) * weights[valid]
    if not segments:
        segments = [(np.arange(x.shape[1]), rank)]
    variance, modes = [], []
    for cols, keep in segments:
        part = x[:, cols]
        keep = min(keep, part.shape[0] - 1, part.shape[1])
        cov = part @ part.T
        vals, vectors = la.eigh(
            cov, subset_by_index=[len(cov) - keep, len(cov) - 1], check_finite=False
        )
        vals, vectors = vals[::-1], vectors[:, ::-1]
        if vals[-1] <= 0:
            raise ValueError("Insufficient numerical rank in the trial")
        patterns = np.zeros((x.shape[1], keep))
        patterns[cols] = (part.T @ vectors) / np.sqrt(vals)
        modes.append(patterns)
        variance.append(vals / (len(values) - 1))
    return EOF(
        mean,
        weights,
        valid,
        np.column_stack(modes),
        np.concatenate(variance),
        np.sum(x * x) / (len(x) - 1),
    )


def truncate(eof, indices):
    return replace(eof, basis=eof.basis[:, indices], variance=eof.variance[indices])


def split_basis(values, area, rank, depth_power=0.0):
    _, _, mask = training_stats(values, area, np.array([0.5, 0.5]))
    n_mld = mask[0].sum()
    chunks = [(np.arange(n_mld), rank), (np.arange(n_mld, mask.sum()), rank)]
    return solve_basis(values, area, [0.5, 0.5], rank, segments=chunks, depth_power=depth_power)


def physical_scores(actual, reconstructed, training_mean, tail, area, published_mask, meta, field):
    """One row per member-year and region. Missing candidate cells receive training climatology."""
    wet = published_mask & np.isfinite(training_mean)
    ok = wet & np.isfinite(actual)
    provided = np.isfinite(reconstructed) & ok
    pred = np.where(provided, reconstructed, training_mean)
    diff = np.where(ok, pred - actual, 0.0).astype(float)
    w = ok * area
    upper = w * (actual >= tail)
    axes = (1, 2)
    out = meta[["member", "year", "g"]].copy()
    out["field"] = field
    out["mse"] = (w * diff**2).sum(axis=axes) / w.sum(axis=axes)
    out["bias"] = (w * diff).sum(axis=axes) / w.sum(axis=axes)
    climatology_error = np.where(ok, actual - training_mean, 0.0)
    out["reference_mse"] = (w * climatology_error**2).sum(axis=axes) / w.sum(axis=axes)
    out["tail_sse"] = (upper * diff**2).sum(axis=axes)
    out["tail_weight"] = upper.sum(axis=axes)
    out["coverage"] = (area * provided).sum(axis=axes) / w.sum(axis=axes)
    return out


def run(cohort, month, role, extended):
    began = time.monotonic()
    root = WORK / "ocean" / cohort / ("extended" if extended else "screen") / role
    out = root / f"{month:02}.parquet"
    if out.exists():
        return
    record("ocean_start", month=month, cohort=cohort, role=role, extended=extended)
    train, tmeta = ocean_data(cohort, month, "train")
    truth, meta = ocean_data(cohort, month, role)
    baseline = MonthlyModel(DEFAULT_WORK / "weights", month).rep
    y, x = np.meshgrid(baseline.lat, baseline.lon, indexing="ij")
    area = np.cos(np.deg2rad(y))
    regions = {
        "basin": area * ((y >= 0) & (y <= 60) & (x >= 260) & (x <= 360)),
        "corridor": area * ((y >= 10) & (y <= 30) & (x >= 280) & (x <= 340)),
    }
    # Physical climatology and cellwise 90th percentile use the fitting cases only.
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*(empty slice|All-NaN slice).*")
        clim = np.nanmean(train.astype(float), axis=0)
        tail = np.nanquantile(train, 0.9, axis=0)
    masks = [baseline.blocks["sst"].valid[0], *baseline.blocks["ocean"].valid]
    chunks = []

    def evaluate(block, names, layout, params, ranks, raw):
        scores = block.transform(raw)
        for rank, indices in ranks:
            candidate = truncate(block, indices)
            mapped = candidate.inverse(scores[:, indices])
            if names[0] == "mld":
                mapped[:, 0] = np.exp(mapped[:, 0])
                mapped[:, 1] = np.maximum(0.0, mapped[:, 1])
            for j, key in enumerate(names):
                k = FIELDS.index(key)
                for region, weights in regions.items():
                    row = physical_scores(
                        truth[:, k], mapped[:, j], clim[k], tail[k], weights, masks[k], meta, key
                    )
                    chunks.append(
                        row.assign(
                            month=month,
                            cohort=cohort,
                            role=role,
                            region=region,
                            layout=layout,
                            parameter=params,
                            rank=rank,
                        )
                    )

    max_rank = max(RANKS)
    joint_train, joint_true = joint_input(train), joint_input(truth)
    if not extended:
        for key, names, raw in [
            ("sst", ["sst"], truth[:, :1]),
            ("ocean", ["mld", "gamma"], joint_true),
        ]:
            fixed = baseline.blocks[key]
            evaluate(fixed, names, "published", 0.0, [(fixed.basis.shape[1], slice(None))], raw)
        for name, values, raw, channels in [
            ("sst", train[:, :1], truth[:, :1], ["sst"]),
            ("joint", joint_train, joint_true, ["mld", "gamma"]),
        ]:
            q = solve_basis(values, area, np.full(len(channels), 1 / len(channels)), max_rank)
            evaluate(q, channels, name, 0.0, [(k, slice(k)) for k in RANKS], raw)
    else:
        for depth in (0.0, 0.35, 0.7, 1.0):
            if depth:
                joint = solve_basis(joint_train, area, [0.5, 0.5], max_rank, depth_power=depth)
                evaluate(
                    joint,
                    ["mld", "gamma"],
                    "depth_weighted_joint",
                    depth,
                    [(k, slice(k)) for k in RANKS],
                    joint_true,
                )
            half = max_rank
            q = split_basis(joint_train, area, half, depth)
            for share in (0.35, 0.5, 0.65):
                ranks = []
                for k in RANKS:
                    h = round(share * k)
                    ranks.append((k, np.r_[np.arange(h), half + np.arange(k - h)]))
                evaluate(q, ["mld", "gamma"], f"split_mld_{share:g}", depth, ranks, joint_true)
        for power in (0.35, 0.7):
            q = solve_basis(joint_train, area, [0.5, 0.5], max_rank, equalize=power)
            evaluate(
                q,
                ["mld", "gamma"],
                "local_standardization",
                power,
                [(k, slice(k)) for k in RANKS],
                joint_true,
            )
    frame = pd.concat(chunks, ignore_index=True)
    root.mkdir(exist_ok=True, parents=True)
    with atomic(out) as tmp:
        frame.to_parquet(tmp, index=False, compression="zstd")
    with atomic(root / f"{month:02}_train.csv") as tmp:
        tmeta.to_csv(tmp, index=False)
    record(
        "ocean_done",
        month=month,
        cohort=cohort,
        role=role,
        extended=extended,
        train=len(train),
        heldout=len(truth),
        rows=len(frame),
        seconds=round(time.monotonic() - began, 2),
        file=str(out.relative_to(WORK)),
    )
    print(month, cohort, role, len(frame), round(time.monotonic() - began, 1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", choices=["original", "expanded", "extended"], default="original")
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--extended", action="store_true")
    args = p.parse_args()
    run(args.cohort, args.month, "val", args.extended)
