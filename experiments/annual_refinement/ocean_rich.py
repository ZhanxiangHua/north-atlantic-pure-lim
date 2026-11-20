"""Learn additional regional EOFs from ocean-only years while fitting circulation on paired years."""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from backbone import Bank
from features import save_block
from ocean import FIELDS, joint_input, ocean_data, physical_scores, solve_basis, truncate
from records import WORK, record

from annual_lim.layout import CONFIG, read
from tc_emulator.common import atomic, save_json


def extra_data(month):
    years = read(Path(__file__).with_name("ocean_extra_years.json"))["years"]
    members = read(CONFIG / "members.json")["train"]
    array = []
    metadata = []
    for y in years:
        for member in members:
            p = WORK / "ocean_only" / member / str(y)
            rec = read(p / "complete.json")
            assert (rec["year"], rec["member"]) == (y, member)
            array.append(np.array(np.load(p / "maps.npy", mmap_mode="r")[month - 1]))
            metadata.append((member, y, rec["g"], rec["sha256"]))
    return np.stack(array), pd.DataFrame(metadata, columns=["member", "year", "g", "source_sha256"])


def run(month):
    start = time.monotonic()
    bank = Bank("expanded", month)
    root = bank.root
    if (root / "rich_complete.json").exists():
        return
    record("rich_ocean_start", month=month)
    pair, tm = ocean_data("expanded", month, "train")
    valid, vm = ocean_data("original", month, "val")
    extra, em = extra_data(month)
    assert set(tm.year).isdisjoint(set(em.year))
    y, x = np.meshgrid(bank.geo.lat, bank.geo.lon, indexing="ij")
    a = np.cos(np.deg2rad(y))
    regions = {
        "basin": a * ((y >= 0) & (y <= 60) & (x >= 260) & (x <= 360)),
        "corridor": a * ((y >= 10) & (y <= 30) & (x >= 280) & (x <= 340)),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clim = np.nanmean(pair.astype(float), axis=0)
        tail = np.nanquantile(pair, 0.9, axis=0)
    masks = [bank.geo.blocks["sst"].valid[0], *bank.geo.blocks["ocean"].valid]
    all_scores = {}
    output = []
    for cohort, nested in (
        ("paired30", []),
        ("ocean44", sorted(em.year.unique())[::2]),
        ("ocean57", sorted(em.year.unique())),
    ):
        mask = em.year.isin(nested)
        fitting = np.concatenate((pair, extra[mask])) if nested else pair
        for key, channels, d, inputs in (
            ("rich_sst", ["sst"], 0.0, lambda v: v[:, :1]),
            ("rich_ocean", ["mld", "gamma"], 0.0, joint_input),
            ("rich_ocean_d070", ["mld", "gamma"], 0.7, joint_input),
        ):
            tr, va = inputs(fitting), inputs(valid)
            block = solve_basis(
                tr, a, np.full(len(channels), 1 / len(channels)), 128, depth_power=d
            )
            scores = block.transform(va)
            if cohort == "ocean57":
                save_block(root / (key + ".npz"), block)
                for role, maps in (("train", pair), ("val", valid), ("extra", extra)):
                    all_scores[role + "_" + key] = block.transform(inputs(maps))
            for rank in (18, 48, 72, 96, 128):
                fitted = truncate(block, slice(rank)).inverse(scores[:, :rank])
                if channels[0] == "mld":
                    fitted[:, 0] = np.maximum(0.01, np.exp(fitted[:, 0]))
                    fitted[:, 1] = np.maximum(fitted[:, 1], 0.0)
                for j, field in enumerate(channels):
                    i = FIELDS.index(field)
                    for area, weights in regions.items():
                        f = physical_scores(
                            valid[:, i],
                            fitted[:, j],
                            clim[i],
                            tail[i],
                            weights,
                            masks[i],
                            vm,
                            field,
                        )
                        output.append(
                            f.assign(
                                cohort=cohort, basis=key, rank=rank, region=area, train=len(tr)
                            )
                        )
    for key in ("year", "member", "g"):
        all_scores["extra_" + key] = np.asarray(em[key], dtype=str if key == "member" else None)
    with atomic(root / "scores_rich.npz") as tmp:
        np.savez(tmp, **all_scores)
    with atomic(root / "rich_projection.parquet") as tmp:
        pd.concat(output, ignore_index=True).to_parquet(tmp, index=False, compression="zstd")
    with atomic(root / "rich_source_manifest.csv") as tmp:
        em.to_csv(tmp, index=False)
    save_json(
        root / "rich_complete.json",
        {
            "month": month,
            "paired_member_months": len(pair),
            "ocean_only": len(extra),
            "extra_years": sorted(map(int, em.year.unique())),
        },
    )
    record(
        "rich_ocean_done",
        month=month,
        paired=len(pair),
        extra=len(extra),
        seconds=round(time.monotonic() - start, 1),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--month", type=int, required=True)
    run(p.parse_args().month)
