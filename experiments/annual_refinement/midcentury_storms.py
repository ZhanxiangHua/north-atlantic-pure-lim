"""Run the frozen storm experiment on a shard of the six new MPI calendar years."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from annual_lim.layout import design, read
from midcentury_prepare import CFG, STUDY
from midcentury_sources import PLAN
from tc_emulator.common import ROOT


def run(index, lanes, workers, base, research):
    # The unchanged storm runner resolves its inputs and logs at import time.
    os.environ["TC_REFINEMENT_WORK"] = str(STUDY)
    from storms import ORIGINS, SEED, SEEDS, simulate

    p = read(PLAN)
    if (list(ORIGINS), SEED, SEEDS) != (p["start_days"], p["simulation_seed"], p["proposals_per_start"]):
        raise ValueError("Storm simulation differs from the registered design")
    drivers = [(p["reference_driver"], "MPI", base, "base_id"), (p["baseline"], "LIM", base, "base_id"),
               (p["candidate"], "LIM", research, "model_id"), ("selected_fourier", "Fourier", research, "model_id"),
               ("native_fourier", "Fourier", base, "base_id")]
    if lanes < 1 or not 0 <= index < lanes * len(drivers):
        raise ValueError("Invalid driver/lane index")
    lane, driver = divmod(index, len(drivers))
    tag, method, weights, field = drivers[driver]
    manifest = read(ROOT / "checkpoints/annual_refinement/manifest.json")["months"]
    for month, ids in manifest.items():
        if read(weights / month / "model.json")["id"] != ids[field]:
            raise ValueError(f"Wrong checkpoint for {tag} in month {month}")
    cases = [(tag, weights, method, m, y, g, month, "ssp245_year") for role, m, y, g in design(CFG)[1]
             if role == "test" for month in p["months"]]
    with ProcessPoolExecutor(workers) as pool:
        for _ in pool.map(simulate, cases[lane::lanes]):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--research", type=Path, required=True)
    args = parser.parse_args()
    run(args.index, args.lanes, args.workers, args.base, args.research)
