"""Repeat the single wind-halo retry at two step caps using the released Research weights."""
import argparse
import calendar
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from annual_lim.layout import read, rng_for
from annual_lim.sample import event
from midcentury_sources import PLAN, REPORT
from tc_emulator import tc_adapter
from tc_emulator.common import ROOT


def check(weights):
    import sys
    sys.path.insert(0, str(ROOT / "notebooks"))
    from benchmark_mpi2050 import sampled_window

    audit, plan = read(REPORT / "boundary_retry.json"), read(PLAN)
    y, m, member = audit["year"], audit["month"], audit["member"]
    key = f"{y}/{m}/{member}/{audit['draw']}"
    g = pd.read_csv(REPORT / "forcing.csv").set_index("year").at[y, "g"]
    if audit["protocol"] != "warming only":
        raise ValueError("The archived boundary check uses a sampled monthly background")
    wind = event(weights, m, y, g, calendar.monthrange(y, m)[1], plan["simulation_seed"], key, audit["draw"])
    window = sampled_window(wind, audit["origin"], plan["window_days"])
    rng = rng_for(plan["simulation_seed"], "FAST", key, audit["origin"])
    seeds = tc_adapter.panel(int(rng.integers(2**32)), plan["proposals_per_start"])
    seed = seeds[seeds.seed_id == audit["seed_id"]]
    original_model = tc_adapter.model

    def fine_model(*args, **kwargs):
        fast, basin = original_model(*args, **kwargs)
        fast.max_step_seconds = audit["reference_step_s"]
        return fast, basin

    def outcome():
        _, records = tc_adapter.tracks(window, seed, plan["window_days"])
        return records.iloc[0]

    recovered = outcome()
    with patch.object(tc_adapter, "model", fine_model):
        fine = outcome()
    for cap, row in [(audit["recovered_step_s"], recovered), (audit["reference_step_s"], fine)]:
        assert (row.tc, row.landfall, row.termination) == (
            audit[f"tc_{cap}"], audit[f"landfall_{cap}"], audit[f"termination_{cap}"])
        np.testing.assert_allclose(row.lmi, audit[f"lmi_{cap}_ms"], rtol=1e-6)
        print(f"{cap}s: {row.termination}, TC={row.tc}, landfall={row.landfall}, peak={row.lmi:.3f} m/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, required=True)
    check(parser.parse_args().research)
