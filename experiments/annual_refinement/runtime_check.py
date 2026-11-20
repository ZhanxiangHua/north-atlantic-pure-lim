"""Exercise a restored annual model with NumPy and PyTorch on the same seeded weather."""
import argparse
import calendar
from pathlib import Path

import numpy as np
import torch

from annual_lim.model import MonthlyModel
from annual_lim.sample import event
from tc_emulator.common import save_json


def check(weights, delta, output):
    torch.set_num_threads(1)
    rows = []
    for month in range(1, 13):
        model = MonthlyModel(weights, month)
        maximum = 0.0
        days = calendar.monthrange(2050, month)[1]
        for warming in (-0.184, 1.799, 4.338):
            z = model.draw_z(warming, 2050, 317, "deployment")
            args = (warming, z, days, 2050, 317, "deployment", 0)
            cpu = model.pc_path(*args)
            alternative = model.pc_path(*args, device="cpu")
            if cpu.shape != (days, model.base.observed_dim):
                raise ValueError("Monthly wind path has the wrong shape")
            np.testing.assert_allclose(cpu, alternative, atol=5e-9, rtol=2e-10)
            maximum = max(maximum, float(np.max(abs(cpu - alternative))))
        row = dict(
            month=month,
            model_id=model.meta["id"],
            state=model.base.observed_dim,
            days=days,
            numpy_torch_pc_max_abs=maximum,
        )
        if month in (2, 5, 8, 11):
            args = (weights, month, 2050, 1.799, 15, 317, "deployment", 1)
            first = event(*args)
            second = event(*args, device="cpu")
            fields = {}
            for key in ("sst", "mld", "gamma", "psl", "ta", "hus", "wind"):
                a, b = first[key].values, second[key].values
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=3e-5, equal_nan=True)
                fields[key] = float(np.nanmax(np.abs(a - b)))
            row["field_max_absolute_differences"] = fields
        rows.append(row)
    save_json(
        output,
        dict(
            months=rows,
            delta_bytes=sum(f.stat().st_size for f in delta.glob("*") if f.is_file()),
            numpy_torch_warmings_K=[-0.184, 1.799, 4.338],
            physical_event_days=15,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--delta", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    check(a.weights, a.delta, a.output)
