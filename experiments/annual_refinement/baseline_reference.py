"""Install the compact v1 numerical calibration references needed to rerun the research workflow."""
import argparse
from pathlib import Path

from annual_lim.layout import DEFAULT_WORK, read
from annual_lim.model import MonthlyModel
from tc_emulator.common import save_json


def initialize(weights, work):
    weights = Path(weights).resolve()
    fit = read(Path(__file__).with_name("v1_fit_reference.json"))
    work = Path(work).resolve()
    work.mkdir(exist_ok=True, parents=True)
    target = work / "weights"
    if not target.exists():
        if target.is_symlink():
            raise ValueError("The existing v1 weight link is broken")
        if target == weights:
            raise ValueError("First extract the v1.0.0 weights")
        target.symlink_to(weights, target_is_directory=True)
    for m in range(1, 13):
        model = MonthlyModel(target, m)
        item = fit[f"{m:02}"]
        if model.meta["id"] != item["id"]:
            raise ValueError(f"Month {m} does not match the published baseline")
        dest = work / "fit" / f"{m:02}.json"
        if dest.exists():
            if read(dest)["id"] != model.meta["id"]:
                raise ValueError(f"Existing fit description has a different ID for month {m}")
        else:
            save_json(dest, item)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--work", default=DEFAULT_WORK, type=Path)
    a = p.parse_args()
    initialize(a.weights, a.work)
