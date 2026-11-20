"""Rebuild one identified research recipe from the public MPI catalogue through twelve monthly weights."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from records import WORK

from annual_lim.layout import CONFIG, DEFAULT_WORK, design, read
from annual_lim.prepare import prepare_case
from tc_emulator.common import save_json


def data_cases(cfg, work, roles, workers):
    cases = [c for c in design(cfg)[1] if c[0] in roles]
    with ProcessPoolExecutor(workers) as pool:
        for _ in pool.map(partial(prepare_case, cfg, work), cases):
            pass


def prep(spec, workers):
    from ocean_only import main as ocean_shard

    cohort = spec["cohort"]
    data_cases(CONFIG, DEFAULT_WORK, {"train", "val"}, workers)
    if cohort in ("expanded", "extended"):
        data_cases(WORK / "configs/additions", WORK / "data", {"train"}, workers)
    if cohort == "extended":
        data_cases(WORK / "configs/wind_infill", WORK / "infill", {"train"}, workers)
    if spec["name"].startswith("rich_"):
        for shard in range(12):
            ocean_shard(shard)


def one_month(spec, month):
    import between
    import calendar_ocean
    import features
    import fullfit
    import nested
    import ocean_rich
    import regional_wind
    import wind_basis
    import wind_moments
    from threadpoolctl import threadpool_limits

    cohort, name, filt = spec["cohort"], spec["name"], spec["filter"]
    with threadpool_limits(4):
        features.make(cohort, month)
    with threadpool_limits(2):
        wind_moments.cache(cohort, month, "train")
        wind_moments.cache("original", month, "val")
    r = fullfit.CASES[name]
    if name.startswith("calendar_"):
        with threadpool_limits(3):
            calendar_ocean.prepare(cohort, month)
    if name.startswith("nested_"):
        with threadpool_limits(3):
            nested.prepare(cohort, month)
    if name.startswith("rich_"):
        with threadpool_limits(4):
            ocean_rich.run(month)
    if r.wind_basis.startswith("mix_"):
        with threadpool_limits(2):
            wind_basis.run(cohort, month)
    with threadpool_limits(3):
        if spec["stage"] == "budget":
            between.run(cohort, month, name, filt)
        else:
            fullfit.run(cohort, month, "basic", name, filt)
    if spec.get("weather_family") == "modal":
        with threadpool_limits(2):
            regional_wind.make(cohort, "train", month)
            regional_wind.make("original", "val", month)
    return month


def postfit_month(spec, source, month):
    import modal_weather
    import ocean_spread
    from threadpoolctl import threadpool_limits

    with threadpool_limits(3):
        if spec["weather_family"] == "modal":
            modal_weather.run(spec["parent"], source, spec["cohort"], month, spec["weather"])
        if spec["spread"] != "own":
            ocean_spread.run(spec["parent"], source, spec["cohort"], month, spec["spread"])


def train(spec, workers):
    import combine
    import seasonal
    from threadpoolctl import threadpool_limits

    source = WORK / "exports" / spec["cohort"] / (spec["name"] + "__" + spec["filter"]) / "weights"
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(partial(one_month, spec), range(1, 13)))
    if "active_recipe" in spec:
        from active_warming import run as fit_active

        parent = spec["active_base"]
        base_file = WORK / "parents" / (parent + ".json")
        reg = dict(
            tag=parent,
            cohort=spec["cohort"],
            name=spec["name"],
            filter=spec["filter"],
            stage=spec["stage"],
            source=str(source),
            jobs={},
        )
        if not base_file.exists():
            save_json(base_file, reg)
        elif any(read(base_file).get(k) != v for k, v in reg.items() if k != "jobs"):
            raise ValueError("The covariance calibration references a different base recipe")
        with threadpool_limits(3), ProcessPoolExecutor(workers) as pool:
            list(pool.map(partial(fit_active, parent, export=spec["active_recipe"]), range(1, 13)))
        source = WORK / "exports/active" / (parent + "__" + spec["active_recipe"]) / "weights"
    if not spec["combined"]:
        return source
    registry = dict(
        tag=spec["parent"],
        cohort=spec["cohort"],
        name=spec["name"],
        filter=spec["filter"],
        stage=spec["stage"],
        source=str(source),
        jobs={},
    )
    existing = WORK / "parents" / (spec["parent"] + ".json")
    if not existing.exists():
        save_json(existing, registry)
    elif any(read(existing).get(k) != v for k, v in registry.items() if k != "jobs"):
        raise ValueError("The working directory already contains an incompatible parent recipe")
    if spec["weather_family"] == "seasonal":
        with threadpool_limits(2):
            seasonal.run(spec["parent"], source, spec["cohort"], spec["weather"])
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(partial(postfit_month, spec, source), range(1, 13)))
    return combine.merge(
        spec["parent"], spec["weather_family"], spec["weather"], spec["spread"], spec["label"]
    )


if __name__ == "__main__":
    from baseline_reference import initialize
    from configure import setup

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--baseline-weights", type=Path, required=True)
    p.add_argument("--stage", choices=["prepare", "train", "all"], required=True)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel member-years during preparation, parallel calendar months during training",
    )
    a = p.parse_args()
    if a.workers < 1:
        p.error("Use at least one worker")
    initialize(a.baseline_weights, DEFAULT_WORK)
    setup()
    spec = read(a.recipe)
    if a.stage in ("prepare", "all"):
        prep(spec, a.workers)
    if a.stage in ("train", "all"):
        print(train(spec, a.workers), flush=True)
