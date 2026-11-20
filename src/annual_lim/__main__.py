import argparse
from pathlib import Path

from tc_emulator.common import ROOT
from .layout import CONFIG, DEFAULT_WORK


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="annual-lim", description="Monthly MPI backgrounds and pure-LIM daily weather"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG,
        help="Catalogue, member split, experiment settings and annual forcing",
    )
    parser.add_argument(
        "--work", type=Path, default=DEFAULT_WORK, help="Prepared months, fitted weights and scores"
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare", help="Read raw MPI fields for complete member-years")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--member", help="Prepare a single member in the registered design")
    p.add_argument("--year", type=int)
    p = sub.add_parser("forcing", help="Rebuild annual warming from training-member global monthly tas")
    p.add_argument("--output", type=Path, required=True)
    for name in ("train", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--month", type=int, choices=range(1, 13), help="All twelve when omitted")
        if name == "validate":
            p.add_argument("--role", choices=("val", "test"), default="test")
    for name in ("sample", "sample-year"):
        p = sub.add_parser(name)
        sources = p.add_mutually_exclusive_group(required=True)
        sources.add_argument("--g", type=float)
        sources.add_argument("--fairgp", type=Path)
        p.add_argument("--scenario")
        p.add_argument("--draw-id")
        p.add_argument("--weights", type=Path, help="Directory containing 01 through 12")
        p.add_argument("--year", type=int, required=True, help="Calendar and random-stream year")
        p.add_argument("--seed", type=int, default=9152026)
        p.add_argument("--background-id", default="0")
        p.add_argument("--weather-id", default="0")
        p.add_argument("--device", choices=("numpy", "cpu", "cuda"), default="numpy")
        p.add_argument("--output", type=Path, required=True)
        if name == "sample":
            p.add_argument("--month", type=int, choices=range(1, 13), required=True)
            p.add_argument("--days", type=int, help="Default is the actual calendar month length")
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        from .prepare import prepare_case, run
        from .layout import design

        if args.member is not None or args.year is not None:
            chosen = [c for c in design(args.config)[1] if c[1] == args.member and c[2] == args.year]
            if len(chosen) != 1:
                parser.error("Select a registered --member and --year together")
            print(prepare_case(args.config, args.work, chosen[0]), flush=True)
        else:
            run(args.config, args.work, args.shard, args.shards, args.workers)
    elif args.stage == "forcing":
        from .forcing import rebuild

        print("1850–1900 mean (degC)", rebuild(args.config, args.output))
    elif args.stage in ("train", "validate"):
        for month in [args.month] if args.month else range(1, 13):
            if args.stage == "train":
                from .train import fit

                fit(args.config, args.work, month)
            else:
                from .validate import assess

                assess(args.config, args.work, month, args.role)
    else:
        from .sample import event, forcing, write, year_sample

        weights = args.weights or (
            ROOT / "weights" if (ROOT / "weights").is_dir() else args.work / "weights"
        )
        g, metadata = forcing(args.g, args.fairgp, args.year, args.scenario, args.draw_id)
        dev = None if args.device == "numpy" else args.device
        if args.stage == "sample-year":
            ds = year_sample(weights, args.year, g, args.seed, args.background_id, args.weather_id, dev)
        else:
            ds = event(
                weights,
                args.month,
                args.year,
                g,
                args.days,
                args.seed,
                args.background_id,
                args.weather_id,
                dev,
            )
        ds.attrs.update(metadata)
        write(ds, args.output)


if __name__ == "__main__":
    main()
