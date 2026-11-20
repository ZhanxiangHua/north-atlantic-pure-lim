"""Create isolated configurations for the data ablation and sealed evaluation years."""
import json
from pathlib import Path

from records import WORK

from annual_lim.layout import CONFIG, ROOT, design, read

SPEC = read(Path(__file__).with_name("design.json"))


def setup():
    original = read(CONFIG / "settings.json")
    seen = set(original["train_years"] + original["validation_years"] + original["test_years"])
    extra, fresh = SPEC["extra_train_years"], SPEC["fresh_test_years"]
    assert not seen.intersection(extra + fresh) and not set(extra).intersection(fresh)
    ocean_years = read(Path(__file__).with_name("ocean_extra_years.json"))["years"]
    assert not (seen | set(extra + fresh)).intersection(ocean_years)
    extended = sorted(original["train_years"] + extra + ocean_years)
    for tag, train, valid, test in (
        ("additions", extra, [], fresh),
        (
            "expanded",
            sorted(original["train_years"] + extra),
            original["validation_years"],
            original["test_years"],
        ),
        ("fresh", original["train_years"], [], fresh),
        ("wind_infill", ocean_years, [], []),
        ("extended", extended, original["validation_years"], original["test_years"]),
    ):
        cfg = WORK / "configs" / tag
        cfg.mkdir(exist_ok=True, parents=True)
        for name in ("catalog.parquet", "members.json", "forcing.csv"):
            if not (cfg / name).exists():
                (cfg / name).symlink_to(CONFIG / name)
        settings = dict(original, train_years=train, validation_years=valid, test_years=test)
        (cfg / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    # Original files stay in place. New member-years are materialized into a separate work tree.
    for role, member, year, _ in design(CONFIG)[1]:
        folder = WORK / "data/prepared" / member
        folder.mkdir(exist_ok=True, parents=True)
        dest = folder / str(year)
        if not dest.exists() and not dest.is_symlink():
            dest.symlink_to(
                ROOT / "runs/annual_lim/prepared" / member / str(year), target_is_directory=True
            )
    for role, member, year, _ in design(WORK / "configs/wind_infill")[1]:
        dest = WORK / "data/prepared" / member / str(year)
        source = WORK / "infill/prepared" / member / str(year)
        if dest.is_symlink() and dest.readlink() == source:
            continue
        if dest.exists() or dest.is_symlink():
            raise ValueError(f"Unexpected location for a new paired member-year: {dest}")
        dest.symlink_to(source, target_is_directory=True)


if __name__ == "__main__":
    setup()
