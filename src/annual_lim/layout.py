"""Paths and the source, member and warming coordinates shared by the stages."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tc_emulator.common import ROOT, digest

CONFIG = ROOT / "configs/annual_lim"
DEFAULT_WORK = ROOT / "runs/annual_lim"


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=4)
def design(config=CONFIG):
    config = Path(config)
    settings, members = read(config / "settings.json"), read(config / "members.json")
    forcing = pd.read_csv(config / "forcing.csv").set_index("year").g
    roles = ("train", "val", "test")
    if any(set(members[a]) & set(members[b]) for i, a in enumerate(roles) for b in roles[i + 1 :]):
        raise ValueError("Member splits overlap")
    out = []
    for role, years in zip(roles, ("train_years", "validation_years", "test_years"), strict=True):
        for year in settings[years]:
            for m in members[role]:
                out.append((role, m, year, float(forcing.loc[year])))
    return settings, out


def case_dir(work, member, year):
    return Path(work) / "prepared" / member / str(year)


def rng_for(seed, *labels):
    return np.random.default_rng(int(digest([int(seed), *labels])[:16], 16))
