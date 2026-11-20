"""Rebuild the small FaIR inputs shipped with the North Atlantic tutorial."""
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pooch
from fair import FAIR
from fair.io import read_properties

DATA = Path(__file__).resolve().parent / "data"
CALIBRATION = "https://raw.githubusercontent.com/OMS-NetZero/FAIR/v2.2.4/examples/data/calibrated_constrained_ensemble/"
PARAMETERS = "calibrated_constrained_parameters_calibration1.4.1.csv"
SPECIES = "species_configs_properties_calibration1.4.1.csv"
CONFIG = 2451


def download(url, checksum):
    path = pooch.retrieve(url, known_hash="sha256:" + checksum)
    return Path(path)


def prepare():
    DATA.mkdir(exist_ok=True)
    parameters = download(CALIBRATION + PARAMETERS, "7b6c5d9fa0b0b0d3eb47189bf5d63cbf77e752ddac682947abee5ff529206780")
    species_file = download(CALIBRATION + SPECIES, "42d04aa1a8f385cc53eae22beab26f857c535c5aa7dfdb98176d712bfc0c95a0")
    header, *rows = parameters.read_bytes().splitlines(keepends=True)
    row, = [r for r in rows if r.split(b",", 1)[0] == str(CONFIG).encode()]
    (DATA / "fair_2451.csv").write_bytes(header + row)
    (DATA / "fair_species.csv").write_bytes(species_file.read_bytes())

    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(1750, 2101, 1)
    f.define_scenarios(["ssp245"])
    f.define_configs([CONFIG])
    f.define_species(*read_properties(filename=species_file))
    f.allocate()
    # FaIR checks the three RCMIP 5.1.0 MD5s, selects global SSP245 inputs,
    # converts units and interpolates each species on its expected clock.
    f.fill_from_rcmip()
    inputs = {name: getattr(f, name).isel(scenario=0, config=0).values
              for name in ("emissions", "concentration", "forcing")}
    np.savez_compressed(DATA / "rcmip_ssp245.npz", **inputs, timebounds=f.timebounds,
                        timepoints=f.timepoints, species=np.asarray(f.species, dtype=str))
    filenames = ["fair_2451.csv", "fair_species.csv", "rcmip_ssp245.npz"]
    manifest = {
        "sha256": {s: sha256((DATA / s).read_bytes()).hexdigest() for s in filenames},
        "fair": {"version": "2.2.4", "calibration": "1.4.1", "config": CONFIG,
                 "parameters": CALIBRATION + PARAMETERS, "species": CALIBRATION + SPECIES,
                 "reference": "https://docs.fairmodel.net/en/v2.2.4/examples/calibrated_constrained_ensemble.html"},
        "rcmip": {"version": "5.1.0", "scenario": "ssp245", "reference": "https://doi.org/10.5281/zenodo.4589756",
                  "processing": "FaIR.fill_from_rcmip on annual steps from 1750 to 2101. Unused species inputs retain NaNs."},
    }
    (DATA / "sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    prepare()
