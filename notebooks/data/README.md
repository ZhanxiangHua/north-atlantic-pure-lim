# Inputs for the FaIR-to-TC notebook

The notebook runs the FaIR climate model and the TC solver from these inputs. The checksums used at startup are in [sources.json](sources.json).

| File | Origin and preparation |
| --- | --- |
| `rcmip_ssp245.npz` | Global SSP2-4.5 emissions, concentrations and prescribed radiative forcing. FaIR 2.2.4 reads RCMIP v5.1.0, converts units and interpolates to its annual grids for 1750–2101. Temperature is computed during notebook execution. [Nicholls and Lewis, RCMIP archive](https://doi.org/10.5281/zenodo.4589756) · [Nicholls et al., 2020](https://doi.org/10.5194/gmd-13-5175-2020) |
| `fair_2451.csv`, `fair_species.csv` | One parameter row (ID 2451) and the full species table from FaIR calibration 1.4.1, distributed with FaIR 2.2.4. Configuration 2451 is an illustrative ensemble member. [FaIR calibration example](https://docs.fairmodel.net/en/v2.2.4/examples/calibrated_constrained_ensemble.html) · [FaIR license](FaIR-LICENSE) |
| `hurdat2_synoptic.csv.gz` | NOAA Atlantic HURDAT2 revision of April 4, 2025. The subset keeps the 00, 06, 12 and 18 UTC fixes for storm IDs from 1980 through 2021. The two extra years retain tracks that cross a year boundary. The notebook applies the month, status, wind and geographic filters. [NOAA data file](https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt) · [File format](https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atl-1851-2021.pdf) |

To rebuild the small input files, run `python notebooks/prepare_tc_inputs.py` from the repository root with the notebook dependencies installed. The script uses the official, checksum-verified source files.

The track solver, drag, bathymetry and coastal land grids are from Jonathan Lin's TC model, commit `a540a1e`. The two solver files contain the external-wind and integration fixes used by this project. The North Atlantic genesis mask was generated with the upstream basin mask code and Todd Karin's `global-land-mask` 1.0.0. [FAST source](https://github.com/linjonathan/tropical_cyclone_risk/tree/a540a1ed86b121f7244557edb691b40561b81939) · [FAST license](../../vendor/tropical_cyclone_risk/LICENSE) · [Land-mask source](https://github.com/toddkarin/global-land-mask)
