# Reproducing the North Atlantic refinement experiment

The main package in `src/annual_lim` loads the research weights with the same monthly sampling API as the published fit. This directory contains the data ablations, trainable alternatives, physical scoring code and PBS launchers. The model equations and the meaning of each score are described in [Methods](METHODS.md).

## Restore the fitted model

Download and extract the v1.0.0 weights as described in the repository README. The compact research checkpoint supplies the arrays that differ. It verifies the base identifier, archive checksums and the byte-level hash of each restored model.

```bash
python experiments/annual_refinement/checkpoint.py restore \
  --base weights \
  --delta checkpoints/annual_refinement \
  --output work/refined_weights
python run.py sample --weights work/refined_weights --month 9 --year 2050 --g 2.0 \
  --days 15 --seed 417 --output work/september.nc
```

`g=2.0` means an annual global mean temperature 2 K above 1850–1900. Give the sampler different background and weather identifiers to generate more independent events. Keep the background identifier fixed when studying the weather variability of one sampled month. Existing annual code can use `work/refined_weights` wherever it accepts a weights directory.

## Refit the selected recipe from CMIP6

Set up the project environment and install the research dependencies. Run all commands from the root of the repository. The allocation-free Python entry point can prepare data and train separately.

```bash
python -m pip install -r experiments/annual_refinement/requirements.txt
source experiments/annual_refinement/env.sh
"$TC_PYTHON" experiments/annual_refinement/retrain.py \
  --recipe reports/annual_refinement/recipe.json --baseline-weights weights \
  --stage prepare --workers 2
"$TC_PYTHON" experiments/annual_refinement/retrain.py \
  --recipe reports/annual_refinement/recipe.json --baseline-weights weights \
  --stage train --workers 2
```

Preparation reads the CMIP6 catalogue, selects the required original training and validation members, obtains the required fields, regrids them and writes checksummed member-year files. The nine-year independent test panel is unnecessary for training. The training command prints the path to the twelve fitted weights. The scripts reuse complete inputs on restart. Each training process holds one calendar month's maps and winds in memory. The largest 57-year refit was configured with 72 GB for three concurrent monthly workers.

The original v1 fit metadata identifies the common spatial region and wind acceptance bands. `baseline_reference.py` binds the downloaded weights to those published IDs. `TC_REFINEMENT_WORK` can move the research cache to another filesystem. The existing annual preprocessing convention keeps the original member-year files under `runs/annual_lim/prepared`. Bare Python requires `PYTHONPATH=src` and the packages in the requirements file. The supplied shell helper sets this path and the numerical thread counts.

## Follow a research comparison

| Scientific question | Code used in the experiment |
| --- | --- |
| How much information is lost when an observed ocean map is compressed? | `ocean.py`, `calendar_ocean.py`, `ocean_alternatives.py`, `nested.py` |
| What changes between 15 and 30 paired years, 57 ocean years, and 57 paired years? | `features.py`, `ocean_only.py`, `ocean_rich.py`, `learning_curve.py`, `audit_infill.py` |
| How do warming, background covariance and observed fields enter the monthly wind? | `backbone.py`, `transfer.py`, `between.py`, `active_warming.py`, `attribution.py` |
| Do adjacent months share a warming response, and does training on ocean fronts improve their reconstruction? | `seasonal_covariance_cv.py`, `ocean_fronts.py`, `front_circulation.py`, `wind_visible_gain.py` |
| Can more ocean spread improve probabilities while preserving the wind regression? | `ocean_spread.py`, `marine_regions.py`, `spatial_check.py` |
| How does a change affect 5-day, 15-day and monthly variability? | `weather.py`, `seasonal.py`, `modal_weather.py`, `weather_controls.py`, `tuned_fourier.py`, `leadlag.py`, `lag_calibration.py` |
| How accurate are physical mixed-layer depths in the extreme upper tail? | `deep_mixed_layer.py` |
| What did independent MPI members and FAST show? | `verify_model.py`, `paired_intervals.py`, `annual_scenario.py`, `storms.py`, `score_storms.py`, `external_comparison.py` |

`design.json` defines members indirectly through the public split, training and test years, the temperature boundaries and the original campaign window. `shortlist.py`, `finalists.py` and `decision.py` document the selection rules. `freeze.py` records the twelve model IDs before the independent tests open. Test and scenario scores cannot enter those selection functions. The journal and exported tables are linked from the [research report](../../reports/annual_refinement/README.md).

## Rebuild the figures and tutorials

The published attempt tables and six-hour tracks carry complete year, month, member, draw, origin and seed identifiers. The research figures that use these public tables can be regenerated without downloading climate fields. Install the notebook requirements for the two tutorials and use the v1 weights for their original walkthroughs.

```bash
python -m pip install -r requirements-notebook.txt -c constraints-benchmark.txt
source experiments/annual_refinement/env.sh
tag="$("$TC_PYTHON" -c 'import json; print(json.load(open("reports/annual_refinement/freeze_receipt.json"))["stage"])')"
"$TC_PYTHON" experiments/annual_refinement/evidence_figures.py --tag "$tag" --report reports/annual_refinement
"$TC_PYTHON" experiments/annual_refinement/training_figures.py --report reports/annual_refinement
"$TC_PYTHON" experiments/annual_refinement/execute_tutorial.py --tutorial annual --output work/annual-executed.ipynb
"$TC_PYTHON" experiments/annual_refinement/execute_tutorial.py --tutorial tc --output work/tc-executed.ipynb
```

## Reproduce the additional five-year warm confirmation

The study committed the primary model IDs, twelve constant gains, five scenario years and eleven calendar months in its confirmation plan before reading the new daily fields. Use the published research weights and comparison tables with the research environment. The acquisition command reads forty member-years. September is prepared as part of the source-year files and excluded by the scoring command.

~~~bash
source experiments/annual_refinement/env.sh
"$TC_PYTHON" experiments/annual_refinement/warm_confirmation.py --setup
"$TC_PYTHON" -m annual_lim --config "$TC_REFINEMENT_WORK/configs/warm_confirmation" \
  --work "$TC_REFINEMENT_WORK/warm_confirmation/input" prepare --workers 2
for month in 1 2 3 4 5 6 7 8 10 11 12; do
  "$TC_PYTHON" experiments/annual_refinement/warm_confirmation.py --month "$month" \
    --weights work/refined_weights
done
"$TC_PYTHON" experiments/annual_refinement/warm_confirmation.py --summary reports/annual_refinement/data
~~~

## Reproduce the one-lag simplification on development validation

After reconstructing the frozen weights and the original and 57-year training moment caches, run each calendar month and pool its results. This comparison uses the seven original validation members.

```bash
source experiments/annual_refinement/env.sh
for month in {1..12}; do
  "$TC_PYTHON" experiments/annual_refinement/order_reduction.py --month "$month"
done
"$TC_PYTHON" experiments/annual_refinement/order_reduction.py --summarize reports/annual_refinement/data
```

The partial-memory trial keeps past-wind coordinates ordered by their conditional correlation with the next day. The precomputed sea-floor grid is derived from MPI-M's native CMIP6 `deptho` file. An optional `--map --deptho /path/to/deptho.nc --historical-mlotst /path/to/mlotst.nc` rebuilds the grid and checks its source coordinates.

```bash
for month in {1..12}; do
  "$TC_PYTHON" experiments/annual_refinement/partial_wind_memory.py --month "$month"
  "$TC_PYTHON" experiments/annual_refinement/bathymetric_mld.py --month "$month"
done
"$TC_PYTHON" experiments/annual_refinement/partial_wind_memory.py --summarize reports/annual_refinement/data
"$TC_PYTHON" experiments/annual_refinement/bathymetric_mld.py --summary
```

The joint follow-up fixes 72 previous-day directions and the physical sea-floor bound before reading six additional annual MPI climates. Its outputs include the individual forecasts, each year's observed and modeled 15-day variances, and the paired mixed-layer depth scores.

```bash
"$TC_PYTHON" experiments/annual_refinement/compact_ocean_confirmation.py --setup
"$TC_PYTHON" -m annual_lim --config "$TC_REFINEMENT_WORK/configs/compact_ocean_confirmation" \
  --work "$TC_REFINEMENT_WORK/compact_ocean_confirmation/input" prepare --workers 2
for month in 1 2 3 4 5 6 7 8 10 11 12; do
  "$TC_PYTHON" experiments/annual_refinement/compact_ocean_confirmation.py --month "$month" \
    --weights work/refined_weights
done
"$TC_PYTHON" experiments/annual_refinement/compact_ocean_confirmation.py --summary reports/annual_refinement/data
```

## Sample the 192-state option or enforce the physical sea floor

From the full source package or the repository, use the restored research weights to create a complete 192-state set. This conversion takes only the twelve parameter files and their EOF representations. The standard sampling command then reads the new weights. The sea-floor command works on either wind model's monthly or annual NetCDF output. It reads the accompanying mapped MPI bathymetry, writes the bounded field in metres and records how many grid cells were adjusted.

```bash
source experiments/annual_refinement/env.sh
"$TC_PYTHON" experiments/annual_refinement/compact_deploy.py weights --source work/refined_weights --output work/compact_weights
"$TC_PYTHON" -m annual_lim sample-year --weights work/compact_weights --year 2050 --g 1.799 --seed 2050 --output work/compact2050.nc
"$TC_PYTHON" experiments/annual_refinement/compact_deploy.py ocean --input work/compact2050.nc --output work/compact2050_bounded.nc
"$TC_PYTHON" experiments/annual_refinement/memory_ocean_figure.py --report reports/annual_refinement
```

To repeat the propagation timing after building the two sets of weights, run `compact_timing.py --month MONTH --full work/refined_weights --compact work/compact_weights` for each month from 1 to 12, then `compact_timing.py --summary reports/annual_refinement/data`. The median time ratio comes from 84 paired calls on the same node within each pair.

## Run focused numerical checks

```bash
source experiments/annual_refinement/env.sh
"$TC_PYTHON" -m pytest -q experiments/annual_refinement/test_refinement.py \
  experiments/annual_refinement/test_training_extensions.py
```

The numerical checks cover rolling-window variances for February through 31-day months, periodic parameter smoothing, the stationary Gaussian state transformation, preservation of coarse wind covariance, physical spatial increments and lossless checkpoint restoration. The additional checks verify member-subset steering variance, the physical ocean-front metric, the upper-tail probability transform, and exact scaling of the wind-visible covariance.
