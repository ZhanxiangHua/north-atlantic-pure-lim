# North Atlantic storm-generation benchmark

A deterministic FaIR SSP2-4.5 run gives **+1.813 K in 2050**, relative to 1850–1900, for constrained configuration 2451. For each historical or future September event, the two FAST runs use the same monthly ocean and atmosphere, candidate locations and seed intensities. Only the weather driver changes. The LIM provides one daily wind history shared across the event. The TC package samples its native hourly Fourier phases separately for each seed using the same conditional mean and local wind covariance. [Executed experiment](../../notebooks/FaIR_EOF_LIM_to_Tropical_Cyclones.ipynb) · [Inputs and sources](../../notebooks/data/README.md)

| September panels | Weather | Seed attempts | Integrated | TCs | TCs with landfall | Median peak wind of accepted TCs (m/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Historical | Pure LIM | 1,024 | 420 | 27 | 18 | 38.5 |
| Historical | Native Fourier | 1,024 | 414 | 31 | 23 | 35.0 |
| 2050 | Pure LIM | 1,024 | 409 | 32 | 29 | 37.5 |
| 2050 | Native Fourier | 1,024 | 389 | 22 | 16 | 35.4 |

Each group has eight independent event panels with 128 candidate disturbances per panel. The historical years are 1985 through 2020 in five-year steps. Each event starts on September 1 at noon and lasts up to 15 days. In 2050 the LIM produced 0.98 additional TCs per 100 attempted seeds. The 95% interval from resampling the eight whole event pairs was −0.29 to +2.25 per 100. A frequency projection would also require a calibrated number of seeds released each season. [Counts and denominators](tc_counts.csv) · [Paired event intervals](paired_differences.csv)

## Historical storm intensity and movement

We align modeled and observed tracks at their first six-hourly tropical fix with maximum surface wind of at least 18 m/s. The next 48 hours must be complete and inside the evaluation domain. The historical observations use the same eight Septembers and include storms reaching the threshold throughout the month. A storm that lacks this complete window can still contribute to the TC count above. The peak is the largest six-hourly wind within the window. Displacement is the great-circle distance between its endpoints. [NOAA HURDAT2 data description](https://www.nhc.noaa.gov/data/#hurdat) · [Per-storm measurements](storm_windows.csv)

| Source | Complete windows | Median 48 h peak wind (m/s) | Median 48 h displacement (km) | Wind distribution distance to NOAA (m/s) | Displacement distribution distance to NOAA (km) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pure LIM | 19 | 36.5 | 1,051 | 7.05 | 336 |
| Native Fourier | 24 | 33.6 | 1,141 | 2.34 | 434 |
| NOAA HURDAT2 | 37 | 33.4 | 971 | 0 | 0 |

The last two columns give the one-dimensional Wasserstein distance, which integrates the absolute gap in quantiles between two distributions. In these historical windows Fourier is closer on wind intensity and LIM is closer on displacement. Both have higher median displacement than the observed storms. [Historical distribution plot](tc_observations.png) · [Quartiles and distribution distances](historical_shapes.csv)

## Figures and machine-readable output

- [FaIR warming pathway](warming.png) and [September EOF, reconstructed SST and LIM winds](eof_lim.png)
- [2050 storm tracks and landfall locations](tc_tracks.png) and [one paired storm's wind and shear histories](storm_evolution.png)
- [Hourly trajectories in NetCDF](tracks.nc) and [all seed outcomes in Parquet](attempts.parquet), joined by `case`, `method` and `seed_id`
- [Solver timings](solver_timing.csv), [timings for each event](event_timings.csv) and [run settings and software versions](run.json)

On four CPU workers with one BLAS thread each, the paired experiment took **17.4 seconds** wall time. Summed across workers, the track calls processed 94.8 seed attempts per second for LIM and 64.8 for Fourier. These rates include screening, integration and recording. They exclude generation of the shared environment and Fourier provider setup. They depend on the machine and on how many seeds reach integration.

A second run in a clean CPU environment (NumPy 2.5.5, SciPy 1.18.1) reproduced all TC classifications and landfall indicators. The largest difference in an individual integrated disturbance’s peak wind was 0.97 m/s, and some last integration times shifted by up to 3 hours. Use the reference numerical versions for comparisons of individual hourly trajectories. [Reference constraints](../../constraints-benchmark.txt)
