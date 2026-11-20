# North Atlantic Pure LIM

Generate Atlantic monthly ocean–atmosphere backgrounds and daily 250/850 hPa winds at a specified global warming level. The regional emulator has separate EOFs and a linear Gaussian weather model for each calendar month. Coupled FAST can use the resulting environment to generate storm tracks and intensities.

| Tutorial | What you can follow |
| --- | --- |
| [Annual pure LIM](notebooks/Annual_Pure_LIM_Tutorial.ipynb) | Follow an August map through training, then use the saved weights to draw two weather histories at +2 K. Section 5 reconstructs the first day’s four wind maps from 120 scores and checks the result against the event file. [Training diagram](reports/annual_lim_math_flow.png) · [Sampling diagram](reports/annual_lim_sampling_flow.png) · [English derivation](reports/annual_lim_linear_algebra.md) |
| [FaIR → EOF → LIM → tropical cyclones](notebooks/FaIR_EOF_LIM_to_Tropical_Cyclones.ipynb) | Run SSP2-4.5 in FaIR, generate September fields and integrate storms for 15 days. Sections 5–8 replay a shared seed, compare the two LIM versions with daily MPI winds, and trace their 48-hour displacement errors to different parts of the distribution. Section 9 adds six complete held-out SSP2-4.5 years, with track maps, yearly scores and paired intervals. [Driver inputs and outputs](reports/annual_lim_tc_flow.png) · [Track maps and statistics](reports/mpi2050/README.md) |

The September 2050 SSP2-4.5 benchmark replays eight held-out MPI members with daily winds in Coupled FAST. With the FaIR input, the 48-hour displacement distribution was 32 km from the MPI replay for the published v1 LIM and 230 km for native Fourier. Formation rates were 2.44 per 100 proposed seeds for both. The report includes paired uncertainty intervals, the native hourly and daily-averaged controls, and a training-only shear calibration. [MPI daily benchmark and figure](reports/mpi2050/README.md)

The annual refinement study compares 15, 30 and 57 years of paired MPI training data. The annual tutorial now follows the covariance of the ocean and atmosphere into monthly steering winds and evaluates a frozen checkpoint on nine additional years. The tropical cyclone tutorial replays daily MPI winds throughout 2050, maps the simulated tracks and rebuilds formation, displacement and intensity statistics from the archived disturbances. [English results](reports/annual_refinement/README.md) · [中文实验报告](reports/annual_refinement/REPORT.zh.md) · [Covariance diagram](reports/annual_refinement/figures/covariance_to_wind.png)

Six additional SSP2-4.5 years (2041–2059) provide 110,592 common disturbances per driver. Against storms driven by daily MPI winds, pooled 48-hour displacement W1 is 29.65 km for published v1, 32.98 km for Research and 73.62 km for matched Fourier on the same monthly backgrounds. The paired Research-minus-v1 change is +3.33 km (95% interval −50.69 to +71.96 km). Research reduces conditional monthly steering RMSE from 0.846 to 0.784 m/s. [Six-year benchmark and tracks](reports/annual_refinement/midcentury/README.md)

The selected 57-year checkpoint reduces mixed-layer map RMSE on the nine new years from 37.81 to 30.73 m and conditional monthly steering RMSE from 0.915 to 0.804 m/s. With two previous MPI wind maps, its day-one steering RMSE is 2.916 m/s, versus 3.990 m/s for a training-calibrated Fourier spectrum. Both LIM checkpoints pass the pooled 15-day variance test in all twelve months. In the all-month 2050 storm replay, MPI, published LIM and research LIM produce 120, 120 and 98 storms from the same 18,432 disturbances. The report gives warm-range results, paired intervals, intensity and displacement. [Independent results and tracks](reports/annual_refinement/README.md) · [Monthly pass counts](reports/annual_refinement/data/multiday_pass_counts.csv)

A separate, preregistered warm-year check covered 440 member-months at +3.52 to +4.49 K. The research LIM passed the 15-day test in 10 of 11 months, training-calibrated Fourier passed 9, and native Fourier passed 0. A proposed monthly covariance gain also passed 10 and missed its registered improvement criterion. [Five warm years and monthly variance figure](reports/annual_refinement/figures/warm_confirmation_15day.png) · [Result and paired interval](reports/annual_refinement/data/warm_confirmation_result.json)

A follow-up on six additional MPI climates confirmed two deployment options. A 192-state wind model gave up 0.0094 m/s of next-day steering accuracy on 528 member-months. Clipping mixed-layer depth at the MPI model's sea floor reduced subpolar 99th-quantile loss from 2.716 to 2.427 m. The default weights use the full 240-state wind. [Independent scores](reports/annual_refinement/data/compact_ocean_confirmation_paired.csv) · [Conversion and physical depth commands](experiments/annual_refinement/README.md) · [Wind memory and physical ocean depth](reports/annual_refinement/figures/memory_and_ocean_floor.png)

## Installation

Use Python 3.11 or later. The two notebooks have been executed with Python 3.12 on CPU. For a new checkout, start in the directory where you want to clone the repository. The v1.0.0 release supplies its fitted weights. In an extracted full package, the `weights/` folder is already populated and you can proceed directly to dependency installation. [v1.0.0 weights and checksums](https://github.com/ZhanxiangHua/north-atlantic-pure-lim/releases/tag/v1.0.0)

~~~bash
git clone https://github.com/ZhanxiangHua/north-atlantic-pure-lim.git
cd north-atlantic-pure-lim
python -m venv .venv
source .venv/bin/activate
curl -fL -o annual_lim_full.tar.gz https://github.com/ZhanxiangHua/north-atlantic-pure-lim/releases/download/v1.0.0/annual_lim_full.tar.gz
tar -xzf annual_lim_full.tar.gz --strip-components=1 annual_lim/weights
~~~

| Task | Install command |
| --- | --- |
| Train or validate EOF/LIM weights, generate backgrounds and winds from a supplied warming level | `python -m pip install -r requirements.txt` |
| Use FaIR and Coupled FAST from Python, including the native Fourier driver | `python -m pip install -r requirements-tc.txt` |
| Open and rerun either notebook, with figures and benchmarks | `python -m pip install -r requirements-notebook.txt` |

For a CPU-only Linux installation, you can first run `python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu` in the new virtual environment. The notebook requirements include both modeling stacks. The checkout supplies the calibrated FaIR scenario inputs, TC solver, thermodynamic tables, geography, one MPI reference month and the six-hourly ensemble tracks. After dependency and weight installation the demonstrations run offline. Set `ANNUAL_LIM_WEIGHTS` to the location of the published weights when rerunning the archived September replay. The research sections of both notebooks read their matched benchmark tables from `reports/annual_refinement/`. Set `TC_DEMO_WORKERS` to control the number of benchmark processes (default 2). [Input versions and provenance](notebooks/data/README.md)

~~~bash
python -m pip install -r requirements-notebook.txt -c constraints-benchmark.txt
python -m jupyterlab notebooks/
~~~

The optional constraints file selects the numerical versions used for the published benchmark. A separate clean run with NumPy 2.5.5 and SciPy 1.18.1 reproduced the TC formation and landfall decisions for all 4,096 seed attempts. Some individual trajectories and termination times shifted across numerical environments.

The training data are MPI-ESM1-2-LR. The catalogue identifies the original CMIP6 files by member, experiment, variable and release. Data citations and the paper behind the warming-conditioned EOF approach are available from the original sources. [Historical experiment](https://doi.org/10.22033/ESGF/CMIP6.6595) · [SSP5-8.5 experiment](https://doi.org/10.22033/ESGF/CMIP6.6705) · [Geogdzhayev et al.](https://doi.org/10.5194/esd-17-235-2026)

## Sample the refined checkpoint

After installing the core dependencies and extracting the v1.0.0 weights into `weights/`, restore the twelve research weights once. Then a monthly sample needs the model, a calendar date, a warming level and a random seed. The command below generates a 15-day North Atlantic event in September 2050 at a prescribed warming of 2 K above 1850–1900.

~~~bash
python experiments/annual_refinement/checkpoint.py restore --base weights \
  --delta checkpoints/annual_refinement --output work/refined_weights
python run.py sample --weights work/refined_weights --month 9 --year 2050 \
  --g 2 --days 15 --seed 417 --output work/september.nc
~~~

The 5.57 MB minimal source package contains the preprocessing, training and inference entry points with the dependencies of the selected refit. The tutorial package also includes both executed notebooks, the TC solver and the research results. [Minimal model source](https://raw.githubusercontent.com/ZhanxiangHua/north-atlantic-pure-lim/main/downloads/annual_lim_minimal.tar.gz) · [Source, tutorials and benchmarks](https://raw.githubusercontent.com/ZhanxiangHua/north-atlantic-pure-lim/main/downloads/annual_lim_code.tar.gz) · [Package checksums](https://github.com/ZhanxiangHua/north-atlantic-pure-lim/blob/main/downloads/SHA256SUMS)

A Git checkout and the full research package already contain the twelve checkpoint deltas. If you started from either source archive, run the following command inside its extracted directory before restoring. The archive supplies the checksum manifest, and the restore command verifies all twelve files.

~~~bash
curl -fL --output 'checkpoints/annual_refinement/#1.delta.zip' \
  'https://raw.githubusercontent.com/ZhanxiangHua/north-atlantic-pure-lim/main/checkpoints/annual_refinement/[01-12].delta.zip'
~~~

The study archive records the exact checkpoint identifiers, training years, new evaluation years and model-selection rule. Its independent reconstruction starts from the prepared MPI fields. [Training and reproduction commands](experiments/annual_refinement/README.md) · [Linear algebra of the refined fit](experiments/annual_refinement/METHODS.md) · [How to read the tables](reports/annual_refinement/DATA.md)

## 运行和复训


给定年份、月份和全球年度升温，生成北大西洋月背景、热力诊断及每天的 250/850 hPa 风。完整包里有十二套训练好的参数。代码包包含相同的预处理、拟合、验证和采样代码，适用于从资料开始复训。两者均带有重算年度升温所需的全球月均气温小表。

解压后在本目录运行。建议使用 Python 3.12。Linux CPU 已实测，环境版本见 `environment.json`。用现有科学计算环境时，可直接从第二行开始。

~~~bash
python -m pip install -r requirements.txt
python run.py sample --year 2050 --month 8 --g 2 --days 15 --output august.nc
python run.py sample-year --year 2048 --g 2 --output full_year.nc
~~~

`wind` 的单位是 m/s，通道顺序为东西风 250 hPa、南北风 250 hPa、东西风 850 hPa、南北风 850 hPa。事件文件保留 FAST 插值所需的时间缓冲，全年文件只保存真实的 365 或 366 个日中心。`--background-id` 和 `--weather-id` 控制背景与天气。全年逐月独立采样。

全流程整理 8,244 个成员月，其中 6,300 个用于拟合、504 个用于验证、1,440 个用于独立测试。工作目录需要容纳整理后的数据和参数。已归档目录会自动尝试本地 GLADE 和公开网络副本。可以先处理一个真实的成员年。

~~~bash
python run.py forcing --output rebuilt_g.csv
python run.py --work work prepare --member r1i1p1f1 --year 2000
python run.py --work work prepare --workers 2
python run.py --work work train
python run.py --work work validate --role val
python run.py --work work validate --role test
python run.py sample --weights work/weights --year 2050 --month 8 --g 2 --output refit.nc
~~~

`cluster/` 提供 Casper 的 PBS 分片入口。设置 `PROJECT_ALLOCATION` 为自己的计算项目号，设置 `ANNUAL_PYTHON` 为安装了上述依赖的解释器绝对路径，在解包目录提交。训练和验证分别按日历月并行。

~~~bash
data_job=$(qsub -A "$PROJECT_ALLOCATION" -v ANNUAL_PYTHON="$ANNUAL_PYTHON" cluster/prepare.pbs)
fit_job=$(qsub -A "$PROJECT_ALLOCATION" -v ANNUAL_PYTHON="$ANNUAL_PYTHON" -W "depend=afterok:${data_job}" cluster/train.pbs)
qsub -A "$PROJECT_ALLOCATION" -v ANNUAL_PYTHON="$ANNUAL_PYTHON" -W "depend=afterok:${fit_job}" cluster/validate.pbs
~~~

中文技术说明从两格点例子开始，介绍背景、天气和风场重建。三页流程图依次讲训练、采样和气旋实验，每一步都写出输入、输出、公式含义及对应章节。全年教程第 5 节用 8 月权重把第 1 天的 120 个风系数还原为四张风图，并与输出文件逐格核对。第二本教程用 FaIR 情景驱动北大西洋逐日风和 Coupled FAST，并给出同种子对照和逐小时轨迹。第 5–6 节现场用 MPI、LIM 和傅里叶三种风回放同一颗种子，再从 8 个留出成员的轨迹重算 48 小时位移分布和配对区间。[中文技术说明](reports/annual_lim_guide.md) · [全年分月验证](reports/annual_lim_results.md) · [数学流程图 PDF](reports/annual_lim_math_flow.pdf) · [气旋实验结果](reports/fair_to_tc/README.md)
