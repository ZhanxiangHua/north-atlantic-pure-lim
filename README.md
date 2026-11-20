# North Atlantic Pure LIM

Generate Atlantic monthly ocean–atmosphere backgrounds and daily 250/850 hPa winds at a specified global warming level. The regional emulator has separate EOFs and a linear Gaussian weather model for each calendar month. Coupled FAST can use the resulting environment to generate storm tracks and intensities.

| Tutorial | What you can follow |
| --- | --- |
| [Annual pure LIM](notebooks/Annual_Pure_LIM_Tutorial.ipynb) | Follow the [mathematical flowchart](reports/annual_lim_math_flow.pdf) through an August example, from weighted EOFs and monthly background covariance to two daily weather draws. Check 15-day wind variance and sample a leap year. [English derivation](reports/annual_lim_linear_algebra.md) |
| [FaIR → EOF → LIM → tropical cyclones](notebooks/FaIR_EOF_LIM_to_Tropical_Cyclones.ipynb) | Run SSP2-4.5 in FaIR, generate September fields, then integrate storms for 15 days. Compare native Fourier weather on paired seed panels and inspect historical wind intensity and displacement against NOAA best tracks. [Benchmark results, figures and tracks](reports/fair_to_tc/README.md) |

The September 2050 SSP2-4.5 benchmark replays eight held-out MPI members with daily winds in Coupled FAST. With the FaIR input, the 48-hour displacement distribution was 32 km from the MPI replay for LIM and 230 km for native Fourier. Formation rates were 2.44 per 100 proposed seeds for both. The report includes paired uncertainty intervals, the native hourly and daily-averaged controls, and a training-only shear calibration. [MPI daily benchmark and figure](reports/mpi2050/README.md)

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

For a CPU-only Linux installation, you can first run `python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu` in the new virtual environment. The notebook requirements include both modeling stacks. The checkout supplies the calibrated FaIR scenario inputs, TC solver, thermodynamic tables, geography and historical track subset. After dependency and weight installation the demonstrations run offline. Set `ANNUAL_LIM_WEIGHTS` to reuse a different fitted-weight directory in the notebooks. Set `TC_DEMO_WORKERS` to control the number of benchmark processes (default 2). [Input versions and provenance](notebooks/data/README.md)

~~~bash
python -m pip install -r requirements-notebook.txt -c constraints-benchmark.txt
python -m jupyterlab notebooks/
~~~

The optional constraints file selects the numerical versions used for the published benchmark. A separate clean run with NumPy 2.5.5 and SciPy 1.18.1 reproduced the TC formation and landfall decisions for all 4,096 seed attempts. Some individual trajectories and termination times shifted across numerical environments.

The training data are MPI-ESM1-2-LR. The catalogue identifies the original CMIP6 files by member, experiment, variable and release. Data citations and the paper behind the warming-conditioned EOF approach are available from the original sources. [Historical experiment](https://doi.org/10.22033/ESGF/CMIP6.6595) · [SSP5-8.5 experiment](https://doi.org/10.22033/ESGF/CMIP6.6705) · [Geogdzhayev et al.](https://doi.org/10.5194/esd-17-235-2026)

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

中文技术说明从两格点例子开始，介绍背景、天气和风场重建。全年教程里的流程图与第 1–9 节逐一对应。第二本教程用 FaIR 情景驱动北大西洋逐日风和 Coupled FAST，并给出同种子对照、逐小时轨迹和 NOAA 历史观测比较。[中文技术说明](reports/annual_lim_guide.md) · [全年分月验证](reports/annual_lim_results.md) · [数学流程图 PDF](reports/annual_lim_math_flow.pdf) · [气旋实验结果](reports/fair_to_tc/README.md)
