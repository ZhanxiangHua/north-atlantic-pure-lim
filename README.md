# North Atlantic Pure LIM

Generate monthly North Atlantic ocean–atmosphere backgrounds and spatially coherent daily 250/850 hPa winds from annual global warming. Each calendar month has its own fitted EOF bases, warming-conditioned background covariance and linear Gaussian weather dynamics. Inference runs offline on a CPU. The full release includes all twelve sets of weights. [Download the model and code for v1.0.0](https://github.com/ZhanxiangHua/north-atlantic-pure-lim/releases/tag/v1.0.0)

**Start with the August example.** The English derivation follows a field from its weighted EOF projection to a covariance for the 15-day mean wind. The executed tutorial lets you hold the background fixed, vary the weather, generate a leap year and check the analytic variance with 384 realizations. [English mathematical derivation](reports/annual_lim_linear_algebra.md) · [Printable English PDF](reports/annual_lim_linear_algebra.pdf) · [English Jupyter tutorial](notebooks/Annual_Pure_LIM_Tutorial.ipynb)

If you cloned this GitHub repository, download `annual_lim_full.tar.gz` from the release page. Run the following command from the repository root to install the weights. If you extracted the full release itself, the weights are already in place.

~~~bash
tar -xzf annual_lim_full.tar.gz --strip-components=1 annual_lim/weights
~~~

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

你可以先读[技术说明](reports/annual_lim_guide.md)中的两格点例子，再对照[各月测试结果](reports/annual_lim_results.md)。可打印的数学公式和验证表在[技术说明 PDF](reports/annual_lim_guide.pdf)。本包中的热力学代码和熵查找表来自提供的 TC 软件，保留了[上游许可证](vendor/tropical_cyclone_risk/LICENSE)。输出的环境场可以交给项目原有的 FAST 轨迹积分入口。

## 英文 Jupyter 教程

在完整包内打开教程，可以直接运行已保存的示例。它从两格点矩阵开始，用真实 8 月参数固定海洋背景、更换天气，再生成一个闰年，并用 384 次独立天气抽样复核多日平均风的解析方差。各图表也保存在 notebook 中。仅使用代码包时，可先复训，或把已有权重目录写入环境变量 `ANNUAL_LIM_WEIGHTS`。[English Jupyter tutorial](notebooks/Annual_Pure_LIM_Tutorial.ipynb)

~~~bash
python -m pip install -r requirements-notebook.txt
python -m jupyterlab notebooks/Annual_Pure_LIM_Tutorial.ipynb
~~~
