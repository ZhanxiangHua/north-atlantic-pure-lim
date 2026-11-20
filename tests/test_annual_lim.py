from types import SimpleNamespace

import cftime
import h5py
import numpy as np
import pytest

from annual_lim.prepare import clock
from annual_lim.statistics import window_moments, variance
from annual_lim.validate import fourier_power, wind_score
from annual_lim.sample import forcing


@pytest.mark.parametrize("year,days", [(1900, 28), (2000, 29), (2001, 28)])
def test_cf_day_and_month_index(year, days, tmp_path):
    path = tmp_path / "sample.h5"
    units = f"days since {year-1}-01-01"
    with h5py.File(path, "w") as h:
        t = h.create_dataset("time", shape=(1200,), dtype="f8")
        t.attrs["units"] = units
        t.attrs["calendar"] = "proleptic_gregorian"
        t[0] = 0.5
        left, edges, nd, cal = clock(h, SimpleNamespace(table="day"), year)
        assert len(edges) - 1 == sum(nd) and nd[1] == days and cal == "proleptic_gregorian"
        assert left == cftime.date2num(cftime.DatetimeProlepticGregorian(year, 1, 1), units, cal)
        np.testing.assert_equal(np.diff(edges), 1)
        t[0] = 15.5
        left, edges, nd, _ = clock(h, SimpleNamespace(table="Amon"), year)
        assert left == 12 and nd[1] == days
        np.testing.assert_array_equal(np.diff(edges), nd)


def test_actual_full_month_and_equal_member_weight():
    paths = []
    for length in (28, 29):
        p = np.zeros((length, 4, 1, 1))
        p[-1, :, :, :] = length
        paths.append(p)
    mu, q = window_moments(paths, np.ones((1, 1)))
    total = variance(mu, q)
    np.testing.assert_allclose(total[-1], 0.0, atol=1.0e-12)
    assert total[0] > 20.0
    # At 20 days the native Fourier harmonics finish a full period.
    assert fourier_power(20) < 1.0e-22
    assert fourier_power(5) > fourier_power(15)


def test_forcing_path_selects_unique_year(tmp_path):
    path = tmp_path / "path.csv"
    path.write_text(
        "scenario,draw_id,year,forced_temperature,unit,baseline\ns1,01,2051,2.1,K,1850-1900\ns1,01,2052,2.2,K,1850-1900\n"
    )
    assert forcing(None, path, 2051, "s1", "01")[0] == 2.1
    with pytest.raises(ValueError, match="exactly one"):
        forcing(None, path, None, "s1", "01")


def test_full_month_bootstrap_has_defined_variance_before_division():
    # Two distinct members have monthly values 1 and 3. A draw containing only
    # one of them has a known zero full-month variance and no variance ratio.
    mu = np.broadcast_to(np.array([1.0, 3.0])[:, None, None], (2, 5, 1))
    second = mu[..., 0] ** 2 + np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    cases = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.25, 0.75]])
    values = {m: np.ones((2, 5)) for m in ("iid", "pure_lim", "fourier")}
    rows = wind_score(mu, second, np.zeros((2, 1)), values, cases, np.tile([0.5, 2.0], (5, 1)))
    selected = {r["window"]: r for r in rows if r["model"] == "pure_lim"}
    assert selected["month"]["observed"] == 1.0 and selected["month"]["ratio"] == 1.0
    assert selected["month"]["bootstrap_valid"] == 2 and selected["1"]["bootstrap_valid"] == 4
    assert np.isfinite(selected["month"]["ci_high"])
