"""Regression checks for a partial-month import and the daily Fourier comparison."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import xarray as xr

from annual_lim.prepare import regional
from tc_emulator.common import MODEL
from tc_emulator.fourier_weather import NativeFourierWind, WindStatistics, daily_factors, hourly_fourier, phase_seed
from tc_emulator.tc_adapter import DAY, replay_event


def test_september_import_matches_full_year(tmp_path):
    row = SimpleNamespace(experiment="ssp245", member="r9i1p1f1", variable="ua", table="day")
    with h5py.File(tmp_path / "calendar.h5", "w") as h:
        h.attrs.update(source_id=MODEL, grid_label="gn", experiment_id=row.experiment,
                       variant_label=row.member, parent_variant_label=row.member)
        days = np.arange(365)
        t = h.create_dataset("time", data=days+0.5)
        t.attrs.update(calendar="proleptic_gregorian", units="days since 2050-01-01", bounds="time_bnds")
        h["time_bnds"] = np.column_stack([days, days+1])
        h["lat"], h["lon"] = [0., 30.], [270., 300.]
        h.create_dataset("plev", data=[85000., 25000.]).attrs["units"] = "Pa"
        raw = h.create_dataset("ua", data=np.arange(365*8).reshape(365, 2, 2, 2))
        raw.attrs.update(units="m s-1", standard_name="eastward_wind", cell_methods="time: mean")
        settings = {"domain": [260, 360, 0, 60], "halo_degrees": 4}
        full = regional(h, row, 2050, settings, None)
        sept = regional(h, row, 2050, settings, None, month=9)
        np.testing.assert_array_equal(sept["data"], full["data"][243:273])
        np.testing.assert_array_equal(sept["plev"], [25000., 85000.])
        assert sept["first_month"] == 9 and sept["ndays"].tolist() == [30]


def test_wind_and_replay_share_calendar_centres():
    # Grid and off-grid native evaluations use identical time factors for constant statistics.
    lon, lat = np.linspace(260., 360., 10), np.linspace(0., 60., 8)
    stats = WindStatistics(np.zeros((4, len(lat), len(lon))),
                           np.tile(np.eye(4), (len(lat), len(lon), 1, 1)))
    key, seed, draw, storm = ("future",), 21, 3, 8
    provider = NativeFourierWind(stats, lon, lat, seed, key, draw, True, origin_day=14, daily_mean=True)
    provider.begin_seed(storm)
    _, hourly = hourly_fourier(phase_seed(seed, key, draw, storm))
    factors = daily_factors(hourly)
    np.testing.assert_allclose(provider(300., 20., 0.), factors[:, 13])
    np.testing.assert_allclose(provider(300., 20., DAY/2), factors[:, 13:15].mean(axis=1))
    ds = xr.Dataset({"value": ("day", np.arange(1, 31))},
                    coords={"day": np.arange(1, 31), "time": ("day", np.arange(30)+0.5)})
    out = replay_event(ds, 15, "id", origin_day=14)
    assert out.value.sel(t_seconds=0).item() == 14 and out.value.isel(t_seconds=-1).item() == 30
    with pytest.raises(ValueError, match="cover"):
        replay_event(ds, 15, "id", origin_day=15)
