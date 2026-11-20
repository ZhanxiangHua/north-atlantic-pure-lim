"""September wind statistics and the distributed TC model's hourly Fourier sampler."""
from contextlib import contextmanager
from datetime import datetime
from threading import Lock
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import interp1d

from .common import fingerprint_rng
from .physical import upstream
from .tc_adapter import DAY, geography


_random_lock = Lock()


@contextmanager
def isolated_phase_seed(seed):
    # The distributed routine draws phases from NumPy's legacy process-global RNG.
    with _random_lock:
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            yield
        finally:
            np.random.set_state(state)


def phase_seed(seed, key, draw, storm=None):
    rng = fingerprint_rng(seed, "TC-package Fourier phases", *key, draw, storm)
    return int(rng.integers(0, 2**32, dtype=np.uint64))


def hourly_fourier(seed, days=30):
    upstream()
    import namelist
    from track.bam_track import BetaAdvectionTrack

    time = np.arange(days*24+1, dtype=float)*3600
    clock = SimpleNamespace(t_s=time, T_Fs=namelist.T_days*DAY, nWLvl=4)
    with isolated_phase_seed(seed):
        factors = BetaAdvectionTrack.gen_synthetic_f(clock)
    return time, factors


def daily_factors(hourly):
    """Integrate piecewise-linear hourly factors over each complete UTC day."""
    return ((hourly[:, :-1] + hourly[:, 1:]) / 2).reshape(hourly.shape[0], -1, 24).mean(axis=2)


def grid_covariance(rep, dim, sigma):
    basis = rep.blocks["wind"]
    # Convert physical EOF loadings to [lat, lon, component, mode].
    maps = (basis.basis[:, :dim]/basis.weights[basis.valid, None]).T
    if not basis.valid.all():
        raise ValueError("Four-component atmospheric winds must cover the comparison grid")
    loadings = maps.reshape(dim, *basis.mean.shape).transpose(2, 3, 1, 0)
    cov = (loadings @ sigma) @ loadings.swapaxes(-2, -1)
    return (cov + cov.swapaxes(-2, -1))/2


class WindStatistics:
    def __init__(self, mean, covariance):
        self.mean = np.asarray(mean, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        self.valid = np.linalg.eigvalsh(self.covariance)[..., 0] > 0
        self.lower = np.zeros_like(self.covariance)
        self.lower[self.valid] = np.linalg.cholesky(self.covariance[self.valid])

    @classmethod
    def observed(cls, wind):
        # env_wind.calc_wnd_stat uses .var() on the diagonal and xr.cov off-diagonal.
        n = len(wind)
        mean = np.mean(wind, axis=0, dtype=np.float64)
        x = (wind-mean).transpose(2, 3, 0, 1)
        cov = (x.swapaxes(-2, -1) @ x)/(n-1)
        j = np.arange(4)
        cov[..., j, j] *= (n-1)/n
        return cls(mean, cov)

    def daily(self, seed):
        _, hourly = hourly_fourier(seed)
        # Exact daily means of the hourly linearly interpolated upstream forcing.
        daily = daily_factors(hourly)
        fluctuation = np.einsum("yxij,jt->tiyx", self.lower, daily, optimize=True)
        # The native point sampler returns zero winds when Cholesky fails.
        return np.where(self.valid, self.mean + fluctuation, 0.).astype("float32")


class NativeFourierWind:
    """Use native spatial interpolation, pointwise Cholesky and hourly time interpolation."""
    def __init__(self, stats, lon, lat, seed, key, draw, independent_storms, *, origin_day=5, daily_mean=False):
        upstream()
        from track.bam_track import BetaAdvectionTrack

        basin = geography()[0]
        native = BetaAdvectionTrack(None, basin, np.datetime64("2000-09-05T12"),
                                     3600, 30*DAY, wind_provider=True)
        native.wind_provider = None
        native.datetime_start = datetime(2000, 9, 1)
        native.wnd_lon, native.wnd_lat = lon, lat
        native.wnd_Mean_Fxs = [native._interp_basin_field(field) for field in stats.mean]
        native.wnd_Cov_Fxs = [[native._interp_basin_field(stats.covariance[..., i, j])
                               for j in range(i+1)] for i in range(4)]
        self.native = native
        self.seed, self.key, self.draw = seed, key, draw
        self.independent_storms = independent_storms
        if not 1 <= origin_day <= 30:
            raise ValueError("September origin day is out of range")
        self.daily_mean = daily_mean
        self.start = (origin_day - 0.5) * DAY
        bounds = [0.5, 29.5] if daily_mean else [0, 30]
        self.t = np.asarray(bounds) * DAY - self.start
        self.origin_s = 0.
        if not independent_storms:
            self.set_phases(None)

    def set_phases(self, storm):
        time, factors = hourly_fourier(phase_seed(self.seed, self.key, self.draw, storm))
        if self.daily_mean:
            factors = daily_factors(factors)
            time = (np.arange(factors.shape[1]) + 0.5) * DAY
        self.native.Fs_i = interp1d(time, factors, axis=1)

    def begin_seed(self, seed_id):
        if self.independent_storms:
            self.set_phases(int(seed_id))

    def __call__(self, lon, lat, seconds):
        return self.native._env_winds(lon, lat, self.start+self.origin_s+seconds)
