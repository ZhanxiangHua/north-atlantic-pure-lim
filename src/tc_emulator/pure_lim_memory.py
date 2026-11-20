"""A finite-memory output of a stable linear Gaussian process."""
from dataclasses import dataclass

import numpy as np

from .pure_lim import Calibration, PureLIM


def response(lags, retention, first, last, feedback, n):
    """Autocovariance after the causal filter, for one or several scalar metrics."""
    damped = lags * retention**np.arange(lags.shape[-1])
    if last == 0:
        return damped[..., :n]
    taps = np.zeros(last+1)
    taps[0] = 1.
    taps[first:] = -feedback/(last-first+1)
    weights = np.correlate(taps, taps, mode="full")
    indices = np.abs(np.arange(n)[:, None] + np.arange(-last, last+1))
    return damped[..., indices] @ weights


@dataclass
class MemoryCalibration(Calibration):
    last_lag: int = 0

    def window_variance(self, g, lag, windows):
        cov = response(lag, self.retention, self.delay, self.last_lag, self.feedback, max(windows))
        values = [(d*cov[0] + 2*np.dot(np.arange(d-1,0,-1), cov[1:d]))/d**2 for d in windows]
        return self.amplitude(np.atleast_1d(g))[:, None]**2 * self.normalization**2 * np.asarray(values)


class MemoryLIM(PureLIM):
    def propagate(self, g, noise, auxiliary_rng, device=None):
        c = self.calibration
        if c.last_lag == 0:
            return super().propagate(g, noise, auxiliary_rng, device)
        g = np.asarray(g)
        n, dim = noise.shape[-2:]
        if n < 1 or not np.isfinite(g).all() or g.shape != noise.shape[:-2]:
            raise ValueError("Finite temperature per event and positive duration required")
        history = auxiliary_rng.normal(size=(*g.shape, c.last_lag, dim))
        path = self.dynamics.simulate(np.concatenate([history, noise], axis=-2), device)
        total = np.concatenate([np.zeros_like(path[..., :1, :]), path.cumsum(axis=-2)], axis=-2)
        width = c.last_lag-c.delay+1
        past = (total[..., width:width+n, :] - total[..., :n, :]) / width
        residual = path[..., c.last_lag:, :] - c.feedback*past
        return residual * (c.amplitude(g)*c.normalization)[..., None, None]
