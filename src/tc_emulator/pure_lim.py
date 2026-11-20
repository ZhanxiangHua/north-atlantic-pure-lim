"""Stationary linear Gaussian weather with a calibrated temporal response."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .wind_lim import Dynamics


def lag_trace(dynamics, metric, max_lag):
    k = dynamics.observed_dim
    cov = dynamics.sigma[:, :k].copy()
    values = []
    for _ in range(max_lag + 1):
        values.append(np.sum(metric * cov[:k]))
        cov = dynamics.a @ cov
    return np.array(values)


@dataclass
class Calibration:
    method: str
    center: float
    g_min: float
    g_max: float
    extension: float
    intercept: float
    slope: float
    retention: float
    phase_low: float = 0.
    phase_high: float = 0.
    delay: int = 0
    feedback: float = 0.
    normalization: float = 1.

    def temperature(self, g):
        return np.clip(np.asarray(g), self.g_min - self.extension, self.g_max + self.extension)

    def phase(self, g):
        fraction = (self.temperature(g) - self.g_min) / (self.g_max - self.g_min)
        return np.clip(self.phase_low + fraction*(self.phase_high-self.phase_low), 0., np.pi/8)

    def amplitude(self, g):
        return np.exp(.5*(self.intercept + self.slope*(self.temperature(g)-self.center)))

    def window_variance(self, g, lag, windows):
        g = np.atleast_1d(g)
        horizon = max(windows)
        h = np.arange(horizon)
        if len(lag) < horizon + self.delay:
            raise ValueError("Too few lags for the weather window and delay")
        damped = lag * self.retention**np.arange(len(lag))
        if self.delay:
            b, d = self.feedback, self.delay
            response = ((1+b*b)*damped[h] - b*(damped[np.abs(h-d)] + damped[h+d])) * self.normalization**2
            covariance = np.broadcast_to(response, (len(g), horizon))
        else:
            covariance = damped[h] * np.cos(self.phase(g)[:, None]*h)
        result = []
        for days in windows:
            v = days*covariance[:, 0] + 2*(covariance[:, 1:days] @ np.arange(days-1, 0, -1))
            result.append(v/days**2)
        return np.array(result).T * self.amplitude(g)[:, None]**2


class PureLIM:
    def __init__(self, parent, calibration):
        self.parent, self.calibration = parent, calibration

    @cached_property
    def dynamics(self):
        r = self.calibration.retention
        base = self.parent
        if r == 1.:
            return base
        return Dynamics(r*base.a, r*r*base.q+(1-r*r)*base.sigma,
                        base.sigma, base.order, base.observed_dim)

    def propagate(self, g, noise, auxiliary_rng, device=None):
        c = self.calibration
        levels = np.asarray(g)
        days, dim = noise.shape[-2:]
        if days < 1 or not np.isfinite(levels).all() or noise.shape[:-2] != levels.shape:
            raise ValueError("Finite temperature per event and positive duration required")
        if c.delay:
            history = auxiliary_rng.normal(size=(*levels.shape, c.delay, dim))
            path = self.dynamics.simulate(np.concatenate([history, noise], axis=-2), device)
            wind = c.normalization * (path[..., c.delay:, :] - c.feedback*path[..., :-c.delay, :])
        elif c.phase_low or c.phase_high:
            x = self.dynamics.simulate(noise, device)
            y = self.dynamics.simulate(auxiliary_rng.normal(size=noise.shape), device)
            # Two independent identical LIMs viewed in a rotating basis are a
            # homogeneous linear Gaussian Markov process with lag factor cos(h*phase).
            angles = c.phase(levels)[..., None] * np.arange(days)
            wind = np.cos(angles)[..., None]*x + np.sin(angles)[..., None]*y
        else:
            wind = self.dynamics.simulate(noise, device)
        return wind * c.amplitude(levels)[..., None, None]

    def sample(self, g, days, rng, device=None):
        """Generate one or a batch of independent events, with a stationary start."""
        noise = rng.normal(size=(*np.shape(g), days, len(self.dynamics.a)))
        return self.propagate(g, noise, rng, device)
