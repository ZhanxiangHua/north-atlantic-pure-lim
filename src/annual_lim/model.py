"""Numerical parameters for one calendar month and the two Gaussian draws."""
from pathlib import Path

import numpy as np

from tc_emulator.background import Background, Circulation
from tc_emulator.common import digest, sha
from tc_emulator.pure_lim_memory import MemoryCalibration, MemoryLIM
from tc_emulator.representation import Representation
from tc_emulator.wind_lim import Dynamics
from .layout import read, rng_for


class MonthlyModel:
    def __init__(self, weights, month):
        folder = Path(weights) / f"{month:02}"
        self.meta = read(folder / "model.json")
        if self.meta["month"] != month or self.meta["format"] != 1:
            raise ValueError("Model does not match requested calendar month or file format")
        for name, val in self.meta["files"].items():
            if sha(folder / name) != val:
                raise ValueError(f"Model checksum mismatch in {folder/name}")
        if digest({k: v for k, v in self.meta.items() if k != "id"}) != self.meta["id"]:
            raise ValueError("Changed monthly model metadata")
        self.rep = Representation.load(folder / "representation.npz")
        with np.load(folder / "parameters.npz", allow_pickle=False) as a:
            L = a["background_root"]
            self.bg = Background(
                float(a["center"]),
                float(a["g_scale"]),
                a["beta"],
                a["beta"][:2],
                L,
                np.zeros_like(L),
                L @ L.T,
            )
            self.mean = Circulation(a["regression"], a["z_scale"])
            self.directions = a["directions"]
            self.bg_amp, self.bg_slope = map(float, a["background_response"])
            self.base = Dynamics(
                a["transition"], a["innovation"], a["stationary"], 2, a["regression"].shape[1]
            )
        self.cal = MemoryCalibration(**self.meta["weather"])
        self.weather = MemoryLIM(self.base, self.cal)

    def factor(self, g):
        # L(g) = L [I + (sqrt(s(g))-1) U U^T]; s acts on wind-relevant directions.
        g_clip = self.cal.temperature(g)
        amp = np.sqrt(self.bg_amp * np.exp(self.bg_slope * (g_clip - self.bg.center)))
        L, U = self.bg.lower0, self.directions
        return L + (amp - 1) * (L @ U) @ U.T

    def background(self, g, rng):
        """Generate z = m(g) + F(g)e, with independent standard normal entries in e."""
        return self.bg.mean([g])[0] + self.factor(g) @ rng.standard_normal(self.bg.beta.shape[1])

    def wind_mean(self, g, z):
        return self.mean.predict(self.bg, np.array([g]), z[None])[0]

    def pc_path(self, g, z, days, year, seed, bg_id, weather_id, device=None):
        """w_t = mu + s(g) kappa (x_t - b mean(x_{t-5}, ..., x_{t-20}))."""
        # Separate streams for the 20-day stationary history and subsequent innovations.
        key = (int(year), self.meta["month"], str(bg_id), str(weather_id))
        noise = rng_for(seed, "wind", *key).standard_normal((days, len(self.base.a)))
        res = self.weather.propagate(g, noise, rng_for(seed, "history", *key), device)
        return self.wind_mean(g, z) + res

    def draw_z(self, g, year, seed, bg_id):
        return self.background(g, rng_for(seed, "background", int(year), self.meta["month"], str(bg_id)))
