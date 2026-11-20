"""Temperature-dependent covariance and a one-parameter circulation interaction."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from .background import Background, fit_mean, polynomial


def year_covariances(z, years):
    return np.stack([np.cov(z[years == year], rowvar=False) for year in np.unique(years)])


def refit_backbone(g, z, w, years, ridge):
    """Reconstruct the selected parent's pooled option from one training subset."""
    center, scale = float(g.mean()), float(g.std())
    design = polynomial((g - center) / scale)
    covs = year_covariances(z, years)
    weights = np.unique(years, return_counts=True)[1] - 1
    pooled = np.average(covs, axis=0, weights=weights)
    pooled = .95 * pooled + .05 * np.diag(pooled.diagonal())
    lower = np.linalg.cholesky(pooled)
    bg = Background(center, scale, np.linalg.lstsq(design, z, rcond=None)[0],
                    np.linalg.lstsq(design[:, :2], z, rcond=None)[0], lower,
                    np.zeros_like(lower), pooled)
    return bg, fit_mean(bg, g, z, w, ridge)


def transfer_matrix(mean):
    return mean.coefficients[2:] / mean.scale[:, None]


def projected_variance(cov, transfer, metric):
    return float(np.sum(cov * (transfer @ metric @ transfer.T)))


def gaussian_nll(cov, sample_cov):
    """Expected nats per dimension, with the reference sample centered within year."""
    lower = np.linalg.cholesky(cov)
    a = solve_triangular(lower, sample_cov, lower=True)
    whitened = solve_triangular(lower, a.T, lower=True)
    return float(.5 * (np.log(2 * np.pi) +
                       (2 * np.log(lower.diagonal()).sum() + np.trace(whitened)) / len(cov)))


@dataclass
class CovarianceLaw:
    lower: np.ndarray
    directions: np.ndarray
    year_covs: np.ndarray
    anchors: np.ndarray
    center: float
    active_scale: float
    log_slope: float
    bandwidth: float
    pooled_fraction: float
    extension: float

    def factor(self, g, kind):
        """Return a square root with orientation z = mean + factor @ epsilon."""
        if kind == "pooled":
            return self.lower
        if kind == "local":
            logits = -.5 * ((g - self.anchors) / self.bandwidth) ** 2
            weights = np.exp(logits - np.max(logits))
            local = np.average(self.year_covs, axis=0, weights=weights)
            pooled = self.lower @ self.lower.T
            return np.linalg.cholesky(self.pooled_fraction * pooled + (1 - self.pooled_fraction) * local)
        if kind != "projected":
            raise ValueError(f"Unknown warming covariance {kind}")
        x = np.clip(g, self.anchors.min() - self.extension, self.anchors.max() + self.extension)
        amplitude = np.sqrt(self.active_scale * np.exp(self.log_slope * (x - self.center)))
        return self.lower + (amplitude - 1) * (self.lower @ self.directions) @ self.directions.T

    def covariance(self, g, kind):
        f = self.factor(g, kind)
        return f @ f.T


def fit_covariance(bg, mean, g, z, years, metric, spec):
    yy, first, counts = np.unique(years, return_index=True, return_counts=True)
    anchors = g[first]
    covs = year_covariances(z, years)
    lower = bg.lower0
    if np.any(bg.lower1):
        raise ValueError("The reference covariance must be the pooled option")
    loading = lower.T @ transfer_matrix(mean)
    energies, vectors = np.linalg.eigh(loading @ metric @ loading.T)
    energies = np.maximum(energies[::-1], 0)
    cumulative = np.cumsum(energies) / energies.sum()
    rank = 1 + np.searchsorted(cumulative, spec["wind_variance_fraction"])
    directions = vectors[:, ::-1][:, :rank]
    projected = solve_triangular(lower.T, directions, lower=False)
    active = np.einsum("ir,yij,jr,r->y", projected, covs, projected, energies[:rank], optimize=True)
    weights = (counts - 1) / (counts - 1).sum()
    center = float(weights @ anchors)
    x = anchors - center
    slope = float((weights * x) @ np.log(active) / (weights @ (x*x)))
    scale = float((weights @ active) / (energies[:rank].sum() * (weights @ np.exp(slope*x))))
    law = CovarianceLaw(lower, directions, covs, anchors, center, scale, slope,
                        spec["local_bandwidth_K"], spec["local_pooled_fraction"], spec["tail_extension_K"])
    return law, dict(rank=int(rank), fraction=float(cumulative[rank - 1]), active_log_variance_slope_per_K=slope,
                     active_variance_at_center_relative_to_pooled=scale, center_g=center,
                     training_active_variance={int(y):float(v) for y, v in zip(yy, active)})


def anomaly_gain(g, law, slope, limits):
    g = np.clip(g, law.anchors.min() - law.extension, law.anchors.max() + law.extension)
    return np.clip(1 + slope * (g - law.center), *limits)


def wind_prediction(bg, mean, g, z, gain):
    forced = mean.coefficients[0] + bg.standard_g(g)[:, None] * mean.coefficients[1]
    return forced + np.asarray(gain)[..., None] * ((z - bg.mean(g)) @ transfer_matrix(mean))


def fit_gain(g, z, w, years, members, metric, ridge, shrinkage):
    """Estimate a slope from predictions for members excluded from each inner fit."""
    predicted, target = np.empty_like(w), np.empty_like(w)
    for group in np.array_split(np.unique(members), 5):
        hold = np.isin(members, group)
        bg, mean = refit_backbone(g[~hold], z[~hold], w[~hold], years[~hold], ridge)
        # Center inside each inner fold and year so differing fitted intercepts do
        # not mix with the within-year circulation response.
        for year in np.unique(years):
            take = hold & (years == year)
            delta = z[take] - z[take].mean(axis=0)
            predicted[take] = delta @ transfer_matrix(mean)
            target[take] = w[take] - w[take].mean(axis=0)
    interaction = (g - g.mean())[:, None] * predicted
    weighted = interaction @ metric
    return float(shrinkage * np.sum(weighted * (target - predicted)) / np.sum(weighted * interaction))
