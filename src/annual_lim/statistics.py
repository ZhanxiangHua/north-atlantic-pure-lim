"""Grid-weighted moments for variable-length months; a member-month has one vote."""
import numpy as np
from scipy.fft import rfft, irfft
from tc_emulator.diagnostics import region_area, steering, steering_metric, weak_runs

LABELS = ("1", "5", "10", "15", "month")
SHEAR_EDGES = np.arange(0.0, 200.05, 0.05)


def window_moments(paths, area):
    """mu_i = E_start[S], q_i = E_start[||S||²]. The spatial weights are in S."""
    weight = np.sqrt(area / 2)
    mu, second = [], []
    for wind in paths:
        s = (steering(wind) * weight).reshape(len(wind), -1)
        sums = np.vstack((np.zeros(s.shape[1]), s.cumsum(axis=0)))
        means, raw = [], []
        for d in (1, 5, 10, 15, len(wind)):
            avg = (sums[d:] - sums[:-d]) / d
            means.append(avg.mean(axis=0))
            raw.append(np.einsum("ij,ij->", avg, avg) / len(avg))
        mu.append(means)
        second.append(raw)
    return np.array(mu), np.array(second)


def variance(mu, second):
    """At each homologous grid point subtract its own mean before integrating."""
    return second.mean(axis=0) - np.square(mu.mean(axis=0)).sum(axis=-1)


def member_bands(mu, second, people, seed, n_test):
    rng = np.random.default_rng(seed)
    names = np.unique(people)
    mu = np.stack([mu[people == p].mean(axis=0) for p in names])
    second = np.stack([second[people == p].mean(axis=0) for p in names])
    gram = np.einsum("iwf,jwf->wij", mu, mu, optimize=True)
    size = min(n_test, len(names) // 2)
    if size < 2:
        raise ValueError("At least four fitting members required")

    def power(ix):
        return second[ix].mean(axis=0) - gram[:, ix][:, :, ix].mean(axis=(1, 2))

    ratios = []
    for _ in range(400):
        perm = rng.permutation(len(names))
        ratio = power(perm[:size]) / power(perm[size : 2 * size])
        ratios.extend((ratio, 1 / ratio))
    return np.quantile(ratios, [0.025, 0.975], axis=0).T


def component_cov(paths, area):
    """Covariance at lags 0–15; products never cross a month boundary."""
    mean = sum(p.mean(axis=0, dtype=float) for p in paths) / len(paths)
    lag = np.zeros((4, 16))
    for length in sorted({len(p) for p in paths}):
        subset = [p for p in paths if len(p) == length]
        power = np.zeros((length + 1, 4))
        for start in range(0, len(subset), 5):
            f = rfft(np.stack(subset[start : start + 5]) - mean, n=2 * length, axis=1)
            power += np.einsum("nfcyx,yx->fc", f.real**2 + f.imag**2, area, optimize=True)
        prod = irfft(power, n=2 * length, axis=0)[:16]
        lag += (prod / (length - np.arange(16))[:, None]).T
    return mean, lag / len(paths)


def shear_hist(paths, area, weak=None):
    hist = np.zeros(len(SHEAR_EDGES) - 1)
    count = 0.0
    for wind in paths:
        shear = np.hypot(wind[:, 0] - wind[:, 2], wind[:, 1] - wind[:, 3])
        if not np.isfinite(shear).all() or shear.max() >= SHEAR_EDGES[-1]:
            raise ValueError("Wind shear outside histogram support")
        hist += np.histogram(shear, SHEAR_EDGES, weights=np.broadcast_to(area / len(wind), shear.shape))[
            0
        ]
        if weak is not None:
            count += weak_runs(shear < weak, area)[5:].sum()
    return hist, count / len(paths)


def quantiles(hist, p):
    return np.interp(p, np.cumsum(hist) / hist.sum(), SHEAR_EDGES[1:])


def metrics(rep):
    """Each G = B B^T converts squared PC distance to a physical wind variance."""
    eof = rep.blocks["wind"]
    k = eof.basis.shape[1]
    area = region_area(rep.lat, rep.lon)
    ocean = area * rep.blocks["sst"].valid[0]
    ocean /= ocean.sum()
    maps = (eof.inverse(np.eye(k)) - eof.inverse(np.zeros((1, k)))) * np.sqrt(ocean)
    G = [steering_metric(rep, k)]
    for c in range(4):
        basis = maps[:, c].reshape(k, -1)
        G.append(basis @ basis.T)
    return area, ocean, np.asarray(G)


def trace_lags(dynamics, G, horizon):
    """C_h = A^h Sigma. H implicitly extracts the first k state coordinates."""
    k = dynamics.observed_dim
    product = dynamics.sigma[:, :k].copy()
    out = []
    for _ in range(horizon):
        out.append(np.einsum("nij,ij->n", G, product[:k]))
        product = dynamics.a @ product
    return np.asarray(out).T
