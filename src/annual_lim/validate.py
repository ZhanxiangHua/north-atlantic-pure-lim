"""Verify frozen monthly weights against separate MPI identities and years."""
from pathlib import Path

import numpy as np

from tc_emulator.common import save_json, sha
from tc_emulator.diagnostics import steering
from tc_emulator.fourier_weather import grid_covariance
from tc_emulator.warming_circulation import transfer_matrix
from .layout import design, read, rng_for
from .model import MonthlyModel
from .statistics import LABELS, component_cov, metrics, quantiles, shear_hist, trace_lags, window_moments
from .train import months


def fourier_power(days):
    # The supplied TC package uses 15 independent sine phases, period 20 days
    # and variance weights n^-3. Integrate its linearly interpolated hourly wind.
    n = np.arange(1.0, 16.0)
    weight = n**-3
    wave = np.exp(2j * np.pi * n[:, None] * np.arange(24 * days + 1) / (24 * 20))
    return float(weight @ abs(np.trapezoid(wave, axis=1) / (24 * days)) ** 2 / weight.sum())


def expected(model, records, area, G):
    g = np.array([r["g"] for r in records])
    gs = np.unique(g)
    k = model.base.observed_dim
    lags = trace_lags(model.base, G, 51)[0]
    transfer = transfer_matrix(model.mean)
    kernel = transfer @ G[0] @ transfer.T
    cov = grid_covariance(model.rep, k, model.base.sigma[:k, :k])
    if np.any(np.linalg.eigvalsh(cov[area > 0])[:, 0] <= 0):
        raise ValueError("TC-package point covariance must be positive definite")
    means = {}
    variances = {}
    bg_power = {}
    for v in gs:
        z = model.bg.mean([v])[0]
        base = steering(model.rep.blocks["wind"].inverse(model.wind_mean(v, z)[None])[0]) * np.sqrt(
            area / 2
        )
        means[v] = base.ravel()
        F = model.factor(v)
        bg_power[v] = float(np.sum((F @ F.T) * kernel))
    mu = np.stack([means[v] for v in g])
    methods = ("pure_lim", "original_lim", "iid", "fourier")
    for r in records:
        v, nd = r["g"], r["days"]
        if (v, nd) in variances:
            continue
        windows = [1, 5, 10, 15, nd]
        pure = model.cal.window_variance(v, lags, windows)[0]
        original = np.array(
            [(d * lags[0] + 2 * lags[1:d] @ np.arange(d - 1, 0, -1)) / d**2 for d in windows]
        )
        iid = lags[0] / np.array(windows)
        fourier = lags[0] * np.array([fourier_power(d) for d in windows])
        variances[v, nd] = {
            m: bg_power[v] + a for m, a in zip(methods, (pure, original, iid, fourier), strict=True)
        }
    return mu, {m: np.stack([variances[r["g"], r["days"]][m] for r in records]) for m in methods}


def crossed_weights(records, seed):
    # Resample complete members and complete years; all rolling windows stay together.
    members, mi = np.unique([r["member"] for r in records], return_inverse=True)
    years, yi = np.unique([r["year"] for r in records], return_inverse=True)
    if len(records) != len(members) * len(years):
        raise ValueError("Expected a complete member by year rectangle")
    rng = np.random.default_rng(seed)
    wm = rng.multinomial(len(members), np.ones(len(members)) / len(members), size=400)
    wy = rng.multinomial(len(years), np.ones(len(years)) / len(years), size=400)
    weights = wm[:, mi] * wy[:, yi]
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.vstack((np.ones(len(records)) / len(records), weights))


def wind_score(obs_mu, obs_second, model_mu, variances, weights, bands):
    def mean_square(mu):
        gram = mu @ mu.T
        return np.einsum("bi,bi->b", weights @ gram, weights)

    shared = weights @ np.square(model_mu).sum(axis=1) - mean_square(model_mu)
    out = []
    for i, label in enumerate(LABELS):
        raw_second = weights @ obs_second[:, i]
        truth = raw_second - mean_square(obs_mu[:, i])
        # A resample can consist solely of one member in one year. Its full-month
        # mean has exactly zero observed variance; a generated/observed ratio is undefined.
        # Exclude only these draws from that window, retaining the same pairs for every model.
        positive = truth > 64 * np.finfo(float).eps * np.maximum(1.0, abs(raw_second))
        if not positive[0]:
            raise ValueError(f"No measurable observed steering variance in the {label}-day cohort")
        ratios = {
            m: ((shared + weights @ v[:, i])[positive] / truth[positive]) for m, v in variances.items()
        }

        def interval(values):
            return (
                tuple(map(float, np.quantile(values, [0.025, 0.975])))
                if len(values) > 1
                else (None, None)
            )

        for m, ratio in ratios.items():
            lo, hi = interval(ratio[1:])
            err = abs(np.log(ratio)) - abs(np.log(ratios["fourier"]))
            paired_lo, paired_hi = interval(err[1:])
            row = dict(
                window=label,
                model=m,
                observed=float(truth[0]),
                ratio=float(ratio[0]),
                ci_low=lo,
                ci_high=hi,
                bootstrap_valid=len(ratio) - 1,
                abs_log_minus_fourier=float(err[0]),
                paired_low=paired_lo,
                paired_high=paired_hi,
            )
            if label in ("5", "10", "15"):
                low, high = bands[i]
                row.update(
                    in_band=bool(low <= ratio[0] <= high),
                    closer_than_iid=bool(abs(ratio[0] - 1) < abs(ratios["iid"][0] - 1)),
                )
            out.append(row)
    return out


def physical(paths, ocean, weak):
    mean, cov = component_cov(paths, ocean)
    hist, count = shear_hist(paths, ocean, weak)
    return dict(
        mean_ms=np.einsum("cyx,yx->c", mean, ocean),
        acf=cov / cov[:, :1],
        shear_ms=quantiles(hist, [0.05, 0.5, 0.95]),
        weak_run_count=np.array(count),
    )


def background_means(samples, area, plev):
    # Regional means use area weights renormalized over valid ocean cells for each field.
    j = int(np.flatnonzero(plev == 60000).item())
    out = {}
    for name in ("sst", "mld", "gamma", "psl", "ta", "hus"):

        def integrate(ds):
            a = ds[name][j] if name in ("ta", "hus") else ds[name]
            if name == "mld":
                a = np.maximum(0.01, a)
            if name == "gamma":
                a = np.maximum(0.0, a)
            if name == "hus":
                a = np.clip(a, 0.0, 0.10)
            valid = np.isfinite(a)
            return np.sum(a[valid] * area[valid]) / area[valid].sum()

        out[name] = np.array([integrate(ds) for ds in samples])
    return out


def assess(config, work, month, role):
    settings, _ = design(config)
    model = MonthlyModel(Path(work) / "weights", month)
    fit = read(Path(work) / "fit" / f"{month:02}.json")
    if fit["id"] != model.meta["id"]:
        raise ValueError("Different monthly calibration")
    sampled, records, grid = months(config, work, month, role)
    if {r["member"] for r in fit["training"]} & {r["member"] for r in records}:
        raise ValueError("Validation leakage")
    area, ocean, G = metrics(model.rep)
    winds = [ds["wind"] for ds in sampled]
    obs_mu, seconds = window_moments(winds, area)
    model_mu, vars_ = expected(model, records, area, G)
    draws = 2
    sim, backgrounds = [], []
    for r in records:
        for j in range(draws):
            ident = f'{role}/{r["member"]}/{j}'
            z = model.draw_z(r["g"], r["year"], settings["seed"], ident)
            pc = model.pc_path(r["g"], z, r["days"], r["year"], settings["seed"], ident, j)
            sim.append(model.rep.blocks["wind"].inverse(pc))
            backgrounds.append(model.rep.decode(z))
    observed_bg = background_means(sampled, ocean, grid[2])
    model_bg = background_means(backgrounds, ocean, grid[2])
    levels = np.array([r["g"] for r in records])
    groups = {
        "all": np.ones(len(levels), bool),
        "cold": levels < model.cal.g_min,
        "inside": (levels >= model.cal.g_min) & (levels <= model.cal.g_max),
        "warm": levels > model.cal.g_max,
    }
    result = {}
    for group, mask in groups.items():
        ii = np.flatnonzero(mask)
        if not len(ii):
            continue
        refs = [records[i] for i in ii]
        weight = crossed_weights(refs, settings["seed"] + month)
        scores = wind_score(
            obs_mu[ii],
            seconds[ii],
            model_mu[ii],
            {m: v[ii] for m, v in vars_.items()},
            weight,
            fit["bands"],
        )
        jj = (ii[:, None] * draws + np.arange(draws)).ravel()
        actual = physical([winds[i] for i in ii], ocean, fit["weak_cut"])
        generated = physical([sim[i] for i in jj], ocean, fit["weak_cut"])
        effects = dict(
            mean_wind_ms=float(max(abs(generated["mean_ms"] - actual["mean_ms"]))),
            shear_quantile_ms=float(max(abs(generated["shear_ms"] - actual["shear_ms"]))),
            weak_spells_per_month=float(abs(generated["weak_run_count"] - actual["weak_run_count"])),
            lag_bias=float(abs(generated["acf"][:, 1:] - actual["acf"][:, 1:]).max()),
        )
        limits = dict(mean_wind_ms=2.0, shear_quantile_ms=3.0, weak_spells_per_month=0.4, lag_bias=0.2)
        bg = {
            m: {"observed": float(observed_bg[m][ii].mean()), "generated": float(model_bg[m][jj].mean())}
            for m in observed_bg
        }
        result[group] = dict(
            months=len(ii),
            years=sorted({r["year"] for r in refs}),
            g_range=[levels[ii].min(), levels[ii].max()],
            windows=scores,
            physics=effects,
            physics_limits=limits,
            background=bg,
            physics_pass=all(effects[k] <= limits[k] for k in limits),
            wind_pass=all(
                s["in_band"] and s["closer_than_iid"]
                for s in scores
                if s["model"] == "pure_lim" and s["window"] in ("5", "10", "15")
            ),
        )
    target = Path(work) / "validation" / role / f"{month:02}.json"
    save_json(
        target,
        dict(
            month=month,
            role=role,
            model_id=model.meta["id"],
            records=records,
            weather_draws=draws,
            groups=result,
            validation_code_sha256=sha(Path(__file__)),
        ),
    )
    print(
        month,
        role,
        [
            (s["window"], round(s["ratio"], 3))
            for s in result["all"]["windows"]
            if s["model"] == "pure_lim"
        ],
        flush=True,
    )
