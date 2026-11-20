"""Fit calendar-specific EOFs, conditional background, VAR(2), and memory calibration."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import os

import numpy as np
from scipy.optimize import least_squares

from tc_emulator.common import digest, lock, save_json, save_npz, sha
from tc_emulator.pure_lim_memory import MemoryCalibration, response
from tc_emulator.representation import BLOCKS, Representation, eof, stack_fields, vertical_weights
from tc_emulator.warming_circulation import fit_covariance, refit_backbone, transfer_matrix
from tc_emulator.wind_lim import from_statistics, pair_statistics
from .layout import case_dir, design, read
from .statistics import (
    LABELS,
    component_cov,
    member_bands,
    metrics,
    quantiles,
    shear_hist,
    trace_lags,
    variance,
    window_moments,
)


def months(config, work, month, role):
    _, cases = design(config)
    chosen = [c for c in cases if c[0] == role]
    data, records = [], []
    grid = None
    for _, member, year, g in chosen:
        folder = case_dir(work, member, year)
        mark = read(folder / "complete.json")
        file = folder / f"{month:02}.npz"
        if mark["role"] != role or mark["g"] != g or sha(file) != mark["hashes"][file.name]:
            raise ValueError(f"Changed input {file}")
        if mark["calendar"] != "proleptic_gregorian":
            raise ValueError(
                "This MPI release and its inference calendar must use proleptic Gregorian dates"
            )
        with np.load(file, allow_pickle=False) as src:
            if str(src["member"]) != member or int(src["year"]) != year or int(src["month"]) != month:
                raise ValueError(f"Member-year-month mismatch {file}")
            ds = {v: src[v] for v in ("sst", "mld", "gamma", "psl", "ta", "hus", "wind")}
            if len(ds["wind"]) != mark["days"][month - 1]:
                raise ValueError(f"Missing days in {file}")
            coords = tuple(src[v] for v in ("lat", "lon", "plev"))
            if grid is None:
                grid = coords
            elif any(not np.array_equal(a, b) for a, b in zip(grid, coords, strict=True)):
                raise ValueError(f"Unexpected grid {file}")
        data.append(ds)
        records.append(
            dict(member=member, year=year, g=g, days=len(ds["wind"]), sha256=mark["hashes"][file.name])
        )
    return data, records, grid


def spatial_fields(samples, grid, settings):
    """Fit U to weighted physical anomalies. Encode as U.T @ D @ (field - mean)."""
    lat, lon, plev = grid
    area = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (len(lat), len(lon)))
    blocks = {}
    for block, fields in BLOCKS.items():
        x = np.stack([stack_fields(ds, fields, settings["log_mld"]) for ds in samples])
        blocks[block] = eof(
            x, area, vertical_weights(fields, plev), settings["ranks"][block], settings["seed"]
        )
    # Concatenation here is only for spatial EOFs. Dynamics below keep all paths separate.
    wind = np.concatenate([ds["wind"] for ds in samples])
    blocks["wind"] = eof(wind, area, np.ones(4) / 4, settings["ranks"]["wind"], settings["seed"])
    return Representation(blocks, lat, lon, plev, settings["log_mld"])


def fit_dynamics(paths, mean, settings):
    """LIM ridge fit M = C_yx (C_xx + lambda I)^-1, then solve Sigma = A Sigma A.T + Q."""
    totals = {}
    k = paths[0].shape[-1]
    for length in sorted({len(w) for w in paths}):
        res = np.stack([w - m for w, m in zip(paths, mean, strict=True) if len(w) == length])
        # Inputs are [x_(t-1), x_(t-2)] from the same member, year and month as x_t.
        for name, v in pair_statistics(res, 2).items():
            totals[name] = totals.get(name, 0) + v
    # Use the smallest configured ridge yielding a stable stationary fit.
    for ridge in settings["lim_ridge_candidates"]:
        try:
            return from_statistics(totals, k, ridge, 2), ridge, int(totals["count"])
        except (ValueError, np.linalg.LinAlgError):
            if ridge == settings["lim_ridge_candidates"][-1]:
                raise


def memory_fit(g, lengths, lags, between, truth, acfs, settings):
    """Fit daily amplitude, its warming slope, damping r and past-mean coefficient b."""
    first, last = settings["memory_lags"]

    def parameter(theta):
        intercept, slope, r, b = map(float, theta)
        v = response(lags[0], r, first, last, b, 1)[0]
        # kappa² = unfiltered daily steering variance / filtered daily steering variance.
        return MemoryCalibration(
            "distributed_memory",
            float(g.mean()),
            float(g.min()),
            float(g.max()),
            settings["tail_extension_K"],
            intercept,
            slope,
            r,
            delay=first,
            last_lag=last,
            feedback=b,
            normalization=float(np.sqrt(lags[0, 0] / v)),
        )

    def residual(theta):
        p = parameter(theta)
        var = np.empty_like(truth)
        for nd in np.unique(lengths):
            use = lengths == nd
            var[use] = p.window_variance(g[use], lags[0], (1, 5, 10, 15, int(nd)))
        # Compare total wind variance, including the circulation predicted from z.
        logerr = np.log((var + between[:, :1]) / truth)
        trace = response(lags[1:], p.retention, first, last, p.feedback, 16)
        # The fixed monthly mean contributes the same covariance at every lag.
        cov = between[:, 1:, None] + (p.amplitude(g) * p.normalization)[:, None, None] ** 2 * trace
        acerr = cov[:, :, 1:] / cov[:, :, :1] - acfs[:, :, 1:]
        return np.r_[logerr.ravel() / np.sqrt(5), acerr.ravel() / np.sqrt(4 * 15)]

    opt = least_squares(
        residual,
        [0.0, -0.08, 0.85, 0.8],
        bounds=([-0.7, -0.35, 0.65, 0.0], [0.7, 0.35, 1.0, 1.2]),
        ftol=1e-11,
        xtol=1e-11,
    )
    if not opt.success:
        raise RuntimeError(opt.message)
    return parameter(opt.x)


def fit(config, work, month):
    settings, _ = design(config)
    weights = Path(work) / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    dest = weights / f"{month:02}"
    with lock(Path(work) / "locks" / f"fit_{month:02}"):
        samples, records, grid = months(config, work, month, "train")
        import tc_emulator.representation as rep_code
        import tc_emulator.pure_lim_memory as memory_code
        import tc_emulator.warming_circulation as bg_code
        import tc_emulator.wind_lim as wind_code
        from tc_emulator import background, common, diagnostics, pure_lim
        from . import layout, statistics, model

        source = [
            Path(__file__),
            *(
                Path(s.__file__)
                for s in (
                    layout,
                    statistics,
                    model,
                    rep_code,
                    memory_code,
                    bg_code,
                    wind_code,
                    background,
                    common,
                    diagnostics,
                    pure_lim,
                )
            ),
        ]
        fingerprint = digest([settings, records, {p.name: sha(p) for p in source}])
        stat_path = Path(work) / "fit" / f"{month:02}.json"
        if (dest / "model.json").exists():
            meta = read(dest / "model.json")
            if meta["training_fingerprint"] != fingerprint:
                raise ValueError("Existing model uses different inputs; select a fresh work directory")
            if digest({k: v for k, v in meta.items() if k != "id"}) != meta["id"]:
                raise ValueError(f"Invalid metadata in {dest}. Use a fresh work directory.")
            healthy = all(
                (dest / name).is_file() and sha(dest / name) == v for name, v in meta["files"].items()
            )
            if stat_path.exists() and healthy and read(stat_path)["id"] == meta["id"]:
                return
        print("Fitting spatial EOFs", month, len(records), flush=True)
        rep = spatial_fields(samples, grid, settings)
        z = np.stack([rep.encode(ds) for ds in samples])
        winds = [ds["wind"] for ds in samples]
        pcs = [rep.blocks["wind"].transform(w) for w in winds]
        g = np.array([r["g"] for r in records])
        years = np.array([r["year"] for r in records])
        # Background z = m(g) + L(g)e. Mean wind = a(g) + K.T @ (z - m(g)).
        bg, mean = refit_backbone(
            g, z, np.stack([w.mean(axis=0) for w in pcs]), years, settings["mean_ridge"]
        )
        mu = mean.predict(bg, g, z)
        base, ridge, npairs = fit_dynamics(pcs, mu, settings)
        area, ocean, G = metrics(rep)
        # L.T @ K @ G @ K.T @ L ranks standardized background directions by
        # how much their variability contributes to the physical steering wind.
        law, details = fit_covariance(
            bg,
            mean,
            g,
            z,
            years,
            G[0],
            dict(
                wind_variance_fraction=settings["covariance_energy"],
                local_bandwidth_K=0.65,
                local_pooled_fraction=0.5,
                tail_extension_K=settings["tail_extension_K"],
            ),
        )
        lags = trace_lags(base, G, 31 + settings["memory_lags"][1])
        obs_mu, obs_sec = window_moments(winds, area)
        bands = member_bands(
            obs_mu,
            obs_sec,
            np.array([r["member"] for r in records]),
            settings["seed"],
            len(read(Path(config) / "members.json")["test"]),
        )
        hist, _ = shear_hist([w for i, w in enumerate(winds) if years[i] <= 1900], ocean)
        weak = float(quantiles(hist, 0.25))
        target, component_acf, between, lengths, anchors = [], [], [], [], []
        for y in np.unique(years):
            ids = np.flatnonzero(years == y)
            target.append(variance(obs_mu[ids], obs_sec[ids]))
            _, cov = component_cov([winds[i] for i in ids], ocean)
            component_acf.append(cov / cov[:, :1])
            centered = mu[ids] - mu[ids].mean(axis=0)
            between.append(np.einsum("ni,mij,nj->m", centered, G, centered, optimize=True) / len(ids))
            anchors.append(g[ids[0]])
            lengths.append(len(winds[ids[0]]))
        cal = memory_fit(
            np.array(anchors),
            np.array(lengths),
            lags,
            np.array(between),
            np.array(target),
            np.array(component_acf),
            settings,
        )
        with tempfile.TemporaryDirectory(dir=weights) as temp:
            package = Path(temp) / "month"
            package.mkdir()
            rep.save(package / "representation.npz")
            save_npz(
                package / "parameters.npz",
                dict(
                    center=bg.center,
                    g_scale=bg.scale,
                    beta=bg.beta,
                    background_root=bg.lower0,
                    directions=law.directions,
                    background_response=[law.active_scale, law.log_slope],
                    regression=mean.coefficients,
                    z_scale=mean.scale,
                    transition=base.a,
                    innovation=base.q,
                    stationary=base.sigma,
                ),
            )
            meta = dict(
                format=1,
                month=month,
                calendar="proleptic_gregorian",
                training_fingerprint=fingerprint,
                weather=asdict(cal),
                files={k: sha(package / k) for k in ("representation.npz", "parameters.npz")},
            )
            meta["id"] = digest(meta)
            save_json(package / "model.json", meta)
            if not dest.exists():
                package.rename(dest)
            else:
                original = read(dest / "model.json")
                if original != meta:
                    raise ValueError(
                        "Refit differs from the archived weights. Use a fresh work directory."
                    )
                # With an unchanged model ID, any restored file has exactly the old published bytes.
                for name, expected in meta["files"].items():
                    target = dest / name
                    if not target.is_file() or sha(target) != expected:
                        os.replace(package / name, target)
        save_json(
            stat_path,
            dict(
                month=month,
                id=meta["id"],
                training=records,
                labels=LABELS,
                bands=bands.tolist(),
                weak_cut=weak,
                total_days=sum(len(w) for w in winds),
                var2_triples=npairs,
                lim_ridge=ridge,
                spectral_radius=base.rho,
                lyapunov_relative_error=base.residual,
                background=details,
                calibration=asdict(cal),
                explained_variance={
                    k: float(b.variance.sum() / b.total_variance) for k, b in rep.blocks.items()
                },
            ),
        )
        print("Fitted", month, meta["id"], "rho", base.rho, flush=True)
