"""Evaluate a fixed deployable model, retaining individual members and windows for paired intervals."""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from between import finite_member_variance
from joint import PhysicalValidation
from ocean import FIELDS, data_paths, panel_cohort, physical_scores
from records import WORK, record
from weather import model_terms, projected

from annual_lim.layout import DEFAULT_WORK, read
from annual_lim.model import MonthlyModel
from annual_lim.statistics import metrics, variance, window_moments
from annual_lim.train import months
from annual_lim.validate import physical
from tc_emulator.common import atomic, save_json
from tc_emulator.diagnostics import steering


def forecast_coefficients(model, leads):
    """Gaussian regression of future weather on the last two observable, filtered wind anomalies."""
    cal = model.cal
    k = model.base.observed_dim
    first, last = cal.delay, cal.last_lag
    taps = np.zeros(last + 1)
    taps[0] = 1.0
    if last:
        taps[first:] = -cal.feedback / (last - first + 1)
    coeff = np.correlate(taps, taps, mode="full")
    product = model.base.sigma[:, :k].copy()
    covs = []
    for h in range(max(leads) + 2 + last):
        covs.append(product[:k].copy())
        product = (cal.retention * model.base.a) @ product

    def filtered(h):
        return sum(
            c * (covs[j] if j >= 0 else covs[-j].T)
            for j, c in zip(range(h - last, h + last + 1), coeff)
        )

    C0 = filtered(0)
    C1 = filtered(1)
    context = np.block([[C0, C1], [C1.T, C0]])
    out = {}
    for h in leads:
        cross = np.column_stack((filtered(h), filtered(h + 1)))
        out[h] = np.linalg.solve(context, cross.T).T
    return out


def verification(stage, weights, cohort, month, role, draws):
    frozen = WORK / "FROZEN.json"
    if role != "val":
        if not frozen.is_file():
            raise ValueError("The test panel opens after the primary recipe has been frozen")
        receipt = read(frozen)
        if stage not in (receipt["stage"], "published_v1"):
            raise ValueError("Additional test candidates would invalidate the frozen comparison")
        if (
            stage != "published_v1"
            and read(Path(weights) / f"{month:02}/model.json")["id"]
            != receipt["months"][f"{month:02}"]
        ):
            raise ValueError("Weight hash differs from the frozen test candidate")
    target = WORK / "verification" / stage / role / f"{month:02}"
    if (target / "complete.json").exists():
        return
    tick = time.monotonic()
    record("model_verification_start", stage=stage, month=month, role=role)
    model = MonthlyModel(weights, month)
    panel, actual_role = panel_cohort(role)
    config, data_root = data_paths(panel)
    data, records, _ = months(config, data_root, month, actual_role)
    meta = pd.DataFrame(records)
    g = meta.g.to_numpy()
    years = meta.year.to_numpy()
    vals, first = np.unique(years, return_index=True)
    lengths = meta.days.to_numpy()
    area, ocean, G = metrics(model.rep)
    winds = [d["wind"] for d in data]
    obs_mu, seconds = window_moments(winds, area)
    within = (
        seconds
        - 2 * np.einsum("nwf,nf->nw", obs_mu, obs_mu[:, -1])
        + np.einsum("nf,nf->n", obs_mu[:, -1], obs_mu[:, -1])[:, None]
    )
    original = MonthlyModel(DEFAULT_WORK / "weights", month)
    # score grid and mask refer to the published region, independent of candidate rank.
    from types import SimpleNamespace

    scoring = PhysicalValidation(
        SimpleNamespace(geo=original.rep, month=month),
        np.array([np.stack([item[k] for k in FIELDS]) for item in data]),
        meta,
        obs_mu[:, -1],
    )
    obs_bg = np.stack([model.rep.encode(x) for x in data])
    prob = scoring.ocean_scores(model)
    wind_score = scoring.wind_errors(model, obs_bg)
    projections = []
    truth = scoring.truth
    decoded = model.rep.decode(obs_bg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for fi, field in enumerate(FIELDS):
            key = "sst" if field == "sst" else "ocean"
            ch = 0 if field != "gamma" else 1
            base = original.rep.blocks[key]
            # Fixed training climatology and tail threshold are shared across candidates, supplied by the original training set.
            ref = WORK / "verification" / "original_ocean_climatology" / f"{month:02}.npz"
            if not ref.is_file():
                from ocean import ocean_data

                tr, _ = ocean_data("original", month, "train")
                with atomic(ref) as tmp:
                    np.savez(
                        tmp,
                        clim=np.nanmean(tr.astype(float), axis=0),
                        tail=np.nanquantile(tr, 0.9, axis=0),
                    )
            with np.load(ref) as f:
                clim, tail = f["clim"][fi], f["tail"][fi]
            predicted = decoded[field]
            if field == "gamma":
                predicted = np.maximum(0.0, predicted)
            if field == "mld":
                predicted = np.maximum(0.01, predicted)
            for region, ar in scoring.regions.items():
                projections.append(
                    physical_scores(
                        truth[:, fi],
                        predicted,
                        clim,
                        tail,
                        ar.reshape(area.shape),
                        base.valid[ch],
                        meta,
                        field,
                    ).assign(region=region)
                )
    lags, bg, forced = model_terms(model, g[first])
    total, center, _ = projected(model.cal, lags, g[first], lengths[first], bg)
    expanded = np.searchsorted(vals, years)
    groupmask = {
        "all": np.ones(len(g), bool),
        "cold": g < -0.0115220616351514,
        "inside": (g >= -0.0115220616351514) & (g <= 3.419637285227324),
        "warm": g > 3.419637285227324,
    }
    wrows = []
    base_data = dict(mu=obs_mu, second=seconds, member=meta.member.to_numpy())
    for group, mask in groupmask.items():
        if not mask.any():
            continue
        ii = expanded[mask]
        targets = variance(obs_mu[mask], seconds[mask])
        corrected = finite_member_variance(base_data, mask)
        sim = total[ii].mean(axis=0) + np.var(g[mask]) * forced
        for j, label in enumerate(("1", "5", "10", "15", "month")):
            wrows.append(
                dict(
                    group=group,
                    window=label,
                    target="total",
                    truth=targets[j],
                    unbiased_truth=corrected[j],
                    model=sim[j],
                )
            )
            if j < 4:
                obs = within[mask, j].mean()
                wrows.append(
                    dict(
                        group=group,
                        window=label,
                        target="within",
                        truth=obs,
                        unbiased_truth=obs,
                        model=center[ii, j].mean(),
                    )
                )
    # Projecting the published data and comparing physical wind errors keeps forecasts comparable across PC ranks.
    coefs = forecast_coefficients(model, [1, 5, 10, 15])
    forecast = []
    for i, (w, z) in enumerate(zip(winds, obs_bg)):
        pc = model.rep.blocks["wind"].transform(w)
        mean = model.wind_mean(g[i], z)
        anomaly = pc - mean
        for lead, coef in coefs.items():
            init = np.arange(1, len(w) - lead)
            pred = mean + np.column_stack((anomaly[init], anomaly[init - 1])) @ coef.T
            sim = model.rep.blocks["wind"].inverse(pred)

            def error(left, right):
                s = steering(left - right)
                return float(np.mean((s * s * area).sum(axis=(-2, -1))))

            forecast.append(
                dict(
                    member=meta.member[i],
                    year=int(years[i]),
                    g=g[i],
                    lead=lead,
                    lim_mse=error(sim, w[init + lead]),
                    persistence_mse=error(w[init], w[init + lead]),
                    mean_mse=error(model.rep.blocks["wind"].inverse(mean[None]), w[init + lead]),
                )
            )
    # Batched storage of physical winds is limited to a single independent ensemble draw at a time.
    fit = read(DEFAULT_WORK / "fit" / f"{month:02}.json")
    maps = {}
    for maskname, mask in groupmask.items():
        if mask.any():
            maps[maskname] = physical(
                [winds[i] for i in np.flatnonzero(mask)], ocean, fit["weak_cut"]
            )
    phys = []
    for draw in range(draws):
        simulated = []
        for i, r in enumerate(records):
            ident = f"{actual_role}/{r['member']}/{draw}"
            z = model.draw_z(r["g"], r["year"], 9152026, ident)
            pc = model.pc_path(r["g"], z, r["days"], r["year"], 9152026, ident, draw)
            simulated.append(model.rep.blocks["wind"].inverse(pc))
        for grp, mask in groupmask.items():
            if not mask.any():
                continue
            p = physical([simulated[i] for i in np.flatnonzero(mask)], ocean, fit["weak_cut"])
            a = maps[grp]
            phys.append(
                dict(
                    group=grp,
                    draw=draw,
                    mean_wind_ms=float(np.max(abs(p["mean_ms"] - a["mean_ms"]))),
                    shear_quantile_ms=float(np.max(abs(p["shear_ms"] - a["shear_ms"]))),
                    weak_spells_per_month=float(abs(p["weak_run_count"] - a["weak_run_count"])),
                    lag_bias=float(np.max(abs(p["acf"][:, 1:] - a["acf"][:, 1:]))),
                    q95_shear=p["shear_ms"][-1],
                    ref_q95_shear=a["shear_ms"][-1],
                )
            )
    for key, frame in (
        ("ocean", prob),
        ("projection", pd.concat(projections)),
        ("monthly_wind", wind_score),
        ("windows", pd.DataFrame(wrows)),
        ("forecast", pd.DataFrame(forecast)),
        ("physics", pd.DataFrame(phys)),
    ):
        with atomic(target / (key + ".parquet")) as tmp:
            frame.to_parquet(tmp, index=False, compression="zstd")
    # Sufficient member-year matrices are needed for pairwise uncertainty in multi-day wind.
    with atomic(target / "panel_moments.npz") as tmp:
        np.savez_compressed(
            tmp,
            year=years,
            member=meta.member.to_numpy(dtype=str),
            g=g,
            mu=obs_mu,
            second=seconds,
            within=within,
            predicted_total=total[expanded],
            predicted_within=center[expanded],
            forced=forced,
        )
    save_json(
        target / "complete.json",
        dict(
            stage=stage,
            month=month,
            role=role,
            model_id=model.meta["id"],
            records=len(meta),
            members=sorted(meta.member.unique()),
            years=sorted(map(int, vals)),
            weather_draws=draws,
            spectral_radius=model.base.rho,
            lyapunov_residual=model.base.residual,
            cohort=cohort,
        ),
    )
    record(
        "model_verification_done",
        stage=stage,
        month=month,
        role=role,
        cases=len(meta),
        draws=draws,
        seconds=round(time.monotonic() - tick, 1),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--cohort", default="original")
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--role", choices=["val", "test", "fresh", "ssp245_year"], default="val")
    p.add_argument("--draws", type=int, default=8)
    a = p.parse_args()
    verification(a.label, a.weights, a.cohort, a.month, a.role, a.draws)
