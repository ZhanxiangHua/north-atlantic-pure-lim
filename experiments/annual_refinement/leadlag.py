"""Compare directional cross-component wind memory with native and fitted Fourier spectra."""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from ocean import GROUP_LIMITS, data_paths, panel_cohort
from paired_intervals import panel_weights, record_estimate
from records import WORK, record
from tuned_fourier import calibrated, harmonic_lags

from annual_lim.layout import DEFAULT_WORK, case_dir, design, read
from annual_lim.model import MonthlyModel
from annual_lim.statistics import trace_lags
from tc_emulator.common import atomic, lock, sha

LAGS = (0, 1, 3, 5, 10, 15)
REGIONS = ("basin", "corridor")
COMPONENTS = ("u250", "v250", "u850", "v850")


def areas(lat, lon):
    y, x = np.meshgrid(lat, lon, indexing="ij")
    area = np.cos(np.deg2rad(y))
    weights = np.stack(
        [
            area * ((y >= s) & (y <= n) & (x >= w) & (x <= e))
            for s, n, w, e in ((0, 60, 260, 360), (10, 30, 280, 340))
        ]
    )
    return weights.reshape(len(weights), -1) / weights.sum(axis=(1, 2))[:, None]


def observed_covariances(wind, weights):
    """Matrix entry (a,b) is future component a times preceding component b."""
    x = wind.reshape(len(wind), 4, -1).astype(float)
    x -= x.mean(axis=0)
    output = np.empty((len(weights), len(LAGS), 4, 4))
    for j, w in enumerate(weights):
        field = (x * np.sqrt(w)).transpose(1, 0, 2)
        for i, h in enumerate(LAGS):
            span = len(x) - h
            past = field[:, :span].reshape(4, -1)
            future = field[:, h:].reshape(4, -1)
            output[j, i] = future @ past.T / span
    return output


def target(month, role):
    cache = WORK / "leadlag/reference" / role / f"{month:02}.npz"
    with lock(cache.with_suffix(".lock")):
        if not cache.is_file():
            cohort, sub = panel_cohort(role)
            cfg, work = data_paths(cohort)
            cases = [c for c in design(cfg)[1] if c[0] == sub]
            values = []
            lengths = []
            for _, m, y, _ in cases:
                folder = case_dir(work, m, y)
                name = f"{month:02}.npz"
                if sha(folder / name) != read(folder / "complete.json")["hashes"][name]:
                    raise ValueError(f"Wind fields changed for {m}/{y}/{month}")
                with np.load(folder / name, allow_pickle=False) as f:
                    wind = f["wind"]
                    values.append(observed_covariances(wind, areas(f["lat"], f["lon"])))
                    lengths.append(len(wind))
            with atomic(cache) as tmp:
                np.savez(
                    tmp,
                    observed=np.stack(values),
                    length=lengths,
                    member=[c[1] for c in cases],
                    year=[c[2] for c in cases],
                    g=[c[3] for c in cases],
                )
    with np.load(cache, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def ordered_filter(positive, taps, retention, days=31):
    """The covariance at a negative lag is the transpose of its positive-lag counterpart."""
    length = len(taps) - 1
    if len(positive) < days + length:
        raise ValueError("Insufficient lag sequence for the weather filter")
    damped = positive * retention ** np.arange(len(positive)).reshape(
        (-1,) + (1,) * (positive.ndim - 1)
    )
    full = np.concatenate((damped[:0:-1].swapaxes(-2, -1), damped))
    coeff = np.correlate(taps, taps, mode="full")
    ix = len(damped) - 1 + np.arange(days)[:, None] + np.arange(-length, length + 1)
    return np.einsum("j,hjrxy->hrxy", coeff, full[ix], optimize=True)


def underlying_lags(model, horizon):
    eof = model.rep.blocks["wind"]
    k = eof.basis.shape[1]
    weights = areas(model.rep.lat, model.rep.lon)
    if not np.all(eof.valid):
        raise ValueError("The MPI atmospheric grid must be complete in all four components")
    B = (eof.basis / eof.weights[eof.valid, None]).reshape(4, -1, k)
    G = np.stack([np.einsum("api,bpj,p->abij", B, B, w, optimize=True) for w in weights])
    return trace_lags(model.base, G.reshape(-1, k, k), horizon).T.reshape(-1, 2, 4, 4)


def weather_shape(raw, cal):
    taps = np.zeros(cal.last_lag + 1)
    taps[0] = 1.0
    if cal.last_lag:
        taps[cal.delay :] = -cal.feedback / (cal.last_lag - cal.delay + 1)
    filtered = ordered_filter(raw, taps, cal.retention) * cal.normalization**2
    np.testing.assert_allclose(filtered[0], filtered[0].swapaxes(-2, -1), rtol=1e-9, atol=1e-9)
    return filtered


def emitted_lags(model):
    return weather_shape(underlying_lags(model, 31 + model.cal.last_lag), model.cal)


@lru_cache(maxsize=4)
def centered_kernel(days):
    """Weights of every signed time lag after removing a realization's own monthly mean."""
    M = np.eye(days) - 1 / days
    rows = []
    for lag in LAGS:
        gram = M[lag:].T @ M[: days - lag] / (days - lag)
        rows.append([np.diag(gram, k=-h).sum() for h in range(1 - days, days)])
    return np.asarray(rows)


def finite_lags(positive, days):
    matrix = np.concatenate((positive[days - 1 : 0 : -1].swapaxes(-2, -1), positive[:days]))
    return np.einsum("lh,hrab->rlab", centered_kernel(days), matrix, optimize=True)


def panel_model(model, meta, source, spectrum=None):
    if spectrum is not None:
        source = spectrum[:, None, None, None] * source[0]
    lengths = meta["length"]
    lookup = {n: finite_lags(source, int(n)) for n in np.unique(lengths)}
    return (
        np.stack([lookup[n] for n in lengths])
        * model.cal.amplitude(meta["g"])[:, None, None, None, None] ** 2
    )


def compare_month(tag, weights, cohort, month, role):
    if role != "val":
        receipt = read(WORK / "FROZEN.json")
        if (
            tag != receipt["stage"]
            or read(weights / f"{month:02}/model.json")["id"] != receipt["months"][f"{month:02}"]
        ):
            raise ValueError("Lagged cross-component tests require a frozen checkpoint")
        cohort = receipt["cohort"]
    saved = WORK / "leadlag" / tag / role / f"{month:02}.npz"
    if saved.is_file():
        return
    pub = MonthlyModel(DEFAULT_WORK / "weights", month)
    model = MonthlyModel(weights, month)
    tuned = calibrated(model, tag, cohort, month)
    meta = target(month, role)
    native = harmonic_lags(20.0, 3.0)
    results = {}
    raw = {id(obj): emitted_lags(obj) for obj in (pub, model)}
    for name, obj, spec in (
        ("v1_LIM", pub, None),
        ("candidate_LIM", model, None),
        ("v1_Fourier", pub, native),
        ("candidate_Fourier", model, native),
        ("candidate_tuned_Fourier", model, tuned),
    ):
        results[name] = panel_model(obj, meta, raw[id(obj)], spec)
    with atomic(saved) as tmp:
        np.savez(tmp, **results, **meta)
    record("cross_component_lags_verified", tag=tag, role=role, month=month, cases=len(meta["g"]))


def errors(obs, pred):
    std = np.sqrt(np.diagonal(obs[:, :, 0], axis1=-2, axis2=-1))
    denominator = std[:, :, :, None] * std[:, :, None, :]
    delta = (pred - obs) / denominator[:, :, None]
    anti = (delta - delta.swapaxes(-2, -1)) / 2
    a, b = np.triu_indices(4, k=1)
    return {
        "all_components": np.sqrt(np.mean(delta**2, axis=(-1, -2))),
        "directional": np.sqrt(np.mean(anti[..., a, b] ** 2, axis=-1)),
    }


def summarize(tag, role, draws=2000):
    root = WORK / "leadlag" / tag / role
    names = (
        "v1_LIM",
        "candidate_LIM",
        "v1_Fourier",
        "candidate_Fourier",
        "candidate_tuned_Fourier",
    )
    comparison = []
    entry = []
    for idx, group in enumerate(("all", "cold", "inside", "warm")):
        history = {}
        last = None
        for month in range(1, 13):
            with np.load(root / f"{month:02}.npz", allow_pickle=False) as f:
                g = f["g"]
                low, high = GROUP_LIMITS
                bins = np.where(g < low, "cold", np.where(g > high, "warm", "inside"))
                if group != "all" and not np.any(bins == group):
                    break
                w, nm, ny = panel_weights(f["member"], f["year"], g, group, draws, 20261120 + idx)
                ordered = list(zip(f["member"], f["year"]))
                if last is not None and last != ordered:
                    raise ValueError("Monthly member-year alignment failed")
                last = ordered
                block = f["observed"]

                def averaged(data):
                    return (w @ data.reshape(w.shape[1], -1)).reshape((len(w),) + data.shape[1:])

                ref = averaged(block)
                for name in ("MPI", *names):
                    fit = ref if name == "MPI" else averaged(f[name])
                    if name != "MPI":
                        for key, value in errors(ref, fit).items():
                            history.setdefault((name, key), []).append(value)
                    if group == "all":
                        for reg, region in enumerate(REGIONS):
                            for j, h in enumerate(LAGS):
                                for a, aa in enumerate(COMPONENTS):
                                    for b, bb in enumerate(COMPONENTS):
                                        entry.append(
                                            dict(
                                                month=month,
                                                region=region,
                                                lag=h,
                                                future=aa,
                                                past=bb,
                                                driver=name,
                                                covariance=fit[0, reg, j, a, b],
                                            )
                                        )
        if not history:
            continue
        reduced = {key: np.mean(value, axis=0) for key, value in history.items()}
        for (name, metric), base in reduced.items():
            if name == "candidate_LIM":
                continue
            for reg, region in enumerate(REGIONS):
                for j, h in enumerate(LAGS[1:], start=1):
                    record_estimate(
                        comparison,
                        group,
                        metric,
                        (region, str(h)),
                        base[:, reg, j],
                        reduced["candidate_LIM", metric][:, reg, j],
                        nm,
                        ny,
                        "raw",
                    )
                    comparison[-1]["comparator"] = name
    dest = WORK / "comparison" / tag / role
    for n, data in (("leadlag_intervals", comparison), ("leadlag_matrices", entry)):
        with atomic(dest / (n + ".csv")) as tmp:
            pd.DataFrame(data).to_csv(tmp, index=False)
    record("cross_component_intervals", tag=tag, role=role, draws=draws)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--role", choices=["val", "test", "fresh", "ssp245_year"], default="val")
    p.add_argument("--month", type=int)
    a = p.parse_args()
    if a.month is None:
        summarize(a.tag, a.role)
    else:
        compare_month(a.tag, a.weights, a.cohort, a.month, a.role)
