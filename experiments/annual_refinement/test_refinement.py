"""Numerical regressions for calendar windows, coupled Gaussian marginals, and the portable patch."""
import json
from types import SimpleNamespace

import numpy as np
from checkpoint import build as build_checkpoint
from checkpoint import checksum, restore
from modal_weather import adjust, mode_projectors, trace_basis
from nested import attach
from ocean_spread import modified
from scipy import linalg
from scipy.special import ndtri
from scipy.stats import qmc
from seasonal import cyclic
from spatial_check import paired_expected
from weather import kernels
from weather_controls import daily_fourier_lags

from annual_lim.statistics import trace_lags
from tc_emulator.background import Background, Circulation
from tc_emulator.wind_lim import Dynamics


def test_calendar_kernel_and_fourier_integral():
    lag = 0.7 ** np.arange(31)
    for n in (28, 29, 30, 31):
        C = linalg.toeplitz(lag[:n])
        raw, cent = kernels(n)
        for j, d in enumerate((1, 5, 10, 15, n)):
            H = np.array([np.isin(np.arange(n), np.arange(i, i + d)) / d for i in range(n - d + 1)])
            for actual, op in [(raw[j] @ lag, H), (cent[j] @ lag, H - 1 / n)]:
                np.testing.assert_allclose(actual, np.trace(op @ C @ op.T) / len(op), atol=1.0e-12)
    assert daily_fourier_lags().shape == (31,)


def test_periodic_smoothing_constant_and_new_year():
    constant = np.repeat([[0.1, 0.02, 0.82, 0.6]], 12, axis=0)
    np.testing.assert_allclose(cyclic(constant, 3.0, (0, 1, 2, 3)), constant, atol=1.0e-12)
    shape = constant.copy()
    shape[0, 1] = 0.22
    sm = cyclic(shape, 1.0, (1,))
    np.testing.assert_allclose(sm[1, 1], sm[-1, 1])
    np.testing.assert_allclose(sm[:, [0, 2, 3]], constant[:, [0, 2, 3]])


def test_modal_congruence_preserves_stationarity_and_trace():
    rng = np.random.default_rng(432)
    k = 6
    A = np.block(
        [[np.diag(np.linspace(0.15, 0.8, k)), 0.06 * np.eye(k)], [np.eye(k), np.zeros((k, k))]]
    )
    q = linalg.block_diag(np.eye(k), np.zeros((k, k)))
    C = linalg.solve_discrete_lyapunov(A, q)
    base = Dynamics(A, q, C, 2, k)
    b = rng.normal(size=(8, k, k))
    G = b @ np.swapaxes(b, 1, 2)
    for kind in ("timescale", "vertical", "joint"):
        p = mode_projectors(base, G, kind)
        v = np.linspace(0.8, 1.13, len(p))
        tensor = trace_basis(base, G, p, 51)
        adj = adjust(base, p, v)
        np.testing.assert_allclose(
            np.einsum("i,j,ijmh->mh", v, v, tensor), trace_lags(adj, G, 51), atol=1.0e-9
        )


def test_nested_marginal_and_wind_null_space():
    rng = np.random.default_rng(621)
    p, k = 78, 12
    s, o = 24, 30
    u = linalg.orth(rng.normal(size=(p, 4)))
    L = np.diag(np.linspace(0.5, 1.5, p))
    beta = rng.normal(scale=0.2, size=(3, p))
    years = np.repeat(np.arange(8), 30)
    g = (years - 3.5) / 3.5
    parent = SimpleNamespace(
        bg=Background(0.0, 1.0, beta, beta[:2], L, L * 0, L @ L.T),
        mean=Circulation(rng.normal(size=(p + 2, k)), np.ones(p)),
        directions=u,
        bg_amp=1.05,
        bg_slope=-0.12,
    )
    z = rng.normal(size=(len(g), s + o + 42))
    w = rng.normal(size=(len(g), k))
    bg, mean, directions = attach(parent, z, w, g, years, s, o, 0.0, 0.5, 0.5)
    idx = np.r_[np.arange(18), s + np.arange(18), s + o + np.arange(42)]
    np.testing.assert_allclose(
        parent.mean.predict(parent.bg, g, z[:, idx]), mean.predict(bg, g, z), atol=1.0e-11
    )

    class Model:
        def factor(self, g):
            a = np.sqrt(self.bg_amp * np.exp(self.bg_slope * g))
            return self.bg.lower0 + (a - 1) * (self.bg.lower0 @ self.directions) @ self.directions.T

    obj = Model()
    obj.bg = bg
    obj.directions = directions
    obj.mean = mean
    obj.bg_amp = parent.bg_amp
    obj.bg_slope = parent.bg_slope
    obj.cal = SimpleNamespace(g_min=-1.0, g_max=1.0)
    for v in [-2.0, 0.0, 2.0]:
        a = np.sqrt(parent.bg_amp * np.exp(parent.bg_slope * v))
        f = L + (a - 1) * (L @ u) @ u.T
        n = obj.factor(v)[idx]
        np.testing.assert_allclose(n @ n.T, f @ f.T, atol=1.0e-10)
    from tc_emulator.warming_circulation import transfer_matrix

    W = transfer_matrix(obj.mean)
    N = linalg.null_space(np.column_stack((obj.directions, obj.bg.lower0.T @ W)).T)
    changed = modified(obj, np.array_split(N, 2, axis=1), np.array([-0.2, 0.37]))
    assert not np.allclose(changed.bg.pooled, obj.bg.pooled)
    for g in [-2.0, 0.0, 2.0]:
        a, b = obj.factor(g), changed.factor(g)
        np.testing.assert_allclose(W.T @ a @ a.T @ W, W.T @ b @ b.T @ W, atol=1.0e-9)


def test_physical_variogram_log_mld_and_sst():
    q = ndtri(qmc.Sobol(2, scramble=True, seed=641).random_base2(17))
    m, v, n, w, c = (np.array([t]) for t in (2.0, 0.35, 1.4, 0.18, 0.12))
    x = m + np.sqrt(v) * q[:, 0]
    y = n + (c / np.sqrt(v)) * q[:, 0] + np.sqrt(w - c * c / v) * q[:, 1]
    for key, a, b in (("sst", x, y), ("mld", np.exp(x), np.exp(y))):
        analytic = paired_expected(key, m, v, n, w, c, q)[0]
        np.testing.assert_allclose(analytic, np.abs(a - b).mean(), rtol=4.0e-4)


def test_lossless_delta_restores_c_order_fortran_order_and_payload(tmp_path):
    rng = np.random.default_rng(310)
    x = rng.normal(size=(15, 20))
    for m in range(1, 13):
        name = f"{m:02}"
        for tag in ("base", "new"):
            folder = tmp_path / tag / name
            folder.mkdir(parents=True)
            a = x if tag == "base" else np.asfortranarray(x)
            b = np.arange(75.0) if tag == "base" else np.linspace(0, 1, 100)
            for file in ("representation.npz", "parameters.npz"):
                np.savez(folder / file, arr=a, delta=b, unchanged=x, scalar=np.asarray(False))
            (folder / "model.json").write_text(
                json.dumps(
                    dict(
                        id=f"{tag}{m}",
                        files={
                            n: checksum(folder / n)
                            for n in ("representation.npz", "parameters.npz")
                        },
                    )
                )
            )
    manifest = build_checkpoint(tmp_path / "base", tmp_path / "new", tmp_path / "payload")
    restore(tmp_path / "base", tmp_path / "payload", tmp_path / "restored")
    second = build_checkpoint(tmp_path / "base", tmp_path / "new", tmp_path / "payload_b")
    assert second == manifest
    for p in (tmp_path / "new").rglob("*"):
        if p.is_file():
            assert checksum(p) == checksum(tmp_path / "restored" / p.relative_to(tmp_path / "new"))


def test_calendar_pooled_ocean_keeps_coarse_coordinates_and_neighbor_climatology():
    from calendar_ocean import ocean_basis

    from tc_emulator.representation import EOF

    rng = np.random.default_rng(181)
    base = linalg.qr(rng.normal(size=(14, 3)), mode="economic")[0]
    parent = EOF(
        np.zeros((1, 2, 7)), np.ones((1, 2, 7)), np.ones((1, 2, 7), bool), base, np.ones(3), 14.0
    )
    fields = [rng.normal(size=(55, 1, 2, 7)) for _ in range(3)]
    result = ocean_basis(parent, fields, 0.0, 0.30, 9)
    shifted = ocean_basis(
        parent, [fields[0], fields[1] + rng.normal(size=(1, 2, 7)), fields[2] - 9.0], 0.0, 0.30, 9
    )
    np.testing.assert_array_equal(result.basis[:, :3], base)
    np.testing.assert_allclose(
        result.transform(fields[0])[:, :3], parent.transform(fields[0]), atol=1e-12
    )
    np.testing.assert_allclose(
        result.basis @ result.basis.T, shifted.basis @ shifted.basis.T, atol=1e-12
    )
    solo = ocean_basis(parent, fields, 0.0, 0.0, 9)
    residual = fields[0].reshape(55, -1) @ (np.eye(14) - base @ base.T)
    _, _, v = np.linalg.svd(residual, full_matrices=False)
    np.testing.assert_allclose(
        solo.basis @ solo.basis.T, base @ base.T + v[:6].T @ v[:6], atol=2e-12
    )


def test_zero_storm_months_and_single_scenario_year():
    import pandas as pd
    from paired_intervals import panel_weights
    from score_storms import distance, limits

    members = np.array(["a", "b"])
    years = np.array([2050, 2050])
    weights, nm, ny = panel_weights(members, years, np.array([1.8, 1.8]), "inside", 40, 29)
    assert (nm, ny) == (2, 1)
    frame = pd.DataFrame(dict(member=members, year=years, tc=[False, False], value=[0.0, 0.0]))
    grid = pd.MultiIndex.from_arrays([members, years])
    assert np.isnan(distance(frame, frame, "value", grid, weights)).all()
    frame.loc[0, ["tc", "value"]] = [True, 4.0]
    values = distance(frame, frame, "value", grid, weights)
    assert (
        values[0] == 0 and limits(values)["ci_high"] == 0 and 0 < limits(values)["valid_draws"] < 40
    )


def test_training_calibrated_fourier_recovers_known_spectrum():
    from tuned_fourier import harmonic_lags, spectrum_fit

    nd = [28, 30, 31]
    ops = [np.stack([kernels(n)[i] for n in nd]) for i in (0, 1)]
    amp = np.array([5.0, 7.0, 9.0])
    bg = np.array([1.0, 2.0, 3.0])
    lag = harmonic_lags(36.0, 1.5)
    raw = bg[:, None] + amp[:, None] * (ops[0] @ lag)
    within = amp[:, None] * (ops[1][:, :4] @ lag)
    fit = spectrum_fit(ops, bg, amp, raw, within)
    assert fit["training_loss"] < 1e-8 and fit["training_loss"] < fit["native_training_loss"]
    np.testing.assert_allclose([fit["period_days"], fit["power_exponent"]], [36.0, 1.5], rtol=0.005)


def test_signed_component_covariances_and_finite_month_centering():
    from leadlag import LAGS, finite_lags, ordered_filter

    A = np.array([[0.61, -0.23], [0.23, 0.61]])
    pos = np.stack([np.linalg.matrix_power(A, i) for i in range(37)])[:, None]
    taps = np.array([1.0, -0.07, -0.13])
    retention = 0.91
    actual = ordered_filter(pos, taps, retention, days=31)

    def original(h):
        return (pos[abs(h), 0] if h >= 0 else pos[-h, 0].T) * retention ** abs(h)

    direct = np.stack(
        [
            sum(x * y * original(h - i + j) for i, x in enumerate(taps) for j, y in enumerate(taps))
            for h in range(31)
        ]
    )
    np.testing.assert_allclose(actual[:, 0], direct, atol=1e-13)
    days = 28
    H = np.eye(days) - 1 / days
    block = np.stack(
        [[direct[i - j] if i >= j else direct[j - i].T for j in range(days)] for i in range(days)]
    )
    center = np.einsum("ti,ijab,sj->tsab", H, block, H)
    target = np.stack([np.mean([center[t + h, t] for t in range(days - h)], axis=0) for h in LAGS])
    np.testing.assert_allclose(finite_lags(actual, days)[0], target, atol=1e-12)
    assert abs(target[1, 0, 1] - target[1, 1, 0]) > 0.1


def test_ocean_residual_factors_match_direct_within_year_covariance():
    import pandas as pd
    from ocean_residual import residual_fit

    rng = np.random.default_rng(405)
    warming = np.repeat([-0.1, 1.0, 3.0], 9)
    maps = rng.normal(size=(len(warming), 1, 1, 7)) * (1 + warming[:, None, None, None] / 3)
    weights = np.linspace(0.7, 2.2, 7).reshape(1, 1, 7)
    block = SimpleNamespace(
        mean=np.zeros_like(weights),
        weights=weights,
        valid=np.ones_like(weights, dtype=bool),
        basis=np.eye(7)[:, :2],
    )
    meta = pd.DataFrame({"year": np.repeat([1900, 2020, 2070], 9), "g": warming})
    grids, fractions, fit = residual_fit(block, maps, meta, ranks=(2, 5))
    residual = maps.reshape(3, 9, 7).copy()
    residual[:, :, :2] = 0.0
    truth = residual.var(axis=1, ddof=1).mean(axis=0)
    np.testing.assert_allclose(grids["full"].ravel(), truth, atol=1e-12)
    np.testing.assert_allclose(grids["rank5"], grids["full"], atol=1e-12)
    assert 0 < fractions["rank2"] < fractions["rank5"] and fit["slope"] > 0


def test_covariance_response_preserves_positive_variance_and_field_cross_terms():
    from covariance_laws import partition, predictions

    rng = np.random.default_rng(78)
    scores = rng.normal(size=(45, 6))
    kernel = np.arange(12, dtype=float).reshape(6, 2) - 5
    physical = kernel @ kernel.T
    cov = np.cov(scores, rowvar=False)
    split = partition(cov, physical, [0, 2, 4])
    np.testing.assert_allclose(split.sum(), np.trace(np.cov(scores @ kernel, rowvar=False)))
    g = np.linspace(0, 3, 15)
    panels = np.exp(0.25 * g[:, None, None]) * split
    result = predictions(g, panels, [-2.0, -0.5, 1.5, 3.5, 5.0], 0.5)
    for value in result.values():
        assert np.linalg.eigvalsh(value).min() > 0
        np.testing.assert_allclose(value[0], value[1])
        np.testing.assert_allclose(value[-1], value[-2])
    np.testing.assert_allclose(result["common_loglinear"], result["field_loglinear"])


def test_shared_spectral_forecast_matches_full_two_component_linear_regression():
    from forecast_controls import linear_weights
    from tuned_fourier import harmonic_lags

    lags = harmonic_lags(36, 1.4)
    c = np.array([[2, -0.4], [-0.4, 0.8]])
    context = np.block([[lags[0] * c, lags[1] * c], [lags[1] * c, lags[0] * c]])
    for h in (1, 5, 10, 15):
        direct = np.linalg.solve(context, np.column_stack((lags[h] * c, lags[h + 1] * c)).T).T
        a, b = linear_weights(lags, h)
        np.testing.assert_allclose(
            direct, np.column_stack((a * np.eye(2), b * np.eye(2))), atol=1e-12
        )
