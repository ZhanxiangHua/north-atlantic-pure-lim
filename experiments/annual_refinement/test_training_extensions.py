"""Numerical invariants for the post-freeze training and development trials."""
import numpy as np
from deep_mixed_layer import exceedance, quantile, tail_map
from ocean_fronts import edges, front_basis, gradient_percentile, score
from scipy.linalg import null_space
from seasonal_covariance_cv import laws, sample_variance
from wind_visible_gain import linked_factor

from tc_emulator.representation import EOF


def test_calendar_slopes_preserve_levels_and_recover_log_linear_signal():
    g = np.linspace(-0.1, 3.5, 20)
    slopes = np.linspace(-0.1, 0.18, 24).reshape(12, 2)
    variance = np.exp(slopes[:, None, :] * g[None, :, None])
    prediction, fit = laws(variance, g, g)
    np.testing.assert_allclose(fit[1], slopes, atol=1e-14)
    np.testing.assert_allclose(prediction[1], variance, atol=1e-14)
    means = np.broadcast_to(variance.mean(axis=1), prediction.mean(axis=2).shape)
    np.testing.assert_allclose(prediction.mean(axis=2), means, atol=1e-14)
    np.testing.assert_allclose(fit[-1], np.broadcast_to(slopes.mean(axis=0), slopes.shape))


def test_member_gram_recovers_physical_spread_after_subset_centering():
    rng = np.random.default_rng(1108)
    wind = rng.normal(size=(12, 5, 2, 15, 6))
    basis = rng.normal(size=(6, 4))
    G = basis @ basis.T
    gram = wind @ G @ wind.swapaxes(-1, -2)
    members = [0, 3, 5, 9, 13]
    w = wind[:, :, :, members]
    d = w - w.mean(axis=3, keepdims=True)
    np.testing.assert_allclose(
        sample_variance(gram, members),
        np.sum((d @ G) * d, axis=(-2, -1)) / (len(members) - 1),
        atol=1e-12,
    )


def test_front_metric_preserves_coarse_projection_and_constant_field_gradient():
    rng = np.random.default_rng(307)
    lat = np.arange(5.0) * 2
    lon = np.arange(7.0) * 3
    wet = np.ones((5, 7), bool)
    wet[2, 2] = False
    adjacent = edges(lat, lon)
    pct = gradient_percentile(np.arange(35.0).reshape(5, 7) ** 2, wet, adjacent)
    assert not pct[2, 2] and np.any(pct >= 0.8)
    errors = np.full((3, 5, 7), 7.0)
    outcome = score(errors, wet, pct, np.ones_like(wet, dtype=float), adjacent)
    np.testing.assert_allclose(outcome["front_rmse"], 7)
    np.testing.assert_allclose(outcome["gradient_rmse_per_100km"], 0)
    valid = wet[None]
    u = np.linalg.qr(rng.normal(size=(int(wet.sum()), 3)))[0]
    parent = EOF(np.zeros(valid.shape), np.ones(valid.shape), valid, u, np.ones(3), 10.0)
    maps = rng.normal(size=(32, 1, 5, 7))
    weighted = front_basis(parent, maps, pct[None], 8, 0.5)
    np.testing.assert_allclose(weighted.transform(maps)[:, :3], parent.transform(maps), atol=1e-13)
    np.testing.assert_allclose(weighted.basis.T @ weighted.basis, np.eye(8), atol=1e-13)


def test_deep_quantiles_match_probabilities_and_respect_censoring():
    z = np.array([-2.0, 0.0, 1.7, 4.0])
    loc = np.log([10.0, 100.0, 2000.0])
    sd = np.array([0.3, 0.5, 0.9])
    for alpha in (1.0, 0.75, 0.5):
        np.testing.assert_allclose(tail_map(tail_map(z, alpha), 1 / alpha), z)
        for tau in (0.9, 0.95, 0.99, 0.995):
            cut = quantile(loc, sd, tau, alpha, np.inf)
            np.testing.assert_allclose(exceedance(loc, sd, cut, alpha, np.inf), 1 - tau, atol=1e-14)
        np.testing.assert_array_equal(exceedance(loc, sd, 2000, alpha, 1000), [0, 0, 0])
        np.testing.assert_array_equal(
            exceedance(np.log([10.0, 30.0]), np.zeros(2), 20, alpha, np.inf), [0, 1]
        )


def test_gain_preserves_wind_null_vectors_and_scales_full_transmitted_covariance():
    rng = np.random.default_rng(431)
    F = rng.normal(size=(11, 11))
    K = rng.normal(size=(11, 3))
    null = null_space((F.T @ K).T)
    old = F.T @ K
    for gain in (0.5, 1.0, 1.7):
        changed = linked_factor(F, K, gain)
        seen = changed.T @ K
        np.testing.assert_allclose(seen.T @ seen, gain * old.T @ old, atol=1e-12)
        np.testing.assert_allclose(changed @ null, F @ null, atol=1e-12)


def test_projection_preserves_endpoints_of_two_day_wind_memory():
    from scipy.linalg import solve_discrete_lyapunov
    from order_reduction import matched_dynamics
    from partial_wind_memory import past_directions, project_state
    from tc_emulator.wind_lim import Dynamics

    k = 3
    first = np.array([[0.62, -0.08, 0.04], [0.11, 0.45, 0.02], [0.03, 0.01, 0.55]])
    second = np.diag([-0.14, 0.16, -0.19])
    a = np.block([[first, second], [np.eye(k), np.zeros((k, k))]])
    q = np.diag([1.2, 0.7, 0.4, 0.0, 0.0, 0.0])
    base = Dynamics(a, q, solve_discrete_lyapunov(a, q), 2, k)
    short, _ = matched_dynamics(base)
    back, axes, correlations = past_directions(base)
    np.testing.assert_array_less(np.diff(correlations), 0.)
    empty, full = [project_state(base, back, axes, i) for i in (0, k)]
    np.testing.assert_allclose(empty.a, short.a, atol=1e-14)
    for lead in (0, 1, 5, 15):
        original = (np.linalg.matrix_power(base.a, lead) @ base.sigma)[:k, :k]
        transformed = (np.linalg.matrix_power(full.a, lead) @ full.sigma)[:k, :k]
        np.testing.assert_allclose(transformed, original, atol=1e-13)


def test_bathymetry_regions_exclude_preprocessing_halo():
    from bathymetric_mld import marine_masks

    lat = np.array([-1., 30., 51., 61., 18.])
    lon = np.array([280., 300., 320., 320., 259.])
    depth = np.array([30., 90., 3200., 70., 40.])
    mask = marine_masks(lat, lon, depth)
    np.testing.assert_array_equal(mask["whole_basin"], [False, True, True, False, False])
    np.testing.assert_array_equal(mask["shelf_at_most_500m"], [False, True, False, False, False])
    np.testing.assert_array_equal(mask["subpolar_Atlantic"], [False, False, True, False, False])


def test_bottom_cap_reads_month_and_legacy_annual_metadata(tmp_path):
    import xarray as xr
    from compact_deploy import cap_mld

    lat, lon = [10.0], [280., 282.]
    path = tmp_path / "bottom.npz"
    np.savez(path, lat=lat, lon=lon, depth_m=np.array([[80., 300.]]))
    monthly = xr.Dataset({"mld": (("lat", "lon"), [[110., 245.]], {"units": "m"})},
                         coords={"lat": lat, "lon": lon})
    clipped = cap_mld(monthly, path)
    np.testing.assert_array_equal(clipped.mld.values, [[80., 245.]])
    annual = monthly.expand_dims(calendar_month=np.arange(1,13)).copy()
    annual.mld.attrs.clear()
    annual.attrs["weather_sampler"] = "pure_lim"
    annual["model_id"] = ("calendar_month", [str(i) for i in range(12)])
    bounded = cap_mld(annual, path)
    assert bounded.mld.attrs["units"] == "m" and bounded.attrs["mld_ceiling_cells"] == 12
    np.testing.assert_array_equal(bounded.mld[:,0,1], np.full(12, 245.))
