"""Train a two-parameter Fourier control against MPI window variances on training years only."""
import numpy as np
from records import WORK, record
from scipy.optimize import minimize
from weather import kernels, model_terms, year_table

from annual_lim.layout import read
from tc_emulator.common import lock, save_json


def harmonic_lags(period, exponent):
    """Daily-averaged covariance from 15 harmonics with unit variance at an instant."""
    h = np.arange(1.0, 16.0)
    powers = h ** (-exponent)
    powers /= powers.sum()
    hourly = np.exp(2j * np.pi * h[:, None] * np.arange(25) / (24 * period))
    daily = powers * np.abs(np.trapezoid(hourly, axis=1) / 24) ** 2
    return daily @ np.cos(2 * np.pi * h[:, None] * np.arange(31) / period)


def spectrum_fit(operators, bg, amplitude, observed_raw, observed_within):
    def error(x):
        h = harmonic_lags(*x)
        raw = bg[:, None] + amplitude[:, None] * (operators[0] @ h)
        within = amplitude[:, None] * (operators[1][:, :4] @ h)
        a = np.log(np.maximum(raw / observed_raw, 1e-12))
        b = np.log(np.maximum(within / observed_within, 1e-12))
        return 0.5 * (np.mean(a**2) + np.mean(b**2))

    starts = [
        (p, q)
        for p in (10.0, 15.0, 20.0, 28.0, 40.0, 60.0, 90.0, 120.0)
        for q in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    ]
    ranked = sorted((error(x), x) for x in starts)
    fits = [(float(loss), tuple(x)) for loss, x in ranked[:4]]
    for _, x in ranked[:4]:
        sol = minimize(
            error, x, method="L-BFGS-B", bounds=[(10.0, 120.0), (0.0, 5.0)], options={"ftol": 1e-12}
        )
        if np.isfinite(sol.fun):
            fits.append((float(sol.fun), tuple(map(float, sol.x))))
    best, (period, exponent) = min(fits)
    return dict(
        period_days=period,
        power_exponent=exponent,
        training_loss=best,
        native_training_loss=error((20.0, 3.0)),
    )


def calibrated(model, tag, cohort, month):
    path = WORK / "tuned_fourier" / tag / f"{month:02}.json"
    with lock(path.with_suffix(".lock")):
        if path.exists():
            result = read(path)
            if (result["model_id"], result["cohort"]) != (model.meta["id"], cohort):
                raise ValueError("Saved Fourier spectrum belongs to a different training fit")
        else:
            source = WORK / "wind_moments" / cohort / "train" / f"{month:02}.npz"
            with np.load(source, allow_pickle=False) as f:
                years, g, nd = year_table(f)
                truth = f["by_year_raw"]
                center = f["by_year_within"][:, :4]
                ncase = len(f["year"])
            ops = [np.stack([kernels(int(n))[i] for n in nd]) for i in (0, 1)]
            lag, bg, _ = model_terms(model, g)
            amp = lag[0, 0] * model.cal.amplitude(g) ** 2
            result = spectrum_fit(ops, bg[:, 0], amp, truth, center)
            result.update(
                model_id=model.meta["id"],
                cohort=cohort,
                month=month,
                cases=ncase,
                training_years=years.astype(int).tolist(),
            )
            save_json(path, result)
            record(
                "training_calibrated_fourier",
                tag=tag,
                month=month,
                cohort=cohort,
                period=result["period_days"],
                power=result["power_exponent"],
            )
    return harmonic_lags(result["period_days"], result["power_exponent"])
