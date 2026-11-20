"""Evaluate temporal skill relative to matched Fourier with paired member and year uncertainty."""
import numpy as np
import pandas as pd
from ocean import GROUP_LIMITS
from paired_intervals import panel_weights, record_estimate

WINDOWS = ("1", "5", "10", "15", "month")


def accumulate(data, views, year_index, history, draws):
    """Every bootstrap draw uses the same members and years in all twelve calendar months."""
    members, years, g = data["member"], data["year"], data["g"]
    mean = data["mu"].astype(float)
    gram = np.einsum("iwf,jwf->wij", mean, mean, optimize=True)
    low, high = GROUP_LIMITS
    bins = np.where(g < low, "cold", np.where(g > high, "warm", "inside"))
    for i, group in enumerate(("all", "cold", "inside", "warm")):
        if group != "all" and not np.any(bins == group):
            continue
        w, nm, ny = panel_weights(members, years, g, group, draws, 20261120 + i)
        raw = w @ data["second"]
        for j in range(raw.shape[1]):
            raw[:, j] -= np.einsum("bi,bi->b", w @ gram[j], w)
        center = w @ data["within"][:, :4]
        g_variance = np.maximum(w @ (g * g) - (w @ g) ** 2, 0.0)
        for name, (total, within, slope) in views.items():
            for target, pred, truth in (
                ("total", w @ total[year_index] + g_variance[:, None] * slope, raw),
                ("within", w @ within[year_index, :4], center),
            ):
                ok = truth > 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(pred))
                ratio = np.divide(pred, truth, out=np.full_like(pred, np.nan), where=ok)
                error = np.abs(np.log(ratio))
                for k in range(error.shape[1]):
                    history.setdefault((group, target, WINDOWS[k], name, nm, ny), []).append(
                        error[:, k]
                    )


def summarize(history, direct):
    rows = []
    for (group, target, window, driver, nm, ny), values in history.items():
        if driver == "candidate_LIM":
            continue
        if len(values) != 12:
            raise ValueError(f"Incomplete monthly comparison for {driver}/{target}/{window}")
        base = np.mean(values, axis=0)
        new = np.mean(history[group, target, window, "candidate_LIM", nm, ny], axis=0)
        match = direct[
            (direct.group == group) & (direct.target == target) & (direct.window == window)
        ]
        ref = match.groupby("model").abs_log_error.mean()
        np.testing.assert_allclose(
            [base[0], new[0]], [ref[driver], ref["candidate_LIM"]], atol=2e-8
        )
        record_estimate(
            rows, group, "variance_log_error", (target, window), base, new, nm, ny, "raw"
        )
        rows[-1]["comparator"] = driver
    return pd.DataFrame(rows)
