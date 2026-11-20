"""Candidate EOF coordinates, conditional circulation and positive-definite background covariance."""
from dataclasses import dataclass

import numpy as np
from features import MAX_OCEAN, load_block
from ocean import truncate
from records import WORK
from scipy.linalg import solve_triangular

from annual_lim.layout import DEFAULT_WORK, read
from annual_lim.model import MonthlyModel
from tc_emulator.background import Background, Circulation, polynomial
from tc_emulator.representation import Representation
from tc_emulator.warming_circulation import transfer_matrix


@dataclass(frozen=True)
class Recipe:
    sst: int = 18
    ocean: int = 18
    ocean_basis: str = "published_ocean"
    sst_basis: str = "published_sst"
    wind: int = 120
    wind_basis: str = "published_wind"
    refit_other: bool = False
    ridge: float = 0.1
    tail_penalty: float = 1.0
    warming_penalty: float = 1.0
    regression_rank: int = 0
    mean_curve: float = 1.0
    covariance: str = "active"
    covariance_slope_shrink: float = 1.0
    covariance_diag_shrink: float = 0.05
    split_mld: float = 0.5
    lim_ridge: float = 0.0
    ocean_marginals: bool = False
    coarse_parent: str = ""
    tail_gain: float = 0.0


class Bank:
    def __init__(self, cohort, month):
        self.root = WORK / "features" / cohort / f"{month:02}"
        self.info = read(self.root / "complete.json")
        self.cohort, self.month = cohort, month
        self.geo = MonthlyModel(DEFAULT_WORK / "weights", month).rep
        with np.load(self.root / "scores.npz", allow_pickle=False) as data:
            self.scores = {v: data[v] for v in data.files}
        if (self.root / "scores_rich.npz").is_file():
            with np.load(self.root / "scores_rich.npz", allow_pickle=False) as data:
                self.scores.update({v: data[v] for v in data.files})
        if (self.root / "scores_nested.npz").is_file():
            with np.load(self.root / "scores_nested.npz", allow_pickle=False) as data:
                self.scores.update({v: data[v] for v in data.files})
        if (self.root / "scores_calendar.npz").is_file():
            with np.load(self.root / "scores_calendar.npz", allow_pickle=False) as data:
                self.scores.update({v: data[v] for v in data.files})
        self.bases = {}
        self.wind_metrics = {}
        self.rotations = {}

    def rotation(self, key):
        if key not in self.rotations:
            with np.load(self.root / "rotations.npz") as f:
                self.rotations[key] = f[key]
        return self.rotations[key]

    def block(self, key):
        if key not in self.bases:
            if key.startswith("mix_"):
                parent = self.block("wind")
                rot = self.rotation(key)
                from dataclasses import replace

                self.bases[key] = replace(
                    parent, basis=parent.basis @ rot, variance=parent.variance @ (rot * rot)
                )
            else:
                self.bases[key] = load_block(self.root / (key + ".npz"))
        return self.bases[key]

    def layout(self, r):
        if r.ocean_basis.startswith("ocean_split"):
            n = round(r.split_mld * r.ocean)
            ocean = np.r_[np.arange(n), MAX_OCEAN + np.arange(r.ocean - n)]
        else:
            ocean = np.arange(r.ocean)
        other = "new_" if r.refit_other and self.cohort != "original" else "published_"
        return [
            (r.sst_basis, np.arange(r.sst)),
            (r.ocean_basis, ocean),
            (other + "psl", np.arange(14)),
            (other + "profiles", np.arange(28)),
        ]

    def z(self, r, role):
        return np.concatenate(
            [self.scores[role + "_" + name][:, idx] for name, idx in self.layout(r)], axis=1
        )

    def representation(self, r):
        blocks = {
            name: truncate(self.block(key), idx)
            for name, (key, idx) in zip(("sst", "ocean", "psl", "profiles"), self.layout(r))
        }
        blocks["wind"] = truncate(self.block(r.wind_basis), slice(r.wind))
        return Representation(blocks, self.geo.lat, self.geo.lon, self.geo.plev, True)

    def wind_monthly(self, r, role):
        if r.wind_basis.startswith("mix_"):
            return self.scores[role + "_wind_monthly"] @ self.rotation(r.wind_basis)[:, : r.wind]
        return self.scores[role + "_" + r.wind_basis + "_monthly"][:, : r.wind]

    def daily(self, r, role):
        if r.wind_basis.startswith("mix_"):
            pcs = np.load(self.root / (role + "_wind_daily.npy"), mmap_mode="r")
            return pcs @ self.rotation(r.wind_basis)[:, : r.wind]
        return np.load(self.root / (role + "_" + r.wind_basis + "_daily.npy"), mmap_mode="r")[
            :, :, : r.wind
        ]

    def G(self, r):
        from annual_lim.statistics import metrics

        key = r.wind_basis, r.wind
        if key not in self.wind_metrics:
            self.wind_metrics[key] = metrics(self.representation(r))[2]
        return self.wind_metrics[key]


def fit_conditional(r, z, w, year, g, metric, extra=None):
    gn = (g - g.mean()) / g.std()
    design = polynomial(gn)
    beta = np.linalg.lstsq(design, z, rcond=None)[0]
    linear = np.linalg.lstsq(design[:, :2], z, rcond=None)[0]
    beta = r.mean_curve * beta + (1 - r.mean_curve) * np.vstack([linear, np.zeros(z.shape[1])])
    if extra is not None:
        eo, eg, ey = extra
        no = r.sst + r.ocean
        combined = np.vstack((z[:, :no], eo))
        full_year = np.r_[year, ey]
        h = polynomial((np.r_[g, eg] - g.mean()) / g.std())
        q = np.linalg.lstsq(h, combined, rcond=None)[0]
        linear[:, :no] = np.linalg.lstsq(h[:, :2], combined, rcond=None)[0]
        beta[:, :no] = r.mean_curve * q + (1 - r.mean_curve) * np.vstack(
            (linear[:, :no], np.zeros(no))
        )
    delta = z - design @ beta
    sd = np.maximum(delta.std(axis=0), 1e-8)
    H = np.column_stack([np.ones(len(g)), gn, delta / sd])
    penalties = np.ones(H.shape[1])
    offset = 2
    for width, leading in [(r.sst, 18), (r.ocean, 18), (14, 14), (28, 28)]:
        if offset == 2 + r.sst and r.ocean_basis.startswith("ocean_split"):
            nm = round(r.split_mld * width)
            penalties[offset + min(9, nm) : offset + nm] = r.tail_penalty
            penalties[offset + nm + min(9, width - nm) : offset + width] = r.tail_penalty
        else:
            penalties[offset + min(leading, width) : offset + width] = r.tail_penalty
        offset += width
    penalties[0] = 0
    penalties[1] = r.warming_penalty
    coef = np.linalg.solve(H.T @ H + len(H) * r.ridge * np.diag(penalties), H.T @ w)
    if 0 < r.regression_rank < coef.shape[1]:
        fit = H[:, 2:] @ coef[2:]
        eig, vec = np.linalg.eigh(fit.T @ fit)
        axes = vec[:, -r.regression_rank :]
        coef[2:] = (coef[2:] @ axes) @ axes.T
    mean = Circulation(coef, sd)
    yy, first, counts = np.unique(year, return_index=True, return_counts=True)
    covs = np.stack([np.cov(z[year == y], rowvar=False) for y in yy])
    weights = (counts - 1) / (counts - 1).sum()
    anchors = g[first]
    center = float(weights @ anchors)
    dx = anchors - center
    q = z.shape[1]
    slopes = np.zeros(q)
    scales = np.ones(q)
    if "block" in r.covariance:
        annual = covs.diagonal(axis1=1, axis2=2)
        relative = annual / np.maximum(np.average(annual, axis=0, weights=weights), 1e-12)
        start = 0
        for n in (r.sst, r.ocean, 14, 28):
            block = slice(start, start + n)
            v = np.mean(relative[:, block], axis=1)
            slope = (
                np.clip(((weights * dx) @ np.log(v)) / (weights @ (dx * dx)), -0.75, 0.75)
                * r.covariance_slope_shrink
            )
            slopes[block] = slope
            scales[block] = 1 / (weights @ np.exp(slope * dx))
            start += n
        d = np.sqrt(scales[None] * np.exp(dx[:, None] * slopes))
        covs = covs / (d[:, :, None] * d[:, None, :])
    pooled = np.average(covs, axis=0, weights=weights)
    pooled = (1 - r.covariance_diag_shrink) * pooled + r.covariance_diag_shrink * np.diag(
        pooled.diagonal()
    )
    if extra is not None:
        if "block" in r.covariance:
            raise ValueError("The ocean-marginal experiment uses pooled or active subspace scaling")
        # Replace the paired ocean marginal by a better-sampled one. A congruence
        # preserves positive definiteness and the fitted marginal of the remaining fields.
        yo, nc = np.unique(full_year, return_counts=True)
        observed = np.average(
            [np.cov(combined[full_year == y], rowvar=False) for y in yo], axis=0, weights=nc - 1
        )
        observed = (1 - r.covariance_diag_shrink) * observed + r.covariance_diag_shrink * np.diag(
            observed.diagonal()
        )
        old = np.linalg.cholesky(pooled[:no, :no])
        new = np.linalg.cholesky(observed)
        transform = solve_triangular(old.T, new.T, lower=False).T
        cross = transform @ pooled[:no, no:]
        pooled[:no, :no] = observed
        pooled[:no, no:] = cross
        pooled[no:, :no] = cross.T
    L = np.linalg.cholesky(pooled)
    bg = Background(float(g.mean()), float(g.std()), beta, linear, L, np.zeros_like(L), pooled)
    directions = np.empty((q, 0))
    amp, slope = 1.0, 0.0
    if "active" in r.covariance:
        loading = L.T @ transfer_matrix(mean)
        e, v = np.linalg.eigh(loading @ metric @ loading.T)
        e = np.maximum(e[::-1], 0)
        rank = 1 + int(np.searchsorted(np.cumsum(e) / e.sum(), 0.8))
        directions = v[:, ::-1][:, :rank]
        projected = solve_triangular(L.T, directions, lower=False)
        energy = np.einsum("ir,yij,jr,r->y", projected, covs, projected, e[:rank], optimize=True)
        slope = float(((weights * dx) @ np.log(energy)) / (weights @ (dx * dx)))
        amp = float((weights @ energy) / (e[:rank].sum() * (weights @ np.exp(slope * dx))))
    return bg, mean, directions, amp, slope, slopes, scales


def build(bank, r, rows=None):
    from types import SimpleNamespace

    all_z = bank.z(r, "train")
    full = rows is None
    rows = np.arange(len(all_z)) if rows is None else rows
    g = bank.scores["train_g"][rows]
    year = bank.scores["train_year"][rows]
    model = MonthlyModel.__new__(MonthlyModel)
    model.rep = bank.representation(r)
    extra = None
    if r.ocean_marginals:
        ez = np.concatenate(
            [bank.scores["extra_" + name][:, idx] for name, idx in bank.layout(r)[:2]], axis=1
        )
        eg, ey = bank.scores["extra_g"], bank.scores["extra_year"]
        keep = (
            np.ones(len(eg), bool)
            if full
            else (
                np.isin(bank.scores["extra_member"], bank.scores["train_member"][rows])
                & (eg >= g.min())
                & (eg <= g.max())
            )
        )
        extra = ez[keep], eg[keep], ey[keep]
    w = bank.wind_monthly(r, "train")[rows]
    if r.coarse_parent:
        from nested import attach

        if r.coarse_parent == "release":
            parent = MonthlyModel(DEFAULT_WORK / "weights", bank.month)
        else:
            parent = build(bank, Recipe(ridge=r.ridge, mean_curve=r.mean_curve), rows)
        model.bg, model.mean, model.directions = attach(
            parent,
            all_z[rows],
            w,
            g,
            year,
            r.sst,
            r.ocean,
            r.tail_gain,
            r.tail_penalty,
            r.mean_curve,
        )
        model.bg_amp, model.bg_slope = parent.bg_amp, parent.bg_slope
        model.block_slope = np.zeros(all_z.shape[1])
        model.block_scale = np.ones(all_z.shape[1])
    else:
        (
            model.bg,
            model.mean,
            model.directions,
            model.bg_amp,
            model.bg_slope,
            model.block_slope,
            model.block_scale,
        ) = fit_conditional(r, all_z[rows], w, year, g, bank.G(r)[0], extra)
    lo, hi = float(g.min()), float(g.max())
    model.cal = SimpleNamespace(
        g_min=lo,
        g_max=hi,
        extension=1.25,
        temperature=lambda warming: np.clip(warming, lo - 1.25, hi + 1.25),
    )
    return model
