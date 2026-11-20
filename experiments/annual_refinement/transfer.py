"""Test regression shrinkage and reduced rank with the full physical steering wind as the target."""
import argparse
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from backbone import Bank, Recipe
from joint import PhysicalValidation
from records import WORK, record

from tc_emulator.background import polynomial
from tc_emulator.common import atomic, save_json
from tc_emulator.diagnostics import steering


def efficient_score(bank, val, r):
    eof = bank.block(r.wind_basis)
    k = r.wind
    # Reconstruct an exact physical basis in float64 without rounding single-mode fields to float32.
    J = np.zeros((k, eof.valid.size))
    J[:, eof.valid.ravel()] = eof.basis[:, :k].T / eof.weights[eof.valid]
    J = steering(J.reshape(k, *eof.valid.shape)) * np.sqrt(val.area / 2)
    J = J.reshape(k, -1)
    zero = (steering(eof.mean) * np.sqrt(val.area / 2)).ravel()
    vb = val.wind - zero
    G = J @ J.T
    c = vb @ J.T
    y2 = np.einsum("ij,ij->i", vb, vb)

    def mse(pred):
        return np.einsum("ij,ij->i", pred @ G, pred) - 2 * np.einsum("ij,ij->i", c, pred) + y2

    return mse


def run(month, cohort):
    out = WORK / "transfer" / cohort / f"{month:02}.parquet"
    if out.exists():
        return
    start = time.monotonic()
    record("transfer_start", month=month, cohort=cohort)
    bank = Bank(cohort, month)
    val = PhysicalValidation(bank)
    y = bank.wind_monthly(Recipe(), "train")
    g = bank.scores["train_g"]
    gv = bank.scores["val_g"]
    h = polynomial((g - g.mean()) / g.std())
    hv = polynomial((gv - g.mean()) / g.std())
    mse = efficient_score(bank, val, Recipe())
    rows = []
    recipes = {}
    for s, o, layout in (
        (18, 18, "published_ocean"),
        (32, 48, "ocean_d070"),
        (48, 72, "ocean_d070"),
        (72, 96, "ocean_d070"),
        (96, 128, "ocean_d070"),
        (72, 96, "ocean"),
        (72, 96, "ocean_d070_forced"),
    ):
        r = Recipe(
            sst=s, ocean=o, sst_basis="published_sst" if s == 18 else "sst", ocean_basis=layout
        )
        z, zv = bank.z(r, "train"), bank.z(r, "val")
        for curve in (0.5, 1.0):
            quad = np.linalg.lstsq(h, z, rcond=None)[0]
            lin = np.vstack((np.linalg.lstsq(h[:, :2], z, rcond=None)[0], np.zeros(z.shape[1])))
            beta = curve * quad + (1 - curve) * lin
            d = z - h @ beta
            sd = np.maximum(d.std(axis=0), 1e-8)
            H = np.column_stack((h[:, :2], d / sd))
            Hv = np.column_stack((hv[:, :2], (zv - hv @ beta) / sd))
            gram = H.T @ H
            rhs = H.T @ y
            for ridge in (0.03, 0.1, 0.3, 1.0, 3.0):
                for tail in (1.0,) if s == 18 else (1.0, 5.0, 25.0, 125.0):
                    diag = np.ones(H.shape[1])
                    diag[0] = 0.0
                    diag[2 + 18 : 2 + s] = tail
                    diag[2 + s + 18 : 2 + s + o] = tail
                    for warming in (0.0, 1.0):
                        diag[1] = warming
                        coef = np.linalg.solve(gram + len(g) * ridge * np.diag(diag), rhs)
                        proj = H[:, 2:] @ coef[2:]
                        _, axes = np.linalg.eigh(proj.T @ proj)
                        for rr in (0, 8, 24, 48):
                            slope = (
                                coef[2:]
                                if rr == 0
                                else (coef[2:] @ axes[:, -rr:]) @ axes[:, -rr:].T
                            )
                            yp = Hv[:, :2] @ coef[:2] + Hv[:, 2:] @ slope
                            tag = f"s{s}_o{o}_{layout}_c{curve:g}_r{ridge:g}_p{tail:g}_w{warming:g}_k{rr}"
                            recipes[tag] = asdict(
                                replace(
                                    r,
                                    mean_curve=curve,
                                    ridge=ridge,
                                    tail_penalty=tail,
                                    warming_penalty=warming,
                                    regression_rank=rr,
                                )
                            )
                            frame = val.meta[["member", "year", "g"]].copy()
                            rows.append(frame.assign(recipe=tag, conditional_mse=mse(yp)))
    with atomic(out) as tmp:
        pd.concat(rows, ignore_index=True).to_parquet(tmp, index=False, compression="zstd")
    save_json(out.with_suffix(".json"), dict(month=month, cohort=cohort, recipes=recipes))
    record(
        "transfer_done",
        month=month,
        cohort=cohort,
        recipes=len(recipes),
        seconds=round(time.monotonic() - start, 1),
    )
    print(month, cohort, len(recipes), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default="original")
    p.add_argument("--month", type=int, required=True)
    a = p.parse_args()
    run(a.month, a.cohort)
