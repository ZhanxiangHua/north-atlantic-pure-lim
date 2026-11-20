"""Versioned paths, content fingerprints and atomic stage publication."""
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = "MPI-ESM1-2-LR"
COMPONENTS = ["u250", "v250", "u850", "v850"]
TABLES = {"tas": "Amon", "ta": "Amon", "hus": "Amon", "psl": "Amon",
          "tos": "Omon", "mlotst": "Omon", "thetao": "Omon", "ua": "day", "va": "day"}


def config(path=None):
    return json.loads(Path(path or os.getenv("TC_CONFIG", ROOT / "configs/mpi_september.json")).read_text())


def run_dir(cfg):
    path = ROOT / cfg["run"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hash(*names):
    return digest({name: sha(Path(__file__).with_name(name + ".py")) for name in names})


@contextlib.contextmanager
def atomic(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.stem + "-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        yield Path(tmp)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def save_json(path, data):
    with atomic(path) as tmp:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")


def save_npz(path, data):
    with atomic(path) as tmp:
        np.savez(tmp, **data)


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        yield


def fingerprint_rng(seed, *ids):
    keys = [int(digest(str(i))[:16], 16) for i in ids]
    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))


def runtime(start):
    return {"elapsed_s": round(time.monotonic() - start, 3), "job": os.getenv("PBS_JOBID"),
            "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "host": platform.node()}


def package_versions():
    names=("numpy","scipy","torch","xarray","netCDF4","cftime","h5py","h5netcdf","zarr","numcodecs",
           "pandas","pyarrow","dask","distributed","requests","matplotlib","global-land-mask")
    return {name:importlib.metadata.version(name) for name in names}


def environment():
    import torch
    return {"python": platform.python_version(), "executable": __import__("sys").executable,
            "packages":package_versions(),"threads":{v:os.getenv(v) for v in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS")},
            "blas": np.show_config(mode="dicts")["Build Dependencies"], "cuda": torch.cuda.is_available(),
            "archives": {f.name: sha(f) for f in ROOT.glob("*.zip")}, **runtime(time.monotonic())}
