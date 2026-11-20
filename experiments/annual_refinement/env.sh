# Shared environment for the Casper experiment jobs and local reproductions.
refinement_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TC_ROOT="$refinement_root"
if [[ -z "${TC_PYTHON:-}" ]]; then
    if [[ -x "$refinement_root/.venv/bin/python" ]]; then
        TC_PYTHON="$refinement_root/.venv/bin/python"
    else
        TC_PYTHON="${ANNUAL_PYTHON:-python}"
    fi
fi
export TC_PYTHON
export TC_REFINEMENT_WORK="${TC_REFINEMENT_WORK:-$refinement_root/runs/annual_refinement_20261120}"
export PYTHONPATH="$refinement_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${TC_BLAS_THREADS:-1}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MPLBACKEND=Agg
