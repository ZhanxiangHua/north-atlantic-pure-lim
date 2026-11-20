"""Execute a public tutorial with the installed interpreter and verify its research outputs."""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from tc_emulator.common import ROOT, atomic, save_json

TUTORIALS = {
    "annual": ("Annual_Pure_LIM_Tutorial.ipynb", "annual-research-panel", 2),
    "tc": ("FaIR_EOF_LIM_to_Tropical_Cyclones.ipynb", "storm-research-panel", 2),
}


def execute(root, which, output):
    filename, cell_id, expected = TUTORIALS[which]
    path = Path(root).resolve() / "notebooks" / filename
    book = nbformat.read(path, as_version=4)
    with tempfile.TemporaryDirectory(prefix="lim-tutorial-kernel-") as tmp:
        kernel = Path(tmp) / "kernels" / "lim-study"
        save_json(
            kernel / "kernel.json",
            dict(
                argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                display_name="North Atlantic LIM",
                language="python",
            ),
        )
        os.environ["JUPYTER_PATH"] = tmp + os.pathsep + os.environ.get("JUPYTER_PATH", "")
        os.environ["ANNUAL_LIM_ROOT"] = str(path.parent.parent)
        client = NotebookClient(
            book,
            kernel_name="lim-study",
            timeout=900,
            record_timing=False,
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.execute()
    code = [c for c in book.cells if c.cell_type == "code"]
    if [c.execution_count for c in code] != list(range(1, len(code) + 1)):
        raise ValueError("Incomplete notebook execution order")
    main = next(c for c in code if c.id == cell_id)
    images = sum("image/png" in o.get("data", {}) for o in main.outputs)
    if images != expected or not any("text/html" in o.get("data", {}) for o in main.outputs):
        raise ValueError("The research tutorial needs its quantitative table and all figures")
    book.metadata.kernelspec = dict(name="python3", display_name="Python 3", language="python")
    nbformat.validate(book)
    with atomic(output) as dest:
        nbformat.write(book, dest)
    print(
        f"Verified {filename}, {len(code)} code cells and {images} new benchmark figures",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--tutorial", choices=TUTORIALS, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    execute(a.root, a.tutorial, a.output)
