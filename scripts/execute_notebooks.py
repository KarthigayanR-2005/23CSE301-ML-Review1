"""Execute every notebook top-to-bottom, in place, with outputs stored.

    python scripts/execute_notebooks.py                 # all three
    python scripts/execute_notebooks.py regression      # one

Deliverable D1 requires notebooks that are fully run with outputs visible and
no execution errors. This script fails loudly if any cell raises - it never
swallows an error to make the notebook look clean.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from src import config

ORDER = ["regression", "classification", "clustering"]


def run_one(name: str, timeout: int = 3600) -> tuple[bool, str]:
    path = config.NOTEBOOKS_DIR / f"{name}.ipynb"
    if not path.exists():
        return False, f"missing {path} - run scripts/build_notebooks.py first"
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(config.NOTEBOOKS_DIR)}},
                            allow_errors=False)
    t0 = time.perf_counter()
    try:
        client.execute()
    except CellExecutionError as e:
        nbformat.write(nb, path)      # keep the partial output for debugging
        return False, f"cell error: {str(e)[:600]}"
    finally:
        nbformat.write(nb, path)
    n_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
    return True, (f"{len(nb.cells)} cells, {n_out} code cells with output, "
                  f"{time.perf_counter() - t0:.1f}s")


def main() -> int:
    targets = sys.argv[1:] or ORDER
    failures = []
    for name in targets:
        print(f"\n=== executing {name}.ipynb ===")
        ok, msg = run_one(name)
        print(("  OK   " if ok else "  FAIL ") + msg)
        if not ok:
            failures.append(name)
    print("\n" + ("all notebooks executed cleanly"
                  if not failures else f"FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
