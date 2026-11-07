"""Run notebooks headless, in order, then stop the Runpod pod.

Each notebook runs in a fresh kernel and is saved after every cell under
``runs/notebook-runs/<timestamp>/``, so a run can be followed by opening the
saved copy. The first failing notebook ends the run. Whatever happens, the
pod named by ``RUNPOD_POD_ID`` is stopped afterwards, unless ``--keep-pod``
is given or the variable is unset, as it is off Runpod.

Launch it so it outlives the terminal:

    nohup python scripts/run_notebooks.py \\
        notebooks/07_improve_executor.ipynb \\
        notebooks/08_open_ended_achievements.ipynb \\
        --set PRESET='"twenty_two_skill"' > notebook-run.log 2>&1 &

``--set NAME=VALUE`` replaces the value of a one-line top-level assignment
``NAME = ...`` in any notebook that has one, so ``--set EVALUATION_SEED=202``
runs notebook 08 on seed 202. VALUE is a Python literal.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STOP_ATTEMPTS = 3


def parse_overrides(pairs: list[str]) -> dict[str, str]:
    """Return ``NAME -> VALUE`` source for each ``NAME=VALUE`` pair."""

    overrides = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not name.isidentifier():
            raise ValueError(f"--set takes NAME=VALUE, not {pair!r}.")
        ast.literal_eval(value)  # Only literals, so the override reads plainly.
        overrides[name] = value
    return overrides


def apply_overrides(notebook: Any, overrides: dict[str, str]) -> set[str]:
    """Rewrite each overridden assignment in place and return the names used.

    Only a top-level assignment that is a whole statement on one line is
    rewritten, so a multi-line value is never cut in half.
    """

    used = set()
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        lines = cell.source.splitlines(keepends=True)
        for index, line in enumerate(lines):
            match = re.match(r"([A-Za-z_]\w*) = (.*?)(\s*#.*)?$", line.rstrip("\n"))
            if match is None or match.group(1) not in overrides:
                continue
            try:
                ast.parse(line)
            except SyntaxError:
                continue
            name = match.group(1)
            ending = "\n" if line.endswith("\n") else ""
            lines[index] = (
                f"{name} = {overrides[name]}  # Set by run_notebooks.py{ending}"
            )
            used.add(name)
        cell.source = "".join(lines)
    return used


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def ensure_dependencies() -> None:
    """Install the packages this runner needs into its own interpreter."""

    missing = [
        package
        for package in ("nbclient", "nbformat", "ipykernel")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        log(f"Installing {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing]
        )


def run_notebook(
    path: Path, notebook: Any, output_dir: Path, timeout: int | None
) -> None:
    """Execute one notebook, saving it after each cell."""

    import nbformat
    from nbclient import NotebookClient

    output_path = output_dir / path.name
    code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    started = time.perf_counter()
    executed = 0

    def on_cell_executed(cell: Any, cell_index: int, **_: Any) -> None:
        nonlocal executed
        if cell.cell_type != "code":
            return
        executed += 1
        nbformat.write(notebook, output_path)
        elapsed = time.perf_counter() - started
        progress = f"{executed} of {code_cells} code cells"
        log(f"{path.name}: cell {cell_index} done, {progress}, {elapsed:.0f}s")

    client = NotebookClient(
        notebook,
        timeout=timeout,
        # The kernel of this interpreter: the notebooks' own kernelspec names
        # whichever kernel the last machine to save them had.
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        on_cell_executed=on_cell_executed,
    )
    log(f"{path.name}: running {code_cells} code cells, saving to {output_path}")
    try:
        client.execute()
    finally:
        # Keep the outputs up to the failing cell, error included.
        nbformat.write(notebook, output_path)
    log(f"{path.name}: finished in {time.perf_counter() - started:.0f}s")


def stop_pod() -> None:
    """Stop this Runpod pod, retrying, and log rather than raise on failure."""

    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        log("RUNPOD_POD_ID is not set, so there is no pod to stop.")
        return
    if shutil.which("runpodctl") is None:
        log("runpodctl is not on PATH; stop the pod by hand.")
        return
    for attempt in range(1, STOP_ATTEMPTS + 1):
        log(f"Stopping pod {pod_id} (attempt {attempt} of {STOP_ATTEMPTS})")
        try:
            result = subprocess.run(
                ["runpodctl", "stop", "pod", pod_id],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            log((result.stdout + result.stderr).strip() or f"exit {result.returncode}")
            if result.returncode == 0:
                return
        except Exception:  # noqa: BLE001 - keep trying whatever went wrong.
            log(traceback.format_exc())
        time.sleep(10)
    log(f"Could not stop pod {pod_id}; stop it by hand.")


def _exit_on_signal(signum: int, _frame: Any) -> None:
    raise SystemExit(f"Received signal {signum}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run notebooks headless, then stop the Runpod pod."
    )
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Replace a one-line top-level assignment; VALUE is a Python literal.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where executed notebooks go; runs/notebook-runs/<timestamp> by default.",
    )
    parser.add_argument(
        "--cell-timeout",
        type=int,
        default=None,
        help="Seconds a cell may run; no limit by default.",
    )
    parser.add_argument(
        "--keep-pod", action="store_true", help="Leave the pod running afterwards."
    )
    args = parser.parse_args(argv)

    # A closed terminal must not end the run, and a kill must still stop the pod.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _exit_on_signal)

    status = 1
    try:
        overrides = parse_overrides(args.overrides)
        notebooks = [path.resolve() for path in args.notebooks]
        for path in notebooks:
            if not path.is_file():
                raise FileNotFoundError(path)
        output_dir = args.output_dir or (
            REPO_ROOT / "runs" / "notebook-runs" / f"{datetime.now():%Y%m%d-%H%M%S}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run.json").write_text(
            json.dumps(
                {
                    "notebooks": [str(path) for path in notebooks],
                    "overrides": overrides,
                    "argv": sys.argv,
                },
                indent=2,
            )
        )
        ensure_dependencies()
        import nbformat

        # Load and check every notebook first, so a mistyped override fails now
        # rather than after hours of the notebooks before it.
        loaded = [nbformat.read(path, as_version=4) for path in notebooks]
        used: set[str] = set()
        for notebook in loaded:
            used |= apply_overrides(notebook, overrides)
        if unused := sorted(set(overrides) - used):
            raise ValueError(f"No notebook assigns {unused} on one line.")
        for path, notebook in zip(notebooks, loaded, strict=True):
            run_notebook(path, notebook, output_dir, args.cell_timeout)
        log(f"All notebooks finished. Outputs: {output_dir}")
        status = 0
    except BaseException:
        log("Run failed:\n" + traceback.format_exc())
    finally:
        if args.keep_pod:
            log("--keep-pod given; leaving the pod running.")
        else:
            stop_pod()
    return status


if __name__ == "__main__":
    sys.exit(main())
