import importlib.util
import signal
from pathlib import Path

import nbformat
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_notebooks.py"
_spec = importlib.util.spec_from_file_location("run_notebooks", SCRIPT)
assert _spec is not None and _spec.loader is not None
run_notebooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_notebooks)


@pytest.fixture
def pod(tmp_path, monkeypatch):
    """A fake runpodctl on PATH that records its arguments."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "runpodctl-calls"
    fake = bin_dir / "runpodctl"
    fake.write_text(f'#!/bin/sh\necho "$@" >> {calls}\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path('/usr/bin')}:{Path('/bin')}")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGHUP, signal.SIGTERM)}
    yield calls
    for sig, handler in handlers.items():
        signal.signal(sig, handler)


def _notebook(path: Path, *sources: str) -> Path:
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(source) for source in sources]
    )
    nbformat.write(notebook, path)
    return path


def _stops(calls: Path) -> list[str]:
    return calls.read_text().splitlines() if calls.exists() else []


def test_overrides_replace_only_one_line_assignments() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("SEED = 51  # The development seed.\nN = 2"),
            nbformat.v4.new_code_cell("PATHS = (\n    1,\n)\n"),
        ]
    )
    used = run_notebooks.apply_overrides(notebook, {"SEED": "202", "PATHS": "()"})
    assert used == {"SEED"}
    assert notebook.cells[0].source.startswith("SEED = 202  # Set by")
    assert notebook.cells[0].source.endswith("\nN = 2")
    assert notebook.cells[1].source == "PATHS = (\n    1,\n)\n"
    with pytest.raises(ValueError):
        run_notebooks.parse_overrides(["SEED=os.getcwd()"])
    with pytest.raises(ValueError):
        run_notebooks.parse_overrides(["SEED"])


def test_a_finished_run_saves_the_outputs_and_stops_the_pod(
    tmp_path, pod, capsys
) -> None:
    first = _notebook(tmp_path / "first.ipynb", "SEED = 51", "print(SEED * 2)")
    second = _notebook(tmp_path / "second.ipynb", "print('second')")
    output_dir = tmp_path / "out"

    status = run_notebooks.main(
        [str(first), str(second), "--set", "SEED=202", "--output-dir", str(output_dir)]
    )

    assert status == 0
    executed = nbformat.read(output_dir / "first.ipynb", as_version=4)
    assert executed.cells[1].outputs[0]["text"] == "404\n"
    # Printed lines reach the log while the run goes, tagged with their cell.
    assert "first.ipynb [1] 404" in capsys.readouterr().out
    assert (output_dir / "second.ipynb").is_file()
    assert _stops(pod) == ["stop pod pod-123"]


def test_a_failed_run_keeps_its_outputs_and_still_stops_the_pod(tmp_path, pod) -> None:
    failing = _notebook(tmp_path / "failing.ipynb", "print('ran')", "1 / 0")
    after = _notebook(tmp_path / "after.ipynb", "print('after')")
    output_dir = tmp_path / "out"

    status = run_notebooks.main(
        [str(failing), str(after), "--output-dir", str(output_dir)]
    )

    assert status == 1
    executed = nbformat.read(output_dir / "failing.ipynb", as_version=4)
    assert executed.cells[1].outputs[0]["ename"] == "ZeroDivisionError"
    # The first failure ends the run.
    assert not (output_dir / "after.ipynb").exists()
    assert _stops(pod) == ["stop pod pod-123"]


def test_an_unknown_override_fails_before_running_and_stops_the_pod(
    tmp_path, pod
) -> None:
    notebook = _notebook(tmp_path / "notebook.ipynb", "SEED = 51")
    output_dir = tmp_path / "out"

    status = run_notebooks.main(
        [str(notebook), "--set", "SEDE=202", "--output-dir", str(output_dir)]
    )

    assert status == 1
    assert not (output_dir / "notebook.ipynb").exists()
    assert _stops(pod) == ["stop pod pod-123"]


def test_keep_pod_leaves_the_pod_running(tmp_path, pod) -> None:
    notebook = _notebook(tmp_path / "notebook.ipynb", "1 / 0")
    status = run_notebooks.main(
        [str(notebook), "--keep-pod", "--output-dir", str(tmp_path / "out")]
    )
    assert status == 1
    assert _stops(pod) == []
