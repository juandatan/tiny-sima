"""Static checks that the notebooks will run on a remote GPU kernel.

The notebooks load a VLM and a trained checkpoint, so they cannot be executed
here. What can be verified without a GPU is everything that fails at import
time: that each cell parses, that every name imported from ``tiny_sima``
actually exists, and that no cell uses a name the notebook never defines.
These are the failures that waste a remote session.
"""

import ast
import builtins
import importlib
import json
from pathlib import Path

import nbformat
import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOK_PATHS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

# Provided by the kernel or by the installation cell rather than by an import.
_IMPLICIT_NAMES = frozenset(
    {
        "repo_root",
        "get_ipython",
        "display",
        "In",
        "Out",
        "_",
    }
)


def _argument_names(arguments: ast.arguments) -> set[str]:
    named = (
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
    )
    names = {argument.arg for argument in named}
    for optional in (arguments.vararg, arguments.kwarg):
        if optional is not None:
            names.add(optional.arg)
    return names


def _code_cells(path: Path) -> list[str]:
    notebook = nbformat.read(path, as_version=4)
    return [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and cell.source.strip()
    ]


def test_notebooks_exist() -> None:
    assert NOTEBOOK_PATHS, "No notebooks found to check."


@pytest.mark.parametrize(
    "path",
    NOTEBOOK_PATHS,
    ids=lambda path: path.name,
)
def test_notebook_is_valid_json_and_parses(path: Path) -> None:
    payload = json.loads(path.read_text())

    assert payload["nbformat"] >= 4
    for index, source in enumerate(_code_cells(path)):
        try:
            ast.parse(source)
        except SyntaxError as error:  # pragma: no cover - failure path
            raise AssertionError(
                f"{path.name} cell {index} does not parse: {error}"
            ) from error


@pytest.mark.parametrize(
    "path",
    NOTEBOOK_PATHS,
    ids=lambda path: path.name,
)
def test_notebook_imports_resolve(path: Path) -> None:
    """Every ``tiny_sima`` symbol a notebook imports must still exist."""

    missing: list[str] = []
    for source in _code_cells(path):
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module is None or not node.module.startswith("tiny_sima"):
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                if not hasattr(module, alias.name):
                    missing.append(f"{node.module}.{alias.name}")

    assert not missing, f"{path.name} imports names that do not exist: {missing}"


@pytest.mark.parametrize(
    "path",
    NOTEBOOK_PATHS,
    ids=lambda path: path.name,
)
def test_notebook_defines_every_name_it_uses(path: Path) -> None:
    """Catch names a cell reads but no earlier cell binds."""

    defined: set[str] = set(_IMPLICIT_NAMES) | set(dir(builtins))
    undefined: dict[str, int] = {}
    for index, source in enumerate(_code_cells(path)):
        tree = ast.parse(source)
        # Bindings are collected for the whole cell before uses are checked, so
        # a function defined lower in a cell may be referenced higher up.
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.alias):
                defined.add((node.asname or node.name).split(".")[0])
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                defined.add(node.name)
                defined.update(_argument_names(node.args))
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
            elif isinstance(node, ast.comprehension):
                for target in ast.walk(node.target):
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
            elif isinstance(node, ast.Lambda):
                defined.update(_argument_names(node.args))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in defined
            ):
                undefined.setdefault(node.id, index)

    assert not undefined, (
        f"{path.name} reads names that no cell defines: {sorted(undefined)}"
    )
