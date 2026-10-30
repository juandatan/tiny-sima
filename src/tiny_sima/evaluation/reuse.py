"""Reuse saved evaluation arms whose setting has not changed.

An incremental change to one planner leaves the baselines beside it as they
were, so a run can reload their saved results instead of rerunning them. Each
saved result carries the provenance it ran under: the seed, episode count,
executor, VLM configuration and anything else its outcome depends on. A saved
result is reused only when its provenance equals the current run's, field for
field; one saved before provenance was recorded is never reused.

Provenance cannot see code. A change to code an arm runs, such as the per-turn
VLM prompt, needs that arm rerun, which is why each result also records the
commit it ran at.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def run_provenance(**fields: Any) -> dict[str, Any]:
    """Return the fields as plain JSON values, so saved and new ones compare.

    Values that JSON cannot hold, such as paths, are stored as strings.
    """

    return json.loads(json.dumps(fields, sort_keys=True, default=str))


def fingerprint(value: Any) -> str:
    """Return a short hash of a JSON value, for provenance of a large one."""

    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def git_commit(repo_root: str | Path) -> str | None:
    """Return the checked-out commit, with ``+dirty`` for local changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}+dirty" if dirty else commit


def with_provenance(
    result: Mapping[str, Any],
    provenance: Mapping[str, Any],
    commit: str | None,
) -> dict[str, Any]:
    """Return a copy of a result stamped with its provenance and commit."""

    return {**result, "provenance": dict(provenance), "commit": commit}


def provenance_mismatch(
    saved: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str | None:
    """Return why a saved result cannot be reused, or None when it can."""

    recorded = saved.get("provenance")
    if not isinstance(recorded, Mapping):
        return "saved without provenance"
    differing = sorted(
        key
        for key in set(recorded) | set(provenance)
        if recorded.get(key) != provenance.get(key)
    )
    if differing:
        return f"{', '.join(differing)} changed"
    return None
