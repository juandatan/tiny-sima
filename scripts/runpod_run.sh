#!/usr/bin/env bash
# Run notebooks headless on a Runpod pod, then stop the pod.
#
#   bash scripts/runpod_run.sh notebooks/07_improve_executor.ipynb \
#       notebooks/08_open_ended_achievements.ipynb [--set NAME=VALUE ...]
#
# The arguments go to scripts/run_notebooks.py. The virtual environment lives
# beside the repository, on the persistent volume, because a stopped pod loses
# its container disk and with it anything installed into the image's Python.
set -euo pipefail

# Runpod exports RUNPOD_POD_ID and the API key runpodctl needs through this
# file, which only an interactive shell sources, so a launch over a plain
# `ssh host command` would otherwise leave the pod running.
if [[ -z "${RUNPOD_POD_ID:-}" && -r /etc/rp_environment ]]; then
    # shellcheck disable=SC1091
    source /etc/rp_environment
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${TINY_SIMA_VENV:-${repo_root}/../tiny-sima-venv}"

if [[ ! -x "${venv}/bin/python" ]]; then
    python_bin=""
    for candidate in python3.12 python3.11; do
        if command -v "${candidate}" > /dev/null; then
            python_bin="${candidate}"
            break
        fi
    done
    if [[ -z "${python_bin}" ]]; then
        echo "Tiny-SIMA needs Python 3.11 or 3.12; neither is on PATH." >&2
        exit 1
    fi
    echo "Creating ${venv} with ${python_bin}"
    "${python_bin}" -m venv "${venv}"
    "${venv}/bin/python" -m pip install --quiet --upgrade pip
fi
"${venv}/bin/python" -m pip install --quiet nbclient nbformat ipykernel

log_dir="${repo_root}/runs/notebook-runs"
mkdir -p "${log_dir}"
log="${log_dir}/$(date +%Y%m%d-%H%M%S).log"
nohup "${venv}/bin/python" "${repo_root}/scripts/run_notebooks.py" "$@" \
    > "${log}" 2>&1 &
echo "Running as PID $!; the pod stops when the run ends."
echo "Follow it with: tail -f ${log}"
