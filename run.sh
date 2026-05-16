#!/usr/bin/env bash
# Wrapper script to run the refusal evaluation pipeline with correct CUDA environment.
#
# Usage:
#   ./run.sh configs/my-model.yaml [extra args...]
#
# If the system lacks a CUDA toolkit (no nvcc on PATH), this script auto-detects
# the conda env "activation-steer" and points CUDA_HOME / LIBRARY_PATH to it.
# If nvcc is already available, it runs directly.

set -euo pipefail

CONFIG="${1:?Usage: run.sh <config.yaml> [extra args...]}"
shift

# If nvcc is already on PATH, run directly
if command -v nvcc &>/dev/null; then
    exec uv run python -m src.compute_refusal_score --config "$CONFIG" "$@"
fi

# Try conda env fallback
CONDA_ENV_NAME="${CONDA_ENV_NAME:-activation-steer}"
CONDA_PREFIX="$(conda env list 2>/dev/null | grep "^${CONDA_ENV_NAME} " | awk '{print $NF}')"

if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: nvcc not found and conda env '${CONDA_ENV_NAME}' not detected."
    echo "Install the CUDA toolkit: conda install -n ${CONDA_ENV_NAME} cuda-toolkit"
    exit 1
fi

if [ ! -f "${CONDA_PREFIX}/bin/nvcc" ]; then
    echo "ERROR: ${CONDA_PREFIX}/bin/nvcc not found."
    echo "Install the CUDA toolkit: conda install -n ${CONDA_ENV_NAME} cuda-toolkit"
    exit 1
fi

echo "Using CUDA toolkit from conda env: ${CONDA_PREFIX}"

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

exec uv run python -m src.compute_refusal_score --config "$CONFIG" "$@"
