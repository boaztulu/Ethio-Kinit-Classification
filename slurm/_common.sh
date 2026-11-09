#!/bin/bash
# Shared SLURM init for EMIR jobs. Sourced by every sbatch script.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

echo "=== Job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "Start: $(date -Iseconds)"

module purge
module load conda/25.7.0 || module load conda || module load conda/25.1.1

CONDA_BASE="$(conda info --base 2>/dev/null)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_NAME="${ENV_NAME:-emir-env}"
ENV_PATH="${ENV_PATH:-$HOME/.conda/envs/${ENV_NAME}}"

if [[ -d "${ENV_PATH}" ]]; then
    conda activate "${ENV_PATH}"
else
    echo "WARNING: ENV_PATH ${ENV_PATH} not found; falling back to env name ${ENV_NAME}"
    conda activate "${ENV_NAME}" || true
fi

PROJECT_ROOT="${PROJECT_ROOT:-.}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/EMIR_Dataset_V1}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/cache}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export PYTHONUNBUFFERED=1
# Don't let ~/.local/lib/python*/site-packages shadow the env (anyio etc.).
export PYTHONNOUSERSITE=1
# Force offline once weights are cached - prevents HF/transformers from
# silently hanging on a slow HTTP probe on compute nodes.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${CACHE_DIR}"

echo "Project: ${PROJECT_ROOT}"
echo "Env:     ${ENV_PATH}"
echo "Data:    ${DATA_ROOT}"
echo "Cache:   ${CACHE_DIR}"
echo "Python:  $(which python) ($(python --version 2>&1))"
