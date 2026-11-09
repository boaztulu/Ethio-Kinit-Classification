#!/bin/bash
#SBATCH --job-name=emir_setup
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_logs/setup_%j.out
#SBATCH --error=slurm_logs/setup_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

echo "=== EMIR env setup job ${SLURM_JOB_ID} on $(hostname) ==="
echo "Start: $(date -Iseconds)"

module purge
module load conda/25.7.0 || module load conda

CONDA_BASE="$(conda info --base 2>/dev/null)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_PATH="$HOME/.conda/envs/emir-env"

if [[ ! -d "${ENV_PATH}" ]]; then
    echo "Creating env at ${ENV_PATH}"
    conda create -y -p "${ENV_PATH}" python=3.10
else
    echo "Env ${ENV_PATH} already exists; reusing"
fi

conda activate "${ENV_PATH}"

python -m pip install --upgrade pip wheel setuptools

# Pin numpy<2 + scipy<1.13 BEFORE torch so torch doesn't pull numpy 2.x
# (otherwise pip then has to uninstall numpy 2.x and reinstall - that step
# hung for 20+ min on the network FS in the first attempt).
python -m pip install --no-cache-dir 'numpy<2' 'scipy<1.13'

# Install torch (will keep our numpy<2 since pip's resolver respects it)
python -m pip install --no-cache-dir --no-deps \
    torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

# torch's other deps (filelock, jinja2, etc.) - let pip pull these from PyPI
python -m pip install --no-cache-dir \
    filelock 'jinja2>=3' 'sympy>=1.13' 'typing-extensions>=4.8' \
    'mpmath>=1.3' 'networkx>=3' 'fsspec>=2024'

# Project deps (numpy/scipy already satisfied, so no uninstall step)
python -m pip install --no-cache-dir -r requirements.txt

# Sanity
python - <<'PY'
import torch, torchaudio, transformers, librosa, numpy, scipy, pandas
print("torch       :", torch.__version__, "cuda:", torch.cuda.is_available())
print("torchaudio  :", torchaudio.__version__)
print("transformers:", transformers.__version__)
print("librosa     :", librosa.__version__)
print("numpy       :", numpy.__version__)
print("pandas      :", pandas.__version__)
PY

echo "Done. End: $(date -Iseconds)"
