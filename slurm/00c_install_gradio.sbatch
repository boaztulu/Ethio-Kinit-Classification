#!/bin/bash
#SBATCH --job-name=emir_gradio
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_logs/gradio_install_%j.out
#SBATCH --error=slurm_logs/gradio_install_%j.err

source slurm/_common.sh

python -m pip install --no-cache-dir 'gradio>=4.40,<5' 'fastapi>=0.110' 'uvicorn>=0.30' Pillow

python - <<'PY'
import gradio as gr
print("gradio:", gr.__version__)
PY

echo "End: $(date -Iseconds)"
