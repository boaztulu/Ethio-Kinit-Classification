#!/bin/bash
#SBATCH --job-name=emir_prep
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_logs/prep_%j.out
#SBATCH --error=slurm_logs/prep_%j.err

source slurm/_common.sh

python -m src.data.preprocess \
    --manifest results/manifest.json \
    --cache_dir "${CACHE_DIR}" \
    --win_sec 3.0 \
    --overlap 0.5 \
    --target_sr 16000 \
    --top_db 30 \
    --n_mels 128

python -m src.data.splits \
    --manifest results/manifest.json \
    --out results/split_paper.json \
    --seed 1234

echo "End: $(date -Iseconds)"
