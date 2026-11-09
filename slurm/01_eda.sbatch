#!/bin/bash
#SBATCH --job-name=emir_eda
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_logs/eda_%j.out
#SBATCH --error=slurm_logs/eda_%j.err

source slurm/_common.sh

python scripts/run_eda.py \
    --data_root "${DATA_ROOT}" \
    --out_dir "${PROJECT_ROOT}" \
    --seed 1234

echo "End: $(date -Iseconds)"
