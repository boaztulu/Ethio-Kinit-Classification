#!/bin/bash
#SBATCH --job-name=emir_dl
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_logs/dl_%j.out
#SBATCH --error=slurm_logs/dl_%j.err

source slurm/_common.sh

python -u scripts/download_pretrained.py

echo "End: $(date -Iseconds)"
