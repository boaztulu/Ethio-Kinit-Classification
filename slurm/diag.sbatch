#!/bin/bash
#SBATCH --job-name=emir_diag
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_logs/diag_%j.out
#SBATCH --error=slurm_logs/diag_%j.err

source slurm/_common.sh

python -u scripts/import_diag.py

echo "End: $(date -Iseconds)"
