#!/bin/bash
#SBATCH --job-name=emir_sanity
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_logs/sanity_%j.out
#SBATCH --error=slurm_logs/sanity_%j.err

source slurm/_common.sh

python scripts/sanity_check.py

echo "End: $(date -Iseconds)"
