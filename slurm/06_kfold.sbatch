#!/bin/bash
#SBATCH --job-name=emir_kfold
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_logs/kfold_%x_%j.out
#SBATCH --error=slurm_logs/kfold_%x_%j.err

source slurm/_common.sh

CONFIG="${CONFIG:?CONFIG is required}"
N_SPLITS="${N_SPLITS:-5}"
SEED="${SEED:-1234}"

python scripts/run_kfold.py \
    --config "$CONFIG" \
    --manifest results/manifest.json \
    --n_splits "$N_SPLITS" \
    --seed "$SEED"

echo "End: $(date -Iseconds)"
