#!/bin/bash
#SBATCH --job-name=emir_train
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_logs/train_%x_%j.out
#SBATCH --error=slurm_logs/train_%x_%j.err

# Usage: sbatch --export=ALL,CONFIG=configs/ast.yaml,SEED=1234,TAG_SUFFIX="" \
#               -J emir_ast slurm/03_train.sbatch

source slurm/_common.sh

CONFIG="${CONFIG:-configs/ekm_mfcc.yaml}"
SEED="${SEED:-1234}"
TAG_SUFFIX="${TAG_SUFFIX:-}"

echo "Config: $CONFIG"
echo "Seed:   $SEED"
echo "Suffix: $TAG_SUFFIX"

python -m src.train \
    --config "$CONFIG" \
    --seed "$SEED" \
    --tag_suffix "$TAG_SUFFIX"

echo "End: $(date -Iseconds)"
