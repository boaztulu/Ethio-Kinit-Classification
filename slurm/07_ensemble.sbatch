#!/bin/bash
#SBATCH --job-name=emir_ens
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_logs/ens_%j.out
#SBATCH --error=slurm_logs/ens_%j.err

source slurm/_common.sh

# Ensemble top-3 single-seed checkpoints
python scripts/ensemble.py --tags effnet_b0_mel_s1234 ast_s7777 wav2vec2_s7777 \
    --out results/ensemble_top3.json

# Ensemble best-of-each-family
python scripts/ensemble.py --tags effnet_b0_mel_s1234 effnet_b0_mel_s2024 effnet_b0_mel_s7777 \
    --out results/ensemble_effnet3.json

# Big ensemble: all best seeds from each arch
python scripts/ensemble.py --tags \
    effnet_b0_mel_s1234 effnet_b0_mel_s7777 \
    effnet_b2_mel_s1234 \
    ast_s7777 \
    wav2vec2_s7777 \
    hubert_s1234 \
    --out results/ensemble_all.json

echo "End: $(date -Iseconds)"
