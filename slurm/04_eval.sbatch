#!/bin/bash
#SBATCH --job-name=emir_eval
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_logs/eval_%x_%j.out
#SBATCH --error=slurm_logs/eval_%x_%j.err

# Usage: sbatch --export=ALL,CKPT=checkpoints/ast_s1234/best.pt,TAG=ast_s1234 \
#               -J emir_eval slurm/04_eval.sbatch

source slurm/_common.sh

CKPT="${CKPT:?CKPT is required}"
TAG="${TAG:-eval}"
OUT_DIR="${OUT_DIR:-results/$TAG/eval}"
EKM_ROWS="${EKM_ROWS:-}"

EXTRA=()
if [[ -n "$EKM_ROWS" ]]; then
    EXTRA+=(--ekm_eval_rows "$EKM_ROWS")
fi

python -m src.eval \
    --checkpoint "$CKPT" \
    --manifest results/manifest.json \
    --index cache/index.json \
    --split results/split_paper.json \
    --out_dir "$OUT_DIR" \
    --batch_size 32 \
    --num_workers 4 \
    "${EXTRA[@]}"

echo "End: $(date -Iseconds)"
