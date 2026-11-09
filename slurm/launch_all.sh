#!/bin/bash
# Submit the full grid of architectures × seeds, chained on the preprocess job.
# Usage: ./slurm/launch_all.sh [PREP_JOB_ID]
# If PREP_JOB_ID is omitted, jobs do not wait on a dependency.

set -euo pipefail
cd "$(dirname "$0")/.."

PREP_JOB="${1:-}"
DEP_ARG=()
if [[ -n "$PREP_JOB" ]]; then
    DEP_ARG=(--dependency=afterok:$PREP_JOB)
fi

declare -A CONFIGS=(
    [ekm_mfcc]=configs/ekm_mfcc.yaml
    [ekm_mel]=configs/ekm_mel.yaml
    [effnet_b0]=configs/effnet_b0_mel.yaml
    [ast]=configs/ast.yaml
    [hubert]=configs/hubert.yaml
    [wav2vec2]=configs/wav2vec2.yaml
)

SEEDS=(1234 2024 7777)

for name in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$name]}"
    for s in "${SEEDS[@]}"; do
        echo "Submitting train: $name seed=$s"
        sbatch "${DEP_ARG[@]}" \
            -J "emir_${name}_s${s}" \
            --export=ALL,CONFIG=$cfg,SEED=$s,TAG_SUFFIX= \
            slurm/03_train.sbatch
    done
done
