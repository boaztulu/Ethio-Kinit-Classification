#!/bin/bash
#SBATCH --job-name=emir_web
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_logs/web_%j.out
#SBATCH --error=slurm_logs/web_%j.err

source slurm/_common.sh

PORT="${PORT:-7860}"
echo "Reach the demo from HiPerGator Desktop (Chrome) at:"
echo "    http://$(hostname -s):${PORT}/"
echo "    http://$(hostname -f):${PORT}/"

# Ensemble of the 3 EffNet-B0 seeds.
# --share publishes a public gradio.live tunnel so the demo is reachable
# without needing direct compute-node access from the browser.
python -u -m web.app \
    --ensemble \
        checkpoints/effnet_b0_mel_s1234/best.pt \
        checkpoints/effnet_b0_mel_s2024/best.pt \
        checkpoints/effnet_b0_mel_s7777/best.pt \
    --server_name 0.0.0.0 \
    --port "${PORT}" \
    --share

echo "End: $(date -Iseconds)"
