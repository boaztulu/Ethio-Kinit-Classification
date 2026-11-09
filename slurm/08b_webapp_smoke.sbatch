#!/bin/bash
#SBATCH --job-name=emir_web_smoke
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_email>
#SBATCH --output=slurm_logs/web_smoke_%j.out
#SBATCH --error=slurm_logs/web_smoke_%j.err

source slurm/_common.sh

PORT=$((7860 + RANDOM % 1000))
echo "Smoke test port: ${PORT}"

# Launch web app in background (CPU-only is fine for the index-page test)
python -u -m web.app \
    --checkpoint checkpoints/effnet_b0_mel_s1234/best.pt \
    --server_name 127.0.0.1 \
    --port "${PORT}" &
APP_PID=$!

# Wait for server to be reachable (max 90 s)
for i in $(seq 1 45); do
    sleep 2
    if curl -sf -o /dev/null "http://127.0.0.1:${PORT}/"; then
        echo "OK: server reachable on try $i"
        break
    fi
    echo "  try $i: not yet"
done

# Hit the root URL and capture status code + first 200 bytes
STATUS=$(curl -s -o /tmp/web_smoke_body_$$ -w '%{http_code}' "http://127.0.0.1:${PORT}/")
echo "GET / -> HTTP ${STATUS}"
echo "--- body head ---"
head -c 800 /tmp/web_smoke_body_$$
echo
echo "--- end body head ---"

# Cleanup
kill "${APP_PID}" 2>/dev/null || true
wait "${APP_PID}" 2>/dev/null || true
rm -f /tmp/web_smoke_body_$$

if [ "${STATUS}" = "200" ]; then
    echo "SMOKE TEST PASSED"
    exit 0
else
    echo "SMOKE TEST FAILED (status ${STATUS})"
    exit 1
fi
