#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
assets=${HARP_ASSETS:-$root/assets}
gpu=${HARP_GPU:-0}
[[ "$gpu" =~ ^[01]$ ]]
test "$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024
mounts=()
for folder in harp quality src scripts manifests; do
  mounts+=(--mount "type=bind,src=$root/$folder,dst=/opt/harp/$folder,readonly")
done
for folder in weights quality; do
  mounts+=(--mount "type=bind,src=$assets/$folder,dst=/opt/harp/assets/$folder,readonly")
done
# The old Wan checkpoint identity hashes every Python file below harp/.
# Mount the original reference source view for these two accepted deployments.
docker run --rm --pull=never --network=none --device="nvidia.com/gpu=$gpu" \
  --user "$(id -u):$(id -g)" --shm-size=4g \
  -e PYTHONPATH=/opt/harp:/opt/harp/src/Wan2.1 -e HOME=/opt/harp/cache \
  -e PYTHONDONTWRITEBYTECODE=1 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "${mounts[@]}" --tmpfs /opt/harp/harp/additional_models:ro \
  --mount "type=bind,src=$root/results,dst=/opt/harp/results" \
  --mount "type=bind,src=$root/cache,dst=/opt/harp/cache" \
  harp-ae-gpu:unified-v0.2 python /opt/harp/scripts/run_model_budget.py "$@"
