#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo 'Usage: bash scripts/run_quality.sh {dit|wan} {smoke|generate|score|development} RUN'
  echo 'Use the same RUN to resume or score a development run.'
}
if [[ $# == 1 && $1 == --help ]]; then usage; exit 0; fi
if [[ $# != 3 ]]; then usage >&2; exit 2; fi
model=$1
phase=$2
run=$3
[[ "$model" == dit || "$model" == wan ]] || { usage >&2; exit 2; }
[[ "$run" =~ ^[a-zA-Z0-9][a-zA-Z0-9-]*$ ]] || { echo 'Invalid run name' >&2; exit 2; }
case "$phase" in smoke|generate|score|development) ;; *) usage >&2; exit 2;; esac

cd "$(dirname "$0")/.."
root=$PWD
assets=${HARP_ASSETS:-$root/assets}
gpu=${HARP_GPU:-0}
gen_image=harp-ae-gpu:unified-v0.2
score_image=harp-ae-gpu:quality-v0.4
mkdir -p results cache
if [[ "$phase" == smoke ]]; then
  exec bash scripts/run_model_budget.sh "$model" "source-budget-$model-smoke-$run"
fi

out="$root/results/$model-$run"
if [[ "$phase" == score && ! -d "$out" ]]; then
  echo "Run generation first: $out" >&2
  exit 2
fi
mkdir -p "$out"
exec 9>"$out/.run.lock"
flock -n 9 || { echo "Another command is using $out" >&2; exit 1; }
echo "Outputs: $out"
common=(--rm --pull=never --network=none --user "$(id -u):$(id -g)" --shm-size=4g
  -e PYTHONPATH=/opt/harp:/opt/harp/src/Wan2.1 -e HOME=/opt/harp/cache -e PYTHONDONTWRITEBYTECODE=1
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e OPENBLAS_NUM_THREADS=4 -e OMP_NUM_THREADS=4
  --mount "type=bind,src=$root/cache,dst=/opt/harp/cache")
for folder in harp quality src scripts manifests; do
  common+=(--mount "type=bind,src=$root/$folder,dst=/opt/harp/$folder,readonly")
done
common+=(--mount "type=bind,src=$assets/weights,dst=/opt/harp/assets/weights,readonly"
  --mount "type=bind,src=$assets/quality,dst=/opt/harp/assets/quality,readonly")
common+=(--tmpfs /opt/harp/harp/additional_models:ro)
if [[ "$model" == dit ]]; then
  container_out=/opt/harp/results/source-budget-dit-review
else
  # The immutable Wan calibration receipt contains these absolute container paths.
  container_out=/opt/harp/results/wan-stage4-9f-v1
fi
common+=(--mount "type=bind,src=$out,dst=$container_out"
  --mount "type=bind,src=$assets/quality,dst=/quality,readonly")

check_gpu() {
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Set HARP_GPU to an NVIDIA device index' >&2; return 2; }
  local used
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  if (( used >= 1024 )); then
    echo "GPU $gpu is busy ($used MiB); select a free GPU" >&2
    return 1
  fi
}
generate() {
  check_gpu
  if [[ "$model" == dit ]]; then
    docker run "$@" --device="nvidia.com/gpu=$gpu" "$gen_image" \
      python /opt/harp/scripts/run_model_budget.py dit source-budget-dit-review --count 16
  else
    docker run "$@" --device="nvidia.com/gpu=$gpu" \
      --mount "type=bind,src=$root/results/wan-stage4-9f-v1,dst=/reference/wan-stage4,readonly" \
      -e RUN_OUTPUT="$container_out" "$gen_image" bash -ec '
        python /opt/harp/scripts/prepare_wan_run.py \
          /reference/wan-stage4 "$RUN_OUTPUT" /opt/harp/src/Wan2.1/harp/lambda.json
        python -m quality.wan_stage4 --phase heldout --root "$RUN_OUTPUT" --frames 9 --resume --gpu-deterministic 1
      '
  fi
}
score() {
  if [[ "$model" == dit ]]; then
    docker run "$@" -e CUDA_VISIBLE_DEVICES= -e TF_ENABLE_ONEDNN_OPTS=0 "$score_image" \
      python /opt/harp/scripts/score_dit_development.py "$container_out"
  else
    check_gpu
    docker run "$@" --device="nvidia.com/gpu=$gpu" -e HF_HOME=/quality/hf-cache \
      -e TOKENIZERS_PARALLELISM=false -e RUN_OUTPUT="$container_out" "$score_image" bash -ec '
        python /opt/harp/scripts/wan_stage4_score.py \
          --root "$RUN_OUTPUT" --output "$RUN_OUTPUT/heldout-scores" --resume
        python -c "import json,sys; p=sys.argv[1]; s=json.load(open(p)); sys.exit(0 if s[\"all_modes_all_dimensions_finite\"] else \"VBench has undefined or failed dimensions; see \" + p)" "$RUN_OUTPUT/heldout-scores/summary.json"
        python /opt/harp/scripts/stage4_wan_uncertainty.py --root "$RUN_OUTPUT"
      '
  fi
}
if [[ "$phase" == generate || "$phase" == development ]]; then
  generate "${common[@]}" 2>&1 | tee -a "$out/generation.log"
fi
if [[ "$phase" == score || "$phase" == development ]]; then
  score "${common[@]}" 2>&1 | tee -a "$out/scoring.log"
fi
