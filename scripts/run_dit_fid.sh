#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 || ! "$2" =~ ^[A-Za-z0-9][A-Za-z0-9-]*$ ]]; then
  echo 'Usage: bash scripts/run_dit_fid.sh {generate|score|run|clear} RUN [COUNT [MODE,MODE,...]]' >&2
  exit 2
fi
phase=$1
case "$phase" in generate|score|run|clear) ;; *) exit 2 ;; esac
count=${3:-100}
mode_list=${4:-dense,residual,planeclear,combined}
[[ "$count" =~ ^[1-9][0-9]*$ && "$count" != 1 ]] || exit 2
[[ "$mode_list" =~ ^(dense|residual|planeclear|combined)(,(dense|residual|planeclear|combined))*$ ]] || exit 2
IFS=, read -r -a modes <<< "$mode_list"
declare -A seen=()
for mode in "${modes[@]}"; do
  [[ ! ${seen[$mode]+yes} ]] || exit 2
  seen[$mode]=1
done
if [[ "$phase" == score || "$phase" == run ]]; then
  [[ ${seen[dense]+yes} && ${#modes[@]} -ge 2 ]] || {
    echo 'Scoring requires Dense and at least one HARP mode' >&2
    exit 2
  }
fi

cd "$(dirname "$0")/.."
root=$PWD
assets=${HARP_ASSETS:-$root/assets}
gpu=${HARP_GPU:-0}
batch_size=${HARP_DIT_BATCH_SIZE:-2}
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || exit 2
name="dit-fid-$2"
target="results/$name"
out="/opt/harp/$target"
mkdir -p results
[[ ! -L results && ! -L "$target" ]] || exit 1
exec 9>"results/.$name.lock"
lock=-x
[[ "$phase" != generate ]] || lock=-s
flock -n "$lock" 9 || { echo "Run in progress: $name" >&2; exit 1; }

if [[ "$phase" == clear ]]; then
  [[ $# == 2 ]] || exit 2
  if [[ -e "$target" ]]; then
    [[ -d "$target" && -f "$target/.harp-dit-paired-run" && ! -L "$target/.harp-dit-paired-run" ]] || exit 1
    rm -rf -- "$target"
    echo "Cleared $target"
  fi
  exit 0
fi
(
  exec 7>"results/.$name-setup.lock"
  flock 7
  if [[ ! -e "$target" ]]; then
    mkdir "$target"
    touch "$target/.harp-dit-paired-run"
  fi
  [[ -d "$target" && -f "$target/.harp-dit-paired-run" && ! -L "$target/.harp-dit-paired-run" ]] || exit 1
)

mkdir -p cache
common=(--rm --pull=never --network=none --user "$(id -u):$(id -g)"
  -e PYTHONPATH=/opt/harp -e HOME=/opt/harp/cache -e PYTHONDONTWRITEBYTECODE=1
  --mount "type=bind,src=$root/quality,dst=/opt/harp/quality,readonly"
  --mount "type=bind,src=$root/results,dst=/opt/harp/results"
  --mount "type=bind,src=$root/cache,dst=/opt/harp/cache")

generate() (
  mode=$1
  exec 8>"results/.$name-$mode.lock"
  flock -n 8 || { echo "Generation in progress: $name/$mode" >&2; exit 1; }
  docker run "${common[@]}" --device="nvidia.com/gpu=$gpu" -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --mount "type=bind,src=$root/harp,dst=/opt/harp/harp,readonly" \
    --mount "type=bind,src=$root/src/dit-original,dst=/opt/harp/src/dit-original,readonly" \
    --mount "type=bind,src=$assets/weights,dst=/opt/harp/assets/weights,readonly" \
    harp-ae-gpu:unified-v0.2 python -m quality.dit --paired "$mode" --balanced-cohort \
      --count "$count" --batch-size "$batch_size" --steps 50 --output "$out/$mode"
)

if [[ "$phase" == generate || "$phase" == run ]]; then
  for mode in "${modes[@]}"; do generate "$mode"; done
fi
if [[ "$phase" == score || "$phase" == run ]]; then
  scorer=(docker run "${common[@]}" -e CUDA_VISIBLE_DEVICES= -e TF_ENABLE_ONEDNN_OPTS=0
    -e OMP_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8
    --mount "type=bind,src=$assets/quality,dst=/quality,readonly" harp-ae-gpu:quality-v0.4)
  comparison=(--output "$out" --count "$count" --modes "$mode_list")
  "${scorer[@]}" python -m quality.dit_paired check "${comparison[@]}"
  for mode in "${modes[@]}"; do
    "${scorer[@]}" python -m quality.fid --samples "$out/$mode" --full-only
  done
  "${scorer[@]}" python -m quality.dit_paired summarize "${comparison[@]}"
fi
