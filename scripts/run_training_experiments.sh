#!/usr/bin/env bash
set -euo pipefail

BASE_ARGS=(
  foundationts train
  --dataset-path time300b_selected
  --steps-per-epoch 80
  --epochs 1
  --batch-size 32
  --seq-max-len 4096
  --seq-stride 4096
  --num-expert-layers 1
  --hidden-size 384
  --n-head 12
  --n-decoder-layers 12
  --num-experts 8
  --k 2
  --d-ff 1536
  --d-expert 768
  --attn-backend flash
  --log-every 10
  --checkpoint-every 0
  --log-perf-metrics
  --mfu-peak-tflops 989
)

LOG_DIR=${LOG_DIR:-experiments/$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOG_DIR"

echo "Logging to $LOG_DIR"

run_case() {
  local name=$1
  shift
  local slug
  slug=$(echo "$name" | tr -cs '[:alnum:]' '_')
  local log_file="$LOG_DIR/${slug}.log"

  echo
  echo ">>> $name"
  echo "Log: $log_file"

  "${BASE_ARGS[@]}" "$@" 2>&1 | tee "$log_file"
}

# run_case "dense_ffn_bf16_sdpa" \
#   --num-experts 1 \
#   --k 1 \
#   --moe-impl standard \
#   --attn-backend sdpa

# run_case "dense_ffn_bf16_flash" \
#   --num-experts 1 \
#   --k 1 \
#   --moe-impl standard \
#   --attn-backend flash

# run_case "dense_ffn_bf16_flash_compile" \
#   --num-experts 1 \
#   --k 1 \
#   --moe-impl standard \
#   --attn-backend flash \
#   --compile

# run_case "dense_compute_matched_moe_active" \
#   --num-experts 1 \
#   --k 1 \
#   --moe-impl standard \
#   --d-expert 1536 \
#   --compile

# # MoE progression.
# run_case "moe_naive_bf16" \
#   --moe-impl standard

# run_case "moe_naive_bf16_compile" \
#   --moe-impl standard \
#   --compile

# run_case "moe_onehot_bf16" \
#   --moe-impl onehot

# run_case "moe_onehot_bf16_compile" \
#   --moe-impl onehot \
#   --compile

# run_case "moe_scatter_bf16" \
#   --moe-impl efficient

# run_case "moe_scatter_bf16_compile" \
#   --moe-impl efficient \
#   --compile

# # Capacity / tile sweeps (scatter + compile).
# run_case "scatter_compile_cap_0_9" \
#   --moe-impl efficient \
#   --capacity-factor 0.9 \
#   --compile

# run_case "scatter_compile_cap_1_1" \
#   --moe-impl efficient \
#   --capacity-factor 1.1 \
#   --compile

# run_case "scatter_compile_cap_1_5" \
#   --moe-impl efficient \
#   --capacity-factor 1.5 \
#   --compile

# run_case "scatter_compile_tile_64" \
#   --moe-impl efficient \
#   --moe-m-tile 64 \
#   --compile

# run_case "scatter_compile_tile_128" \
#   --moe-impl efficient \
#   --moe-m-tile 128 \
#   --compile

# # Expert layer tests.
# run_case "per_expert_tile_1" \
#   --moe-impl standard \
#   --moe-m-tile 1 \
#   --compile

# run_case "per_expert_tile_64" \
#   --moe-impl standard \
#   --moe-m-tile 64 \
#   --compile

# run_case "per_expert_tile_128" \
#   --moe-impl standard \
#   --moe-m-tile 128 \
#   --compile

# run_case "logical_dense_cap_0_9" \
#   --moe-impl efficient \
#   --capacity-factor 0.9

# run_case "logical_dense_cap_1_1" \
#   --moe-impl efficient \
#   --capacity-factor 1.1

run_case "logical_dense_cap_1_3" \
  --moe-impl efficient \
  --capacity-factor 1.3

run_case "logical_dense_cap_1_5" \
  --moe-impl efficient \
  --capacity-factor 1.5
