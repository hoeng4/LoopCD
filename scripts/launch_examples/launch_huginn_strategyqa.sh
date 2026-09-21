#!/usr/bin/env bash
# Huginn-0125 LoopCD on StrategyQA (train split, 2290 docs), 8-way. Paper config: K_a=8, λ=0.3.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
OUT_DIR=${OUT_DIR:-runs/huginn_strategyqa_loopcd_ka8_l0.3}
mkdir -p "$OUT_DIR"
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$s nohup python -m loopcd.huginn.iter_cd_huginn_patched \
    --task strategyqa --max-new-tokens 256 \
    --ka-mode fixed --k-amateur 8 --k-expert 32 \
    --lambda-cd 0.3 --alpha-plaus 0.1 \
    --shard-idx $s --shard-count 8 \
    --out "$OUT_DIR/shard_${s}.json" \
    > "$OUT_DIR/shard_${s}.log" 2>&1 &
done; wait
echo "StrategyQA LoopCD run done: $OUT_DIR"
