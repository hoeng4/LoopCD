#!/usr/bin/env bash
# Ouro-1.4B LoopCD on GSM8K, 8-way (Ouro is smaller — can also do 2-3/GPU if VRAM allows).
# Paper "Ours" config for Ouro-1.4B GSM8K: K_a=0, λ=0.3.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
OUT_DIR=${OUT_DIR:-runs/ouro14b_gsm8k_loopcd_k0_l0.3}
mkdir -p "$OUT_DIR"
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$s nohup python -m loopcd.ouro.iter_cd_ouro_patched \
    --model ByteDance/Ouro-1.4B \
    --task gsm8k --max-new-tokens 256 \
    --ka-mode fixed --k-amateur 0 \
    --lambda-cd 0.3 --alpha-plaus 0.1 \
    --shard-idx $s --shard-count 8 \
    --out "$OUT_DIR/shard_${s}.json" \
    > "$OUT_DIR/shard_${s}.log" 2>&1 &
done; wait
echo "Ouro GSM8K LoopCD run done: $OUT_DIR"
