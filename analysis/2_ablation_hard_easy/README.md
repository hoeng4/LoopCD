# Applying LoopCD only at HARD or EASY tokens

The gate is computed online at each decode step from the 32-iteration
trajectory LoopCD already needs: HARD iff `d = m_peak − m_final ≥ ε` for the
final-iteration top-1 token, with the token type taken from that token's
surface form. Where the gate fires the next token comes from the LoopCD blend,
elsewhere from greedy decoding.

```bash
for mode in full hard_only easy_only hard_num easy_num hard_op easy_op; do
  for s in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$s python -m analysis.2_ablation_hard_easy.gated_loopcd \
      --mode $mode --task gsm8k --shard-idx $s --shard-count 8 \
      --out runs/gated/${mode}_s${s}.json &
  done
  wait
done
```

Modes: `full` (LoopCD everywhere), `hard_only` / `easy_only`, and the
hardness × type combinations `hard_num`, `easy_num`, `hard_op`, `easy_op`.
Hyperparameters default to the paper's (`K_a = 8`, `λ = 0.3`, `α = 0.1`,
`ε = 1.5`).

Each run records per document the completion, both verdicts, how many tokens
the gate fired at (`n_gate`) and how often the blend changed the token
(`n_fire`).
