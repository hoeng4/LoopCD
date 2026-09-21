# Reproduce paper results

## Paper Table 2 (Huginn-0125 "Ours")

Task-specific hyperparameters (from the Appendix B.2 search):

| task | K_a | λ | max_new | reported flex | reported strict/plus |
|---|---:|---:|---:|---:|---:|
| GSM8K       | **6**  | 0.3 | 256  | 36.62 | 26.46 |
| MATH-500    | 8      | **0.1** | 512  | 15.60 | 15.20 |
| HumanEval+  | 8      | 0.3 | 512  | 30.49 | 27.44 |
| MBPP+       | 8      | 0.3 | 1024 | 42.06 | 36.77 |
| StrategyQA  | 8      | 0.3 | 256  | —     | 55.24 |

Baseline (`--ka-mode full`) numbers per task:

| task | flex | strict |
|---|---:|---:|
| GSM8K       | 33.21 | 23.12 |
| MATH-500    | 13.60 | 12.20 |
| HumanEval+  | 24.39 | 20.73 |
| MBPP+       | 40.74 | 33.60 |
| StrategyQA  |       | 54.02 |

## Paper Table 2 (Ouro-1.4B "Ours") — best K_a, λ per task

| task | K_a | λ | reported flex | reported strict |
|---|---:|---:|---:|---:|
| GSM8K       | 0 | 0.3 | 81.05 | 64.06 |
| MATH-500    | 1 | 0.1 | 51.40 | 37.40 |
| HumanEval+  | 1 | 0.3 | 74.39 | 70.12 |
| MBPP+       | 0 | 0.2 | 74.60 | 62.96 |
| StrategyQA  | 0 | 0.3 | —     | 66.81 |

## Example: full 8-way GSM8K on Huginn (paper "Ours")

```bash
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$s python -m loopcd.huginn.iter_cd_huginn_patched \
    --task gsm8k --max-new-tokens 256 \
    --ka-mode fixed --k-amateur 6 --k-expert 32 \
    --lambda-cd 0.3 --alpha-plaus 0.1 \
    --shard-idx $s --shard-count 8 \
    --out runs/huginn_gsm8k_ka6_l0.3/shard_${s}.json &
done
wait
```

Baseline (same command with `--ka-mode full`, drop the CD args) →
33.21 % flex / 23.12 % strict on the same 1,319 doc set.

## Hyperparameter search space (Appendix B.2)

- Huginn: `K_a ∈ {6, 8, 10}`, `λ ∈ {0.1, 0.2, 0.3}`
- Ouro:   `K_a ∈ {0, 1, 2}`,     `λ ∈ {0.1, 0.2, 0.3}`
- Plausibility filter α is fixed at `0.1` throughout the paper.

## Determinism

For **exact** reproduction of Table 2 (byte-identical completions):

1. Use greedy decoding (default in this repo: `do_sample=False`).
2. Apply the zero-init determinism patch — this is done automatically
   at the top of the driver:
   ```python
   model.initialize_state = lambda e, scale=1.0: torch.zeros_like(e)
   ```
3. Same Docker container / torch + transformers version as in
   `requirements.txt`.

Under these conditions, our internal reruns of Huginn iter-CD are
**byte-identical (4722 / 4722)** across sessions.

If you re-enable Huginn's default random initial latent state (drop
step 2), each seed gives a slightly different completion. The paper's
appendix reports mean accuracy over four seeds for the fixed
configuration.
