# Usage

Both models (Huginn, Ouro) share the same command-line interface.
The driver **is** the entry point — no custom top-level `run.py`,
just call the module directly.

## Huginn-0125

```bash
python -m loopcd.huginn.iter_cd_huginn_patched \
  --task {gsm8k, math500, humaneval, mbpp, strategyqa} \
  --ka-mode {full, fixed}                              \
  --k-amateur 8      # amateur iter (fixed mode)       \
  --k-expert 32      # expert iter (usually r=32)      \
  --lambda-cd 0.3    # contrast strength               \
  --alpha-plaus 0.1  # plausibility filter α           \
  --max-new-tokens 256                                 \
  --shard-idx 0 --shard-count 8                        \
  --out runs/<name>.json
```

- `--ka-mode full` → baseline greedy (no CD).
- `--ka-mode fixed` → LoopCD with the chosen `K_a, λ, α`.

## Ouro-1.4B

```bash
python -m loopcd.ouro.iter_cd_ouro_patched \
  --model ByteDance/Ouro-1.4B \
  --task gsm8k \
  --ka-mode fixed --k-amateur 1 --lambda-cd 0.1 --alpha-plaus 0.1 \
  --max-new-tokens 256 \
  --out runs/ouro_gsm8k_loopcd.json
```

Ouro `total_ut_steps = 4`, so valid `K_a ∈ {0, 1, 2}` and
`K_e = 4` (== `total_ut_steps`).

## Task-specific `--max-new-tokens`

Match these to the paper for exact-match reproduction:

| task | max_new_tokens |
|---|---|
| GSM8K | 256 |
| MATH-500 | 512 |
| HumanEval | 512 |
| MBPP | 1024 |
| StrategyQA | 256 |

## Sharding

Both drivers use deterministic **positional** sharding
(`sub[i % shard_count == shard_idx]`), so 8-way and 24-way runs are
consistent — the union covers the full dataset exactly once. Launch
one process per (GPU × shard) with `CUDA_VISIBLE_DEVICES=$s ... --shard-idx $s`.

See [`scripts/launch_examples/`](../scripts/launch_examples/) for a
ready-to-use 8-way template.

## Output format

Each shard writes `{out}/shard_<idx>.json`:

```json
{
  "config": { …args dict… },
  "summary": {"n_done": 165, "n_flex": 61, "n_strict": 42, "t_total_s": 3212.4},
  "rows": [
    { "doc_id": 0, "completion": "…", "flex_ok": true, "strict_ok": false,
      "n_gen": 87, "dt_s": 19.3 },
    …
  ]
}
```

Resume-safe: re-running the same command picks up from `rows` and
skips completed `doc_id`s. Save cadence: every 10 docs per shard.
