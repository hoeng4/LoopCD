# Analyses

`capture.py` runs greedy decoding once per benchmark and records, for every
generated token, the per-iteration LM-head readout needed to label it HARD or
EASY (`hard_tokens.py`). The sub-packages read those captures.

```bash
python -m analysis.capture --task gsm8k --limit 1319 \
    --shard-idx 0 --shard-count 8 --out captures/gsm8k/shard_0.json
```

| directory | contents |
|---|---|
| [`1_figure3_hard_token_types/`](1_figure3_hard_token_types/) | Figure 3 — which token types are labelled HARD |
| [`2_ablation_hard_easy/`](2_ablation_hard_easy/) | applying LoopCD only at HARD or EASY tokens |
