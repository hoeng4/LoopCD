# [EMNLP 2026 Main] LoopCD: Loop-wise Contrastive Decoding for Improving Reasoning in Looped Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2609.24196-b31b1b.svg)](https://arxiv.org/abs/2609.24196)

Official Implementation of "[LoopCD: Loop-wise Contrastive Decoding for Improving Reasoning in Looped Language Models](https://arxiv.org/abs/2609.24196)"

## Quick start

```bash
# 1) Set up the environment (Python venv + deps)
bash scripts/setup_env.sh

# 2) Run baseline greedy on GSM8K
python -m loopcd.huginn.iter_cd_huginn_patched \
  --task gsm8k --ka-mode full --max-new-tokens 256 \
  --out runs/huginn_baseline_gsm8k.json

# 3) Run LoopCD on GSM8K (paper hparams)
python -m loopcd.huginn.iter_cd_huginn_patched \
  --task gsm8k --ka-mode fixed --k-amateur 8 --k-expert 32 \
  --lambda-cd 0.3 --alpha-plaus 0.1 --max-new-tokens 256 \
  --out runs/huginn_loopcd_gsm8k.json
```

Full launch templates for each benchmark are in
[`scripts/launch_examples/`](scripts/launch_examples/).

---

## What's here

```
LoopCD_official/
├── loopcd/                 method + eval code
│   ├── huginn/              Huginn-0125 driver (baseline / iter-CD)
│   ├── ouro/                Ouro-1.4B driver
│   └── eval/                task helpers (GSM8K, MATH-500, HE+, MBPP+, StrategyQA)
├── analysis/               Figure 3, HARD/EASY ablation
├── scripts/                setup + launch templates
├── third_party/            third-party dep install instructions (evalplus, lm-eval)
├── docs/                   install / usage / reproduce notes
└── requirements.txt        pinned Python deps
```

---

## Supported models

- **Huginn-0125** (`tomg-group-umd/huginn-0125`, 3.5B, r=32) — depth-recurrent LM
- **Ouro-1.4B** (`ByteDance/Ouro-1.4B`) — universal transformer style, `total_ut_steps=4`

Models are downloaded on-the-fly by HuggingFace `transformers`; no
checkpoint files are shipped with this repo.

---

## Supported benchmarks

| task | script |
|---|---|
| GSM8K (arithmetic) | `loopcd/eval/eval_gsm8k.py` |
| MATH-500 (math) | `loopcd/eval/eval_math500.py` |
| HumanEval / HumanEval+ (code) | `loopcd/eval/eval_humaneval.py` |
| MBPP / MBPP+ (code) | `loopcd/eval/eval_mbpp.py` |
| StrategyQA (commonsense QA) | `loopcd/eval/eval_strategyqa.py` |

MATH-500 uses `lm-evaluation-harness`'s Minerva scorer, and HE+/MBPP+
use `evalplus` — see [`third_party/README.md`](third_party/README.md)
for installation.

---

## Reproducing the paper

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for exact commands and
canonical hyperparameters per task.

Paper Table 2 headline numbers (Huginn-0125):

| task | flex (base) | strict / plus |
|---|---|---|
| GSM8K       | 36.62 | 26.46 |
| MATH-500    | 15.60 | 15.20 |
| HumanEval+  | 30.49 | 27.44 |
| MBPP+       | 42.06 | 36.77 |
| StrategyQA  |       | 55.24 |

Full grid of `K_a × λ` sweeps and multi-seed nondet runs used in the
paper appendices is documented in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).
