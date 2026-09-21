# Third-party dependencies

Two vendored evaluators are used by the paper's scorers:

## `lm-evaluation-harness` — for MATH-500 (Minerva scorer)

We rely on `lm_eval.tasks.minerva_math.utils` for MATH answer
normalization and `math_verify` for equivalence checking.

```bash
pip install lm-eval==0.4.4
```

Alternative (if you need the *exact* commit the paper used):

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness \
  --branch v0.4.4 --depth 1  third_party/lm-evaluation-harness
export PYTHONPATH="$PWD/third_party/lm-evaluation-harness:$PYTHONPATH"
```

`loopcd/eval/eval_math500.py` imports via `importlib`, so vendoring
in `third_party/` works with the setup script's `PYTHONPATH` export.

## `evalplus` — for HumanEval+ and MBPP+

We rely on `evalplus.sanitize` (function extraction) and
`evalplus.eval._run_check` (test harness).

```bash
pip install evalplus==0.3.1
```

Or vendored:

```bash
git clone https://github.com/evalplus/evalplus \
  --branch v0.3.1 --depth 1  third_party/evalplus
export PYTHONPATH="$PWD/third_party/evalplus:$PYTHONPATH"
```

## Datasets

All benchmarks are pulled by HuggingFace `datasets` on first use:

| task | HF dataset |
|---|---|
| GSM8K | `gsm8k` |
| MATH-500 | `HuggingFaceH4/MATH-500` |
| HumanEval / HumanEval+ | pulled by `evalplus.data.get_human_eval_plus()` |
| MBPP / MBPP+ | pulled by `evalplus.data.get_mbpp_plus()` |
| StrategyQA | `wics/strategy-qa` (train split; official test labels are hidden) |

No dataset files are shipped with this repo. First run downloads and
caches them under `~/.cache/huggingface/`.
