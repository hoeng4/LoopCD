# Install

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.5 (required for Huginn's `HuginnDynamicCache`;
  earlier versions error out on the class-level cache attributes)
- HuggingFace `transformers` ≥ 4.54.1
- CUDA-capable GPU with ≥ 40 GB VRAM (Huginn 3.5B needs ~30 GB per
  instance in bf16; smaller GPUs work with 4-bit quantization but
  numbers differ from the paper).

## Set up a venv

```bash
git clone <this-repo>
cd LoopCD_official
bash scripts/setup_env.sh
```

The setup script:
1. Creates a `venv/` in the repo root.
2. Installs pinned deps from `requirements.txt`.
3. Applies the Huginn `HuginnDynamicCache` patch
   (`loopcd/huginn/patch_huginn_cache.py`) once, in the HF cache —
   needed under transformers ≥ 4.54.
4. Applies the Ouro `UniversalTransformerCache` patch
   (`loopcd/ouro/patch_ouro_cache.py`) — token-identical to
   `use_cache=False`, needed under transformers ≥ 4.54.
5. Sets `PYTHONPATH` for the current shell.

Manual alternative:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "from loopcd.huginn import patch_huginn_cache; patch_huginn_cache.apply()"
python -c "from loopcd.ouro   import patch_ouro_cache;   patch_ouro_cache.apply()"
export PYTHONPATH=$(pwd)
```

## Third-party evaluators

MATH-500 uses `lm-evaluation-harness`'s Minerva scorer and HE+/MBPP+
use `evalplus`. See [`third_party/README.md`](../third_party/README.md).

## Sanity check

```bash
# Small smoke test: 3 GSM8K docs with LoopCD on Huginn
python -m loopcd.huginn.iter_cd_huginn_patched \
  --task gsm8k --ka-mode fixed --k-amateur 8 --lambda-cd 0.3 \
  --max-new-tokens 128 --limit 3 --shard-idx 0 --shard-count 1 \
  --out /tmp/smoke.json
```

Expect flex accuracy to print at the end; runtime ~1–2 min on a
40 GB+ GPU.
