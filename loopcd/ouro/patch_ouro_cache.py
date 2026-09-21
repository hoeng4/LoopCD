"""Patch Ouro's cached `modeling_ouro.py` for transformers >= 4.54.

Mirrors `loopcd/huginn/patch_huginn_cache.py`. Ouro has the same root cause: `transformers>=4.54` made
`Cache.key_cache` / `Cache.value_cache` read-only properties on the base
`Cache` class, and `UniversalTransformerCache.__init__` assigns to them
directly → AttributeError. Fix: shadow them at class level.

Usage:
    python -m loopcd.ouro.patch_ouro_cache
    python -m loopcd.ouro.patch_ouro_cache --model ByteDance/Ouro-1.4B
Idempotent; backs up to <file>.bak_pre_patch before writing.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path


OLD_BLOCK = (
    "class UniversalTransformerCache(Cache):\n"
    "    \"\"\"Cache implementation that supports Ouro's multi-step Universal Transformer loops.\"\"\"\n"
    "\n"
    "    def __init__(self, max_cache_size: Optional[int] = None):\n"
)
NEW_BLOCK = (
    "class UniversalTransformerCache(Cache):\n"
    "    \"\"\"Cache implementation that supports Ouro's multi-step Universal Transformer loops.\"\"\"\n"
    "\n"
    "    # transformers>=4.54 made key_cache/value_cache read-only properties on\n"
    "    # Cache; shadow them at the class level so __init__ can assign lists.\n"
    "    key_cache = None\n"
    "    value_cache = None\n"
    "\n"
    "    # transformers>=4.54 Cache.get_mask_sizes delegates to self.layers[idx]\n"
    "    # but UniversalTransformerCache uses flat key_cache[step*L+layer], with\n"
    "    # self.layers always empty. Override to return (seen+new, 0) directly.\n"
    "    def get_mask_sizes(self, cache_position, layer_idx):\n"
    "        new_tokens = cache_position.shape[-1] if cache_position is not None else 1\n"
    "        return self._seen_tokens + new_tokens, 0\n"
    "\n"
    "    def __init__(self, max_cache_size: Optional[int] = None):\n"
)


def _find_files(model: str, hf_home: Path) -> list[Path]:
    org, name = model.split("/")
    paths = []
    hub_dir = hf_home / "hub" / f"models--{org}--{name}"
    if hub_dir.exists():
        paths.extend(hub_dir.rglob("modeling_ouro.py"))
    mod_dir = hf_home / "modules" / "transformers_modules" / org / name
    if mod_dir.exists():
        paths.extend(mod_dir.rglob("modeling_ouro.py"))
    return [p.resolve() for p in paths if p.is_file()]


def _patch_one(path: Path, dry_run: bool = False) -> str:
    src = path.read_text()
    if NEW_BLOCK in src:
        return "already-patched"
    if OLD_BLOCK not in src:
        return "unknown-format"
    if dry_run:
        return "would-patch"
    bak = path.with_suffix(path.suffix + ".bak_pre_patch")
    if not bak.exists():
        bak.write_text(src)
    path.write_text(src.replace(OLD_BLOCK, NEW_BLOCK))
    return "patched"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ByteDance/Ouro-1.4B")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    hf_home = Path(args.hf_home).expanduser().resolve()
    print(f"[config] model={args.model}  HF_HOME={hf_home}  dry-run={args.dry_run}")
    files = _find_files(args.model, hf_home)
    if not files:
        print(f"[error] no modeling_ouro.py found. Run AutoModelForCausalLM.from_pretrained first.")
        sys.exit(2)
    print(f"[found] {len(files)} file(s):")
    for f in files: print(f"  - {f}")

    n_patched = n_skipped = n_unknown = 0
    for f in files:
        status = _patch_one(f, dry_run=args.dry_run)
        print(f"  [{status:>16s}] {f}")
        if status in ("patched", "would-patch"): n_patched += 1
        elif status == "already-patched": n_skipped += 1
        elif status == "unknown-format": n_unknown += 1
    print(f"[summary] patched={n_patched}  already={n_skipped}  unknown={n_unknown}")
    if n_unknown:
        sys.exit(1)


if __name__ == "__main__":
    main()
