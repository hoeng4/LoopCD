"""Iter-axis CD evaluation via monkey-patched `model.generate`.

Loads the benchmark dataset directly via the task helper (no baseline-json
intermediate); outputs neutral fields {completion, flex_ok, strict_ok} so the
same script handles both baseline (`--ka-mode full`) and CD (`--ka-mode fixed`).
Comparison between two run JSONs (by doc_id) is a downstream analysis step.

Supports: --task {gsm8k, math500, humaneval, mbpp}
          --ka-mode {full, fixed}
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

_here = Path(__file__).parent
_helpers = _here.parent / "helpers"
_spec = importlib.util.spec_from_file_location("cd_generate", str(_here / "cd_generate.py"))
cd_generate = importlib.util.module_from_spec(_spec); sys.modules["cd_generate"] = cd_generate
_spec.loader.exec_module(cd_generate)


def _load_task_helper(task: str):
    mapping = {"gsm8k": "eval_gsm8k.py", "math500": "eval_math500.py",
               "humaneval": "eval_humaneval.py", "mbpp": "eval_mbpp.py",
               "strategyqa": "eval_strategyqa.py"}
    if task not in mapping:
        raise ValueError(f"unknown --task: {task}")
    spec = importlib.util.spec_from_file_location(f"_task_{task}", str(_helpers / mapping[task]))
    mod = importlib.util.module_from_spec(spec); sys.modules[f"_task_{task}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _shard_key(did):
    if isinstance(did, int): return did
    s = str(did)
    if "/" in s:
        try: return int(s.rsplit("/", 1)[1])
        except: pass
    try: return int(s)
    except: return abs(hash(s)) & 0x7FFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tomg-group-umd/huginn-0125")
    ap.add_argument("--task", choices=["gsm8k", "math500", "humaneval", "mbpp", "strategyqa"], default="gsm8k")
    ap.add_argument("--ka-mode", choices=["full", "fixed"], default="fixed",
                    help="full = pure HF model.generate (no CD); fixed = cd_generate.install + generate.")
    ap.add_argument("--k-amateur", type=int, default=8)
    ap.add_argument("--k-expert", type=int, default=32)
    ap.add_argument("--lambda-cd", type=float, default=0.3)
    ap.add_argument("--alpha-plaus", type=float, default=0.1)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=99999)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--include-doc-ids-file", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[config] task={args.task}  ka_mode={args.ka_mode}  K_a={args.k_amateur}  K_e={args.k_expert}  "
          f"lambda_cd={args.lambda_cd}  alpha_plaus={args.alpha_plaus}", flush=True)

    task_helper = _load_task_helper(args.task)

    # ----- doc list (always from the task helper) -----
    sub = task_helper.load_docs()
    print(f"[dataset] {args.task}: n={len(sub)}", flush=True)

    if args.include_doc_ids_file:
        include = {x.strip() for x in open(args.include_doc_ids_file) if x.strip()}
        sub = [r for r in sub if str(r["doc_id"]) in include]
        print(f"[include-filter] kept {len(sub)} docs", flush=True)
    sub = [r for r in sub if _shard_key(r["doc_id"]) % args.shard_count == args.shard_idx]
    if args.limit and len(sub) > args.limit:
        sub = sub[: args.limit]

    # ----- resume from existing JSON -----
    rows = []; done_ids = set()
    if os.path.exists(args.out):
        try:
            existing = json.load(open(args.out))
            rows = existing.get("rows", [])
            done_ids = {r["doc_id"] for r in rows}
            print(f"[resume] {len(done_ids)} docs already complete", flush=True)
        except Exception as e:
            print(f"[resume] WARNING: {e}; starting fresh", flush=True); rows = []
    sub = [r for r in sub if r["doc_id"] not in done_ids]
    print(f"[shard {args.shard_idx}/{args.shard_count}] remaining n={len(sub)}", flush=True)
    if not sub:
        print("[done-already] nothing to do."); return

    # ----- model -----
    print(f"[load] {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.initialize_state = lambda e, scale=1.0: torch.zeros_like(e)
    model.eval()

    task_stops = task_helper.setup(tokenizer) if hasattr(task_helper, "setup") else task_helper.STOPS

    if args.ka_mode == "fixed":
        cd_generate.install(model, k_amateur=args.k_amateur,
                            lambda_cd=args.lambda_cd, alpha_plaus=args.alpha_plaus)

    # Config matches the canonical recipe from huginn-0125 model card:
    #   https://huggingface.co/tomg-group-umd/huginn-0125
    cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        stop_strings=task_stops,
        use_cache=True,
        do_sample=False, temperature=None, top_k=None, top_p=None, min_p=None,
        return_dict_in_generate=True,
        eos_token_id=65505, bos_token_id=65504, pad_token_id=65509,
    )

    # ----- decode loop -----
    n_flex = sum(1 for r in rows if r.get("flex_ok"))
    n_strict = sum(1 for r in rows if r.get("strict_ok"))
    t_total = sum(r.get("dt_s", 0.0) for r in rows)

    for i, r in enumerate(sub):
        try:
            prompt = task_helper.fmt_prompt(r["question"], tokenizer)
        except TypeError:
            prompt = task_helper.fmt_prompt(r["question"])
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        t0 = time.time()
        out = model.generate(ids, cfg, tokenizer=tokenizer, num_steps=args.k_expert)
        dt = time.time() - t0
        text = tokenizer.decode(out.sequences[0][ids.shape[1]:], skip_special_tokens=True)
        try:
            comp = task_helper.truncate_at_stops(text, task_stops)
        except TypeError:
            comp = task_helper.truncate_at_stops(text)
        s = task_helper.score(comp, r)
        flex_ok = bool(s["flex_ok"]); strict_ok = bool(s["strict_ok"])

        n_flex += int(flex_ok); n_strict += int(strict_ok); t_total += dt

        rows.append({
            "doc_id": r["doc_id"],
            "completion": comp,
            "flex_ok": flex_ok, "strict_ok": strict_ok,
            "n_gen": out.sequences.shape[1] - ids.shape[1],
            "dt_s": dt,
        })

        print(f"[{i+1:>4}/{len(sub)}] doc={str(r['doc_id'])[:12]:>12} "
              f"flex={int(flex_ok)} strict={int(strict_ok)} "
              f"n_gen={rows[-1]['n_gen']} dt={dt:.1f}s", flush=True)

        # Save every 10 docs (and on the last doc); retry on NAS file-lock transients.
        if (i + 1) % 10 == 0 or (i + 1) == len(sub):
            payload = {
                "config": vars(args),
                "summary": {"n_done": len(rows), "n_flex": n_flex, "n_strict": n_strict, "t_total_s": t_total},
                "rows": rows,
            }
            for _retry in range(5):
                try:
                    with open(args.out, "w") as _f:
                        json.dump(payload, _f, indent=2)
                    break
                except (BlockingIOError, OSError) as _e:
                    time.sleep(2)
            else:
                print(f"[warn] save failed after 5 retries — last error: {_e}", flush=True)

    if args.ka_mode == "fixed":
        cd_generate.uninstall(model)

    print(f"\n[done] n={len(sub)}  t={t_total:.1f}s")
    print(f"  flex={n_flex}/{len(rows)}={100*n_flex/max(1,len(rows)):.2f}%  "
          f"strict={n_strict}/{len(rows)}={100*n_strict/max(1,len(rows)):.2f}%")


if __name__ == "__main__":
    main()
