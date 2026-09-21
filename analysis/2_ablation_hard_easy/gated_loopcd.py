"""Apply LoopCD only at tokens passing a hardness / type gate.

The gate is computed online at each decode step from the same 32-iteration
trajectory the method already needs, so it uses no information the decoder
would not have: HARD iff ``d = max_i m_i - m_r >= eps`` for the final-iteration
top-1 token, and the token type comes from that token's surface form. Where the
gate fires the next token is picked by the LoopCD blend, elsewhere greedily.

Modes
-----
``full``                            LoopCD everywhere (matches the main results)
``hard_only`` / ``easy_only``       gate on hardness
``hard_num`` / ``easy_num``         hardness x numeric tokens
``hard_op`` / ``easy_op``           hardness x arithmetic operators

Usage
-----
    python -m analysis.2_ablation_hard_easy.gated_loopcd \\
        --mode hard_only --shard-idx 0 --shard-count 8 \\
        --out runs/gated/hard_only_s0.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..tasks import load_task_helper, shard_key

_OPERATORS = {"+", "-", "*", "/", "=", "x", "×", "÷", "%"}
_PUNCT = {".", ",", ":", ";", "!", "?", "(", ")"}


def token_type(text: str) -> str:
    t = text.strip()
    if re.fullmatch(r"\d+", t):
        return "NUM"
    if t in _OPERATORS:
        return "OP"
    if t in _PUNCT:
        return "PUNCT"
    if t.isalpha():
        return "WORD"
    return "OTHER"


def gate_fires(mode: str, hard: bool, ttype: str) -> bool:
    return {
        "full": True,
        "hard_only": hard,
        "easy_only": not hard,
        "hard_num": hard and ttype == "NUM",
        "easy_num": (not hard) and ttype == "NUM",
        "hard_op": hard and ttype == "OP",
        "easy_op": (not hard) and ttype == "OP",
    }[mode]


def cd_pick(logits_per_iter, k_amateur, lambda_cd, alpha_plaus):
    """LoopCD blend, identical to ``loopcd.huginn.cd_generate``."""
    log_p_e = F.log_softmax(logits_per_iter[-1], dim=-1)
    log_p_a = F.log_softmax(logits_per_iter[k_amateur - 1], dim=-1)
    thr = float(log_p_e.max().item()) + math.log(alpha_plaus)
    score = log_p_e - lambda_cd * log_p_a
    score = torch.where(log_p_e >= thr, score, torch.full_like(score, float("-inf")))
    return int(score.argmax().item())


def hardness(logits_per_iter, winner):
    z_y = logits_per_iter[:, winner].clone()
    other = logits_per_iter.clone()
    other[:, winner] = float("-inf")
    m = z_y - other.max(dim=-1).values
    return float(m.max().item() - m[-1].item())


@torch.no_grad()
def decode_doc(model, tokenizer, prompt, stops, mode, *, r, k_amateur,
               lambda_cd, alpha_plaus, eps, max_new, eos_id):
    device = model.device
    HDCache = sys.modules[type(model).__module__].HuginnDynamicCache  # type: ignore
    cache = HDCache()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    gen_ids: list[int] = []
    n_gate = n_fire = 0

    def pick(logits_per_iter):
        nonlocal n_gate, n_fire
        greedy = int(logits_per_iter[-1].argmax().item())
        d = hardness(logits_per_iter, greedy)
        if not gate_fires(mode, d >= eps, token_type(tokenizer.decode([greedy]))):
            return greedy
        n_gate += 1
        chosen = cd_pick(logits_per_iter, k_amateur, lambda_cd, alpha_plaus)
        n_fire += chosen != greedy
        return chosen

    # prefill
    embeds, prelude_idx = model.embed_inputs(input_ids, past_key_values=cache, use_cache=True)
    x = torch.zeros_like(embeds)
    block_idx = prelude_idx
    latents = []
    for step in range(r):
        x, block_idx, _ = model.iterate_one_step(
            embeds, x, block_idx=block_idx, past_key_values=cache, current_step=step)
        latents.append(x[:, -1:, :].clone())
    expert = model.predict_from_latents(x, past_key_values=cache).logits[0, -1, :].float()
    last_pos = torch.tensor([prompt_len - 1], device=device)
    per_iter = [
        model.predict_from_latents(latents[k], past_key_values=cache,
                                   cache_position=last_pos).logits[0, -1, :].float()
        for k in range(r - 1)
    ] + [expert]
    model.predict_from_latents(x[:, -1:, :], past_key_values=cache, cache_position=last_pos)
    gen_ids.append(pick(torch.stack(per_iter, 0)))

    pos = prompt_len
    while len(gen_ids) < max_new and gen_ids[-1] != eos_id:
        if any(s in tokenizer.decode(gen_ids, skip_special_tokens=True) for s in stops):
            break
        cp = torch.tensor([pos], device=device)
        embeds, _ = model.embed_inputs(
            torch.tensor([[gen_ids[-1]]], device=device),
            past_key_values=cache, use_cache=True, cache_position=cp)
        x = torch.zeros_like(embeds)
        block_idx = prelude_idx
        latents = []
        for step in range(r):
            x, block_idx, _ = model.iterate_one_step(
                embeds, x, block_idx=block_idx, past_key_values=cache,
                cache_position=cp, current_step=step)
            latents.append(x.clone())
        per_iter = [
            model.predict_from_latents(h, past_key_values=cache,
                                       cache_position=cp).logits[0, -1, :].float()
            for h in latents
        ]
        gen_ids.append(pick(torch.stack(per_iter, 0)))
        pos += 1

    return gen_ids, n_gate, n_fire


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="tomg-group-umd/huginn-0125")
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "strategyqa"])
    ap.add_argument("--mode", required=True,
                    choices=["full", "hard_only", "easy_only",
                             "hard_num", "easy_num", "hard_op", "easy_op"])
    ap.add_argument("--k-amateur", type=int, default=8)
    ap.add_argument("--lambda-cd", type=float, default=0.3)
    ap.add_argument("--alpha-plaus", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=1.5)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--eos-id", type=int, default=65505)
    ap.add_argument("--limit", type=int, default=99999)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    task = load_task_helper(args.task)
    docs = task.load_docs()
    if args.limit < len(docs):
        docs = docs[: args.limit]
    docs = [d for d in docs
            if shard_key(d["doc_id"]) % args.shard_count == args.shard_idx]

    rows, done = [], set()
    if os.path.exists(args.out):
        try:
            rows = json.load(open(args.out))["rows"]
            done = {r["doc_id"] for r in rows}
        except Exception:
            rows = []
    todo = [d for d in docs if d["doc_id"] not in done]
    print(f"[{args.mode} shard {args.shard_idx}/{args.shard_count}] "
          f"{len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.initialize_state = lambda e, scale=1.0: torch.zeros_like(e)
    model.eval()
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("model landed on CPU -- refusing to run")
    stops = task.setup(tokenizer) if hasattr(task, "setup") else task.STOPS

    for i, doc in enumerate(todo):
        t0 = time.time()
        try:
            prompt = task.fmt_prompt(doc["question"], tokenizer)
        except TypeError:
            prompt = task.fmt_prompt(doc["question"])
        gen_ids, n_gate, n_fire = decode_doc(
            model, tokenizer, prompt, stops, args.mode, r=args.r,
            k_amateur=args.k_amateur, lambda_cd=args.lambda_cd,
            alpha_plaus=args.alpha_plaus, eps=args.eps, max_new=args.max_new,
            eos_id=args.eos_id)
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        try:
            text = task.truncate_at_stops(text, stops)
        except TypeError:
            text = task.truncate_at_stops(text)
        score = task.score(text, doc)
        rows.append({"doc_id": doc["doc_id"], "completion": text,
                     "flex_ok": bool(score["flex_ok"]),
                     "strict_ok": bool(score["strict_ok"]),
                     "n_gen": len(gen_ids), "n_gate": n_gate, "n_fire": n_fire,
                     "dt_s": round(time.time() - t0, 2)})
        print(f"[{i + 1}/{len(todo)}] doc={doc['doc_id']} "
              f"flex={int(score['flex_ok'])} gate={n_gate} fire={n_fire} "
              f"dt={rows[-1]['dt_s']:.1f}s", flush=True)
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            json.dump({"config": vars(args), "rows": rows}, open(args.out, "w"))
    json.dump({"config": vars(args), "rows": rows}, open(args.out, "w"))

    n_flex = sum(r["flex_ok"] for r in rows)
    n_gate = sum(r["n_gate"] for r in rows)
    n_gen = sum(r["n_gen"] for r in rows)
    print(f"[done] {args.mode}: flex {n_flex}/{len(rows)} = "
          f"{100 * n_flex / max(1, len(rows)):.2f}%   "
          f"gated {100 * n_gate / max(1, n_gen):.1f}% of tokens")


if __name__ == "__main__":
    main()
