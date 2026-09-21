"""Record, for every generated token, the per-iteration logit trajectory needed
to label it HARD or EASY (paper Appendix C), plus the candidate pools used by
the token-replacement experiment of Table 1.

The model is decoded **greedily** (argmax of the final recurrent iteration),
i.e. exactly the `--ka-mode full` baseline, and at every decoding step the LM
head is additionally read at *every* recurrent iteration so that

    m_i = z_i(y*) - max_{v != y*} z_i(v)

is available for i = 1..r.  Only the summary statistics are stored:

    ds      d = max_i m_i - m_r          (Appendix C confidence drop)
    peaks   argmax_i m_i                 (iteration of peak commitment)
    pools   top-k token ids at iter r    (the "top-k tokens" of Table 1)
    pools_a top-k token ids at the peak iteration (early/amateur candidates)

Reading the LM head r times per token costs ~10x a plain greedy decode, so this
is a one-off analysis pass, not part of LoopCD inference.

Usage
-----
    python -m analysis.capture --task gsm8k --limit 1319 \
        --shard-idx 0 --shard-count 8 --out captures/gsm8k_s0.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .tasks import load_task_helper, shard_key


@torch.no_grad()
def greedy_capture(model, tokenizer, prompt, stops, *, r=32, max_new=256,
                   topk=10, cache_strategy="full"):
    """Greedy decode + per-iteration LM-head readout.

    Returns ``(gen_ids, ds, pools, peaks, pools_a)``.
    """
    import sys
    HDCache = sys.modules[type(model).__module__].HuginnDynamicCache  # type: ignore
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cache = HDCache(lookup_strategy=cache_strategy)

    def run(step_ids, cache_position):
        freqs_cis = (model.freqs_cis[:, : step_ids.shape[1]] if cache_position is None
                     else model.freqs_cis[:, cache_position])
        emb = model.transformer.wte(step_ids)
        if model.emb_scale != 1:
            emb = emb * model.emb_scale
        block_idx = torch.tensor(-1, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.prelude:
            block_idx += 1
            emb = block(emb, freqs_cis, block_idx, None, cache)
        x = model.initialize_state(emb, scale=1.0)
        latents = []
        for step in range(r):
            x, block_idx = model.core_block_forward(
                x, emb, freqs_cis, None, cache, block_idx, current_step=step)
            latents.append(x.clone())
        # The final call (iter r) writes the canonical coda KV, matching the
        # production forward pass.
        return torch.stack([
            model.predict_from_latents(
                h, attention_mask=None, position_ids=None,
                cache_position=cache_position, past_key_values=cache,
            ).logits[:, -1, :].float().squeeze(0)
            for h in latents], 0)

    def summarise(Z):
        final = Z[-1]
        winner = int(final.argmax().item())
        z_y = Z[:, winner].clone()
        Z_other = Z.clone()
        Z_other[:, winner] = float("-inf")
        m = z_y - Z_other.max(dim=-1).values
        peak = int(m.argmax().item())
        d = float(m.max().item() - m[-1].item())
        pool = [int(v) for v in torch.topk(final, topk).indices.cpu().tolist()]
        pool_a = [int(v) for v in torch.topk(Z[peak], topk).indices.cpu().tolist()]
        return d, pool, peak, pool_a, winner

    gen, ds, pools, peaks, pools_a = [], [], [], [], []

    def push(Z):
        d, pool, peak, pool_a, winner = summarise(Z)
        ds.append(d); pools.append(pool); peaks.append(peak)
        pools_a.append(pool_a); gen.append(winner)
        return winner

    last = push(run(ids, None))
    pos = ids.shape[1]
    for _ in range(1, max_new):
        if last == tokenizer.eos_token_id:
            break
        last = push(run(torch.tensor([[last]], device=device),
                        torch.tensor([pos], device=device)))
        pos += 1
        if any(s in tokenizer.decode(gen, skip_special_tokens=True) for s in stops):
            break
    return gen, ds, pools, peaks, pools_a


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tomg-group-umd/huginn-0125")
    ap.add_argument("--task", default="gsm8k",
                    choices=["gsm8k", "strategyqa"])
    ap.add_argument("--limit", type=int, default=1319)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--topk", type=int, default=10)
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
    if os.path.exists(args.out):                       # resume, never restart
        try:
            rows = json.load(open(args.out))["rows"]
            done = {r["doc_id"] for r in rows}
            print(f"[resume] {len(done)} docs already captured", flush=True)
        except Exception as exc:
            print(f"[resume] starting fresh ({exc})", flush=True)
            rows = []
    todo = [d for d in docs if d["doc_id"] not in done]
    print(f"[shard {args.shard_idx}/{args.shard_count}] todo={len(todo)}", flush=True)
    if not todo:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.initialize_state = lambda e, scale=1.0: torch.zeros_like(e)
    model.eval()
    stops = task.setup(tokenizer) if hasattr(task, "setup") else task.STOPS

    for i, doc in enumerate(todo):
        t0 = time.time()
        try:
            prompt = task.fmt_prompt(doc["question"], tokenizer)
        except TypeError:
            prompt = task.fmt_prompt(doc["question"])
        gen, ds, pools, peaks, pools_a = greedy_capture(
            model, tokenizer, prompt, stops, r=args.r,
            max_new=args.max_new, topk=args.topk)
        text = tokenizer.decode(gen, skip_special_tokens=True)
        try:
            text = task.truncate_at_stops(text, stops)
        except TypeError:
            text = task.truncate_at_stops(text)
        score = task.score(text, doc)
        rows.append({
            "doc_id": doc["doc_id"], "n": len(gen),
            "flex_ok": bool(score["flex_ok"]), "strict_ok": bool(score["strict_ok"]),
            "gen_ids": gen, "ds": [round(float(x), 4) for x in ds],
            "pools": pools, "peaks": peaks, "pools_a": pools_a,
            "dt_s": time.time() - t0,
        })
        print(f"[{i + 1}/{len(todo)}] doc={doc['doc_id']} n={len(gen)} "
              f"ok={int(score['flex_ok'])} dt={rows[-1]['dt_s']:.1f}s", flush=True)
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            json.dump({"config": vars(args), "rows": rows}, open(args.out, "w"))
    json.dump({"config": vars(args), "rows": rows}, open(args.out, "w"))
    print("[done]", args.out, flush=True)


if __name__ == "__main__":
    main()
