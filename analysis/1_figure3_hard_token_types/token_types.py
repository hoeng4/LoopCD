"""Figure 3 -- which tokens are labelled HARD.

Reads the captures produced by :mod:`analysis.capture` and reports, over the
generated tokens of the documents the model answers correctly:

* the most frequent HARD token types, split by whether the document ended up
  correct or wrong (the two bars of Figure 3);
* the share of each surface category (numeric, operator, punctuation, word)
  among HARD tokens, next to its share among all generated tokens.

Usage
-----
    python -m analysis.1_figure3_hard_token_types.token_types \\
        --captures "captures/gsm8k/shard_*.json" --task gsm8k
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter

from ..hard_tokens import EPS_DEFAULT

_NUMERIC = re.compile(r"^-?[0-9]+([.,][0-9]+)?$")
_OPERATORS = {"+", "-", "*", "/", "=", "x", "×", "÷", "%", "$"}
_PUNCT = {".", ",", ":", ";", "!", "?", "(", ")"}


def token_category(text: str) -> str:
    t = text.strip()
    if not t:
        return "WHITESPACE"
    if _NUMERIC.match(t):
        return "NUMERIC"
    if t in _OPERATORS:
        return "OPERATOR"
    if t in _PUNCT:
        return "PUNCT"
    if t.isalpha():
        return "WORD"
    return "OTHER"


def main():
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "strategyqa"])
    ap.add_argument("--model", default="tomg-group-umd/huginn-0125")
    ap.add_argument("--metric", default="flex", choices=["flex", "strict"])
    ap.add_argument("--eps", type=float, default=EPS_DEFAULT)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = []
    for path in sorted(glob.glob(args.captures)):
        with open(path) as fh:
            rows += json.load(fh)["rows"]

    key = args.metric + "_ok"
    hard_by_outcome = {True: Counter(), False: Counter()}
    cat_hard, cat_all = Counter(), Counter()
    n_tok = n_hard = 0

    for row in rows:
        correct = bool(row[key])
        ids = row["gen_ids"][: row["n"]]
        for i, token_id in enumerate(ids):
            text = tokenizer.decode([token_id])
            category = token_category(text)
            is_hard = row["ds"][i] >= args.eps
            n_tok += 1
            cat_all[category] += 1
            if is_hard:
                n_hard += 1
                cat_hard[category] += 1
                hard_by_outcome[correct][text] += 1

    print(f"documents {len(rows)}   generated tokens {n_tok}   "
          f"HARD {n_hard} ({100 * n_hard / max(1, n_tok):.2f}%)\n")

    print(f"top-{args.top} HARD token types")
    print(f"  {'token':<14}{'wrong':>8}{'correct':>9}")
    total = hard_by_outcome[True] + hard_by_outcome[False]
    for text, _ in total.most_common(args.top):
        print(f"  {text!r:<14}{hard_by_outcome[False][text]:>8}"
              f"{hard_by_outcome[True][text]:>9}")

    print(f"\n{'category':<12}{'of HARD':>10}{'of all':>10}")
    for category, count in cat_all.most_common():
        print(f"{category:<12}{100 * cat_hard[category] / max(1, n_hard):9.1f}%"
              f"{100 * count / max(1, n_tok):9.1f}%")


if __name__ == "__main__":
    main()
