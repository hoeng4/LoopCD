"""GSM8K task helper for iter-CD pipeline.

Exposes: STOPS, fmt_prompt(q[,tok]), truncate_at_stops(c[,stops]),
score(text, doc), setup(tok), load_docs().
Uses regex-based answer extraction (#### N or "The answer is N").
"""
from __future__ import annotations
import sys, re



FEWSHOTS = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove "
     "today. After they are done, there will be 21 trees. How many trees did the "
     "grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more "
     "were planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars "
     "are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces "
     "do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had "
     "32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
]
STOPS = ["\n\n", "Q:", "</s>"]
_STRICT_RE = re.compile(r"The answer is (\-?[0-9\.\,]+)\.")
_FLEX_RE   = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def fmt_prompt(q):
    parts = [f"Q: {qq}\nA: {aa}" for qq, aa in FEWSHOTS]
    parts.append(f"Q: {q}\nA:")
    return "\n\n".join(parts)


def _normalise(t): return re.sub(r"[,\$]|\.$", "", t).strip()


def extract_strict(comp):
    m = _STRICT_RE.search(comp)
    return _normalise(m.group(1)) if m else None


def extract_flex(comp):
    ms = list(_FLEX_RE.finditer(comp))
    if not ms: return None
    last = ms[-1]
    return _normalise(last.group(1) or last.group(2) or "")


def truncate_at_stops(comp):
    for s in STOPS:
        if s and s in comp:
            comp = comp.split(s)[0]
    return comp


# ---- unified task-helper interface (matches eval_math500.py) ----

def score(text, doc_or_gold):
    """Returns {'strict_ok', 'flex_ok', ...}. Accepts either doc dict (with
    'gold' key) or bare gold string for backward compat."""
    if isinstance(doc_or_gold, dict):
        gold = doc_or_gold["gold"]
    else:
        gold = doc_or_gold
    comp = truncate_at_stops(text)
    gold_norm = _normalise(gold)
    sp = extract_strict(comp)
    fp = extract_flex(comp)
    return {
        "strict_ok": (sp == gold_norm),
        "flex_ok":   (fp == gold_norm),
        "strict_pred": sp,
        "flex_pred":   fp,
        "gold_extracted": gold_norm,
    }


def setup(tokenizer):
    """No-op for text tasks; here for interface uniformity."""
    return STOPS


def fmt_prompt_with_tok(question, tokenizer=None):
    """Variant taking optional tokenizer (ignored for gsm8k)."""
    return fmt_prompt(question)


def load_docs(limit=None):
    """Load GSM8K test split as [{doc_id, question, gold}, ...]."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    out = []
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        out.append({
            "doc_id": i,
            "question": ex["question"],
            # Gold is the part after ####, normalized.
            "gold": ex["answer"].split("####")[-1].strip(),
        })
    return out
