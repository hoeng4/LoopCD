"""StrategyQA task helper for iter-CD pipeline.

Matches PruneCD's evaluation recipe verbatim:
  - 6-shot CoT prompt, "Q:/A:" format ending with "So the answer is yes/no."
  - Stop words: ["Q:", "\n\n##", "\n"] (paper uses tail-length 3 substring match;
    we approximate with the same stop list via HF stop_strings)
  - Data: official AI2 strategyqa_train.json (2290 examples, all have answers)
  - Score: extract "yes"/"no" after "answer is", map to boolean, compare to gold

Exposes the unified task-helper interface used by Huginn iter-CD scripts:
  STOPS, fmt_prompt, truncate_at_stops, score, setup, load_docs
"""
from __future__ import annotations
import json, os
from pathlib import Path

ANSWER_TRIGGER = "So the answer is"
SHORT_ANSWER_TRIGGER = "answer is"
STOPS = ["Q:", "\n\n##", "\n"]

_HERE = Path(__file__).parent
_DATA = _HERE / "data" / "strategyqa_train.json"


_QA = [
    ("Do hamsters provide food for any animals?",
     "Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.",
     "yes"),
    ("Could Brooke Shields succeed at University of Pennsylvania?",
     "Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.",
     "yes"),
    ("Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?",
     "Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5.",
     "no"),
    ("Yes or no: Is it common to see frost during some college commencements?",
     "College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.",
     "yes"),
    ("Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?",
     "The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.",
     "no"),
    ("Yes or no: Would a pear sink in water?",
     "The density of a pear is about 0.6 g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float.",
     "no"),
]


def _demo_text():
    parts = []
    for q, chain, ans in _QA:
        parts.append(f"Q: {q}\nA: {chain} {ANSWER_TRIGGER} {ans}.")
    return "\n\n".join(parts) + "\n\n"


def fmt_prompt(question, tokenizer=None):
    return _demo_text() + f"Q: {question}\nA:"


def truncate_at_stops(comp, stops=None):
    stops = stops if stops is not None else STOPS
    for s in stops:
        if s and s in comp:
            comp = comp.split(s)[0]
    return comp


def _clean_answer(model_pred):
    """Mirror PruneCD clean_answer with random_guess=False semantics.

    Returns "yes" / "no" / None (None = parse failure, counted as wrong).
    """
    s = model_pred.lower()
    if "thus, yes." in s:
        return "yes"
    if SHORT_ANSWER_TRIGGER.lower() in s:
        pred = s.split(SHORT_ANSWER_TRIGGER.lower())[1].split(".")[0].strip()
    else:
        return None
    if pred not in ("yes", "no"):
        return None
    return pred


def _gold_to_str(gold):
    if isinstance(gold, bool):
        return "yes" if gold else "no"
    g = str(gold).strip().lower()
    if g in ("true", "yes"): return "yes"
    if g in ("false", "no"): return "no"
    return g


def score(text, doc_or_gold):
    if isinstance(doc_or_gold, dict):
        gold = doc_or_gold["gold"]
    else:
        gold = doc_or_gold
    comp = truncate_at_stops(text)
    pred = _clean_answer(comp)
    gold_norm = _gold_to_str(gold)
    ok = (pred is not None) and (pred == gold_norm)
    return {
        "strict_ok": ok,
        "flex_ok": ok,
        "strict_pred": pred,
        "flex_pred": pred,
        "gold_extracted": gold_norm,
    }


def setup(tokenizer):
    return STOPS


def load_docs(limit=None):
    if not _DATA.exists():
        raise FileNotFoundError(
            f"Missing {_DATA}. Run: cd {_HERE/'data'} && "
            "curl -sSL -o strategyqa_dataset.zip "
            "https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip && "
            "unzip -q -o strategyqa_dataset.zip")
    items = json.load(open(_DATA))
    out = []
    for i, ex in enumerate(items):
        if limit is not None and i >= limit:
            break
        out.append({
            "doc_id": ex.get("qid", i),
            "question": ex["question"],
            "gold": _gold_to_str(ex["answer"]),
        })
    return out
