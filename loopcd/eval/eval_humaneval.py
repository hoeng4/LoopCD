"""HumanEval+ evaluator. Same interface as eval_gsm8k.py /
eval_math500.py — exposes load_docs(), fmt_prompt(q, tokenizer), score(text, doc),
truncate_at_stops(text, stops).

Scoring delegates to evalplus.eval.untrusted_check via eval_code_common
(`_run_check`, `_build_prompt`, `_trim_eos`). Reports `strict_ok` = plus suite
(extended tests), `flex_ok` = base suite (canonical tests).
"""
from __future__ import annotations
from typing import Any
import importlib.util, sys
from pathlib import Path

_here = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "_he_pf", str(_here / "eval_code_common.py"))
_he_pf = importlib.util.module_from_spec(_spec); sys.modules["_he_pf"] = _he_pf
_spec.loader.exec_module(_he_pf)


# Static fallback stops; overridden per-tokenizer by setup().
STOPS: list[str] = list(_he_pf.BASE_EOS)


def setup(tokenizer) -> list[str]:
    """Inspect tokenizer chat template; return correct stops list.

    Sets the module STOPS as a side effect so older code paths still work.
    """
    global STOPS
    if tokenizer.chat_template is None:
        STOPS = list(_he_pf.BASE_EOS) + list(_he_pf.HE_EXTRA_DIRECT)
    else:
        STOPS = list(_he_pf.BASE_EOS) + list(_he_pf.HE_EXTRA_CHAT)
    return STOPS


def fmt_prompt(question: str, tokenizer=None) -> str:
    """Build HumanEval+ prompt. `question` is the raw task_prompt from evalplus.
    Tokenizer is used to determine whether to wrap in chat template."""
    if tokenizer is None:
        return question  # caller should have called setup; bare prompt
    prompt, _eos = _he_pf._build_prompt(tokenizer, question)
    return prompt


def truncate_at_stops(comp: str, stops: list[str] | None = None) -> str:
    return _he_pf._trim_eos(comp, stops or STOPS)


def score(text: str, doc: dict) -> dict[str, Any]:
    """Run the model output through evalplus's sanitize + test harness.

    `doc` must carry `_problem` (evalplus problem dict) and `_gt` (groundtruth)
    fields that load_docs() attaches. Returns {strict_ok, flex_ok, sanitized}.
    `strict_ok` = passed plus tests; `flex_ok` = passed base tests.
    """
    from evalplus.sanitize import sanitize
    problem = doc["_problem"]; gt = doc["_gt"]
    san = sanitize(text, entrypoint=problem["entry_point"])
    if not san.strip():
        san = sanitize(problem["prompt"] + text, entrypoint=problem["entry_point"])
    _, base_ok = _he_pf._run_check("base", problem, san, gt)
    _, plus_ok = _he_pf._run_check("plus", problem, san, gt)
    return {
        "strict_ok": bool(plus_ok),
        "flex_ok":   bool(base_ok),
        "strict_pred": "",
        "flex_pred":   "",
        "sanitized": san,
    }


def load_docs(limit: int | None = None) -> list[dict]:
    """Return HumanEval+ tasks as [{doc_id, question, gold, _problem, _gt}, ...]."""
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.evaluate import get_groundtruth
    problems = get_human_eval_plus()
    hashcode = get_human_eval_plus_hash()
    gt_all = get_groundtruth(problems, hashcode, set())
    # Sort by numeric suffix of task_id ("HumanEval/0" → 0)
    def _order(tid):
        try: return int(tid.rsplit("/", 1)[1])
        except: return tid
    out = []
    for tid in sorted(problems.keys(), key=_order):
        if limit is not None and len(out) >= limit: break
        p = problems[tid]
        out.append({
            "doc_id": tid,
            "question": p["prompt"],
            "gold": p.get("canonical_solution", ""),
            "_problem": p,
            "_gt": gt_all[tid],
        })
    return out
