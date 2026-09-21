"""MBPP+ evaluator. Same interface as gsm8k/math500/humaneval helpers.

Scoring uses evalplus.eval.untrusted_check on the MBPP+ extended test suite,
exposed via the existing eval_code_common._run_check (which already
handles both `humaneval` and `mbpp` benches based on problem['task_id']).

`strict_ok` = passed plus suite; `flex_ok` = passed base suite. Matches HumanEval+.
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


STOPS: list[str] = list(_he_pf.BASE_EOS)


def setup(tokenizer) -> list[str]:
    global STOPS
    if tokenizer.chat_template is None:
        STOPS = list(_he_pf.BASE_EOS) + list(_he_pf.HE_EXTRA_DIRECT)
    else:
        STOPS = list(_he_pf.BASE_EOS) + list(_he_pf.HE_EXTRA_CHAT)
    return STOPS


def fmt_prompt(question: str, tokenizer=None) -> str:
    if tokenizer is None:
        return question
    prompt, _eos = _he_pf._build_prompt(tokenizer, question)
    return prompt


def truncate_at_stops(comp: str, stops: list[str] | None = None) -> str:
    return _he_pf._trim_eos(comp, stops or STOPS)


def score(text: str, doc: dict) -> dict[str, Any]:
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
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import get_groundtruth
    problems = get_mbpp_plus()
    hashcode = get_mbpp_plus_hash()
    gt_all = get_groundtruth(problems, hashcode, MBPP_OUTPUT_NOT_NONE_TASKS)
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
