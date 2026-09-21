"""MATH-500 task helper for iter-CD pipeline.

Exposes: STOPS, fmt_prompt(q[,tok]), truncate_at_stops(c[,stops]),
score(text, doc), setup(tok), load_docs().

Prompt: 4-shot Minerva paper canonical (arXiv 2206.14858 Appendix D.2,
Listing 2). Examples are concatenated as
``"Problem:\\n{p}\\n\\nSolution:\\n{sol}\\n\\n"`` — note the **newline**
between "Solution:" and the solution body. lm-eval-harness's
``minerva_math_algebra.yaml`` uses ``target_delimiter=" "`` (single space)
by default, which differs from the paper. We follow the paper because
the legacy ``huginn_math500_baseline.json`` was generated with this
format and our NEW pipeline reproduces it byte-identically for natural-
stop docs (no repetition-loop cap-hits).

Fewshot pool: lm-eval ``list_fewshot_samples()`` (same 4 examples as
paper Listing 2).

Scoring: lm-eval-harness ``process_results`` (exact_match via sympy
is_equiv + math_verify via math_verify.verify). Requires
``pip install lm-eval[math]``.
"""
from __future__ import annotations
from typing import Any


STOPS: list[str] = ["Problem:"]


# --- prompt ---

_FEWSHOT_PREFIX: str | None = None  # built lazily once per process


def _doc_to_text(problem: str) -> str:
    return "Problem:" + "\n" + problem + "\n\n" + "Solution:"


def _load_minerva_utils():
    """Load lm_eval/tasks/minerva_math/utils.py directly as a standalone module.

    Bypasses `lm_eval.tasks.__init__` which transitively pulls in heavy
    dependencies (sacrebleu, pytablewriter, peft, evaluate, ...) that we
    don't need just to score MATH-500. The utils.py module itself only
    requires: stdlib + datasets + sympy + math_verify (all in requirements.txt).
    """
    import importlib.util as _il
    from pathlib import Path as _Path
    import lm_eval as _le  # safe — its __init__ is light (no .tasks load)
    utils_path = _Path(_le.__file__).parent / "tasks" / "minerva_math" / "utils.py"
    spec = _il.spec_from_file_location("_minerva_math_utils", utils_path)
    m = _il.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_MU = None
def _mu():
    """Lazy singleton accessor for the directly-loaded minerva_math utils."""
    global _MU
    if _MU is None:
        _MU = _load_minerva_utils()
    return _MU


def _ensure_fewshot_prefix() -> str:
    global _FEWSHOT_PREFIX
    if _FEWSHOT_PREFIX is None:
        parts = []
        for fs in _mu().list_fewshot_samples():
            parts.append(_doc_to_text(fs["problem"]) + "\n" + fs["solution"] + "\n\n")
        _FEWSHOT_PREFIX = "".join(parts)
    return _FEWSHOT_PREFIX


def fmt_prompt(question: str, tokenizer=None) -> str:
    """4-shot Minerva-paper-canonical prompt (see module docstring for rationale)."""
    return _ensure_fewshot_prefix() + _doc_to_text(question)


# --- task-helper interface ---

def setup(tokenizer) -> list[str]:
    """Static stops; tokenizer ignored."""
    return STOPS


def truncate_at_stops(comp: str, stops=None) -> str:
    out = comp
    for s in (stops or STOPS):
        if s in out:
            out = out.split(s)[0]
    return out


def score(text: str, doc_or_gold) -> dict[str, Any]:
    """Wrap lm-eval `process_results` for one (pred, gold) pair.

    `doc_or_gold` must be a doc dict carrying `_solution` (full raw gold
    solution text with \\boxed{...}); we extract & normalize the gold answer
    inline. Returns {'strict_ok' = exact_match, 'flex_ok' = math_verify}.
    """
    mu = _mu()

    if isinstance(doc_or_gold, dict):
        solution = doc_or_gold.get("_solution") or doc_or_gold.get("gold")
    else:
        solution = doc_or_gold

    boxed = mu.last_boxed_only_string(solution)
    if boxed is None:
        answer = ""
    else:
        try:
            answer = mu.normalize_final_answer(mu.remove_boxed(boxed))
        except Exception:
            answer = ""
    doc = {"problem": "", "solution": solution, "answer": answer}
    res = mu.process_results(doc, [text])
    return {
        "strict_ok": bool(res.get("exact_match", 0)),
        "flex_ok":   bool(res.get("math_verify", 0)),
        "strict_pred": "",
        "flex_pred":   "",
        "gold_extracted": answer,
    }


def load_docs(limit: int | None = None) -> list[dict]:
    """Load MATH-500 test split as [{doc_id, question, gold, _solution}, ...]."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    n = len(ds)
    for i in range(n):
        if limit is not None and i >= limit:
            break
        sol = ds[i]["solution"]
        out.append({
            "doc_id": i,
            "question": ds[i]["problem"],
            "gold": sol,
            "_solution": sol,
        })
    return out
