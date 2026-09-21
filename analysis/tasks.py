"""Small helpers shared by the analysis scripts."""
from __future__ import annotations

import importlib.util
import sys
import zlib
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent.parent / "loopcd" / "eval"

_MODULES = {
    "gsm8k": "eval_gsm8k.py",
    "strategyqa": "eval_strategyqa.py",
}


def load_task_helper(task: str):
    """Load one of ``loopcd/eval/eval_*.py`` by task name."""
    if task not in _MODULES:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(_MODULES)}")
    name = f"_loopcd_task_{task}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(_EVAL_DIR / _MODULES[task]))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def shard_key(doc_id) -> int:
    """Stable integer key for sharding."""
    if isinstance(doc_id, int):
        return doc_id
    s = str(doc_id)
    if "/" in s:
        try:
            return int(s.rsplit("/", 1)[1])
        except ValueError:
            pass
    try:
        return int(s)
    except ValueError:
        return zlib.crc32(s.encode()) & 0x7FFFFFFF
