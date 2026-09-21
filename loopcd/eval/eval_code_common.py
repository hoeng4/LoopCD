"""Shared low-level helpers for HumanEval+ / MBPP+ task helpers.

Exposes: BASE_EOS, HE_EXTRA_DIRECT, HE_EXTRA_CHAT, _trim_eos, _build_prompt,
_run_check. Used internally by eval_humaneval.py and eval_mbpp.py.
"""
from __future__ import annotations
import sys



BASE_EOS = ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint("]
HE_EXTRA_DIRECT = ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
HE_EXTRA_CHAT = ["\n```\n"]

DEFAULT_INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
)
DEFAULT_RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
)


def _trim_eos(text, eos_list):
    min_idx = len(text)
    for eos in eos_list:
        idx = text.find(eos)
        if idx != -1 and idx < min_idx:
            min_idx = idx
    return text[:min_idx].replace("\t", "    ")


def _build_prompt(tokenizer, task_prompt):
    from evalplus.provider.utility import make_raw_chat_prompt
    if tokenizer.chat_template is None:
        eos = BASE_EOS + HE_EXTRA_DIRECT
        return task_prompt, eos
    wrapped = make_raw_chat_prompt(task_prompt, DEFAULT_INSTRUCTION_PREFIX,
                                   DEFAULT_RESPONSE_PREFIX, tokenizer)
    eos = BASE_EOS + HE_EXTRA_CHAT
    return wrapped, eos


def _run_check(suite, problem, code, gt, fast_check=False):
    from evalplus.eval import untrusted_check, PASS
    if suite == "base":
        inputs, expected, ref_time = problem["base_input"], gt["base"], gt["base_time"]
    else:
        inputs, expected, ref_time = problem["plus_input"], gt["plus"], gt["plus_time"]
    bench = "humaneval" if "HumanEval" in problem.get("task_id", "") else "mbpp"
    stat, _ = untrusted_check(bench, code, inputs, problem["entry_point"],
                              expected, problem["atol"], ref_time, fast_check=fast_check)
    return str(stat), stat == PASS

