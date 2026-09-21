"""Analyses behind the paper's motivation section and the HARD/EASY ablation.

:mod:`capture` records, once per benchmark, the per-iteration logit trajectory
of every generated token; the sub-packages read those captures.

* :mod:`hard_tokens` -- HARD / EASY definition
* :mod:`capture`     -- greedy decode + per-iteration LM-head readout
* :mod:`tasks`       -- task-helper loading and sharding

Sub-packages:

* ``1_figure3_hard_token_types``  -- Figure 3, which token types are HARD
* ``2_ablation_hard_easy``        -- applying LoopCD only at HARD or EASY tokens
"""
