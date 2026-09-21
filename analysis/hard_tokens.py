"""HARD / EASY token definition (paper Appendix C).

For a generated position t, let ``z_i(v)`` be the logit of token ``v`` at
recurrent iteration ``i`` (i = 1..r), and let

    y* = argmax_v z_r(v)

be the token the model actually emits (greedy decoding reads the final
iteration).  The *fixed-target logit margin* of ``y*`` at iteration ``i`` is

    m_i = z_i(y*) - max_{v != y*} z_i(v)

with peak / final values ``m_peak = max_i m_i`` and ``m_final = m_r``.  The
*confidence drop* of the final winner is

    d = m_peak - m_final

and a generated token is HARD iff ``d >= eps`` (eps = 1.5 in the paper), EASY
otherwise.  Intuitively, HARD tokens are the ones the model committed to early
and then walked back -- the late-flip positions LoopCD is designed to fix.

Note on probabilities vs logits: ``log softmax`` is shift invariant, so
``log p_i(y) - log p_i(z) == z_i(y) - z_i(z)``.  The margin can therefore be
computed from stored per-iteration *probabilities* without any loss.

On Huginn-0125 / GSM8K greedy (r = 32, eps = 1.5), EASY covers roughly 89% of
generated tokens and HARD the remaining 11%.
"""
from __future__ import annotations

import numpy as np

EPS_DEFAULT = 1.5
_TINY = 1e-12


def fixed_target_margin(logits_per_iter: np.ndarray, winner: int | None = None) -> np.ndarray:
    """``m_i`` for i = 1..r.

    Parameters
    ----------
    logits_per_iter : (r, V) array of logits (or log-probs -- both work).
    winner : token id of ``y*``; defaults to the argmax at the final iteration.
    """
    Z = np.asarray(logits_per_iter, dtype=np.float64)
    if winner is None:
        winner = int(Z[-1].argmax())
    z_y = Z[:, winner].copy()
    Z_other = Z.copy()
    Z_other[:, winner] = -np.inf
    return z_y - Z_other.max(axis=1)


def margin_from_probs(probs_per_iter: np.ndarray, winner_row: int) -> np.ndarray:
    """Same as :func:`fixed_target_margin` but for a (n_tracked, r) matrix of
    probabilities over a *tracked subset* of the vocabulary (the runner-up is
    virtually always inside the tracked top-k)."""
    P = np.asarray(probs_per_iter, dtype=np.float64)
    L = np.log(np.maximum(P, _TINY))
    L_other = L.copy()
    L_other[winner_row] = -np.inf
    return L[winner_row] - L_other.max(axis=0)


def confidence_drop(margin: np.ndarray) -> float:
    """``d = m_peak - m_final``."""
    m = np.asarray(margin, dtype=np.float64)
    return float(m.max() - m[-1])


def peak_iteration(margin: np.ndarray) -> int:
    """``argmax_i m_i`` (0-indexed); the iteration the model was most
    committed to ``y*`` before revising it."""
    return int(np.asarray(margin).argmax())


def is_hard(drop: float, eps: float = EPS_DEFAULT) -> bool:
    return float(drop) >= eps


def classify(drops, eps: float = EPS_DEFAULT) -> list[str]:
    return ["hard" if is_hard(d, eps) else "easy" for d in drops]
