"""Parameterized Nested Greedy decoder D(q; alpha) (Method 3.2, Eqs. 7-10)
and the oracle-marginal-gain NGA of Preliminary Alg. 1.
"""
from __future__ import annotations

import numpy as np

from .features import ToolIndex
from .data import Query


def decode(index: ToolIndex, q: Query, alpha, b: int,
           use_syn: bool = True, use_cost: bool = True,
           n_lightweight: list | None = None) -> list[str]:
    """Greedy composition with the parameterized proxy gain (Eqs. 7-10).

    alpha = [alpha1, alpha2, alpha3]: relevance / synergy / cost weights.
    Returns selected tool ids after b steps; O(N*b) lightweight ops, no LLM.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    rel = index.rel_norm(q)
    cost = index.cost if use_cost else 0.0
    sel: list[int] = []
    sel_ids: list[str] = []
    for t in range(b):
        if use_syn and alpha[1] > 0:
            syn = index.cfg.lambda_s * index.sem_syn(sel) + \
                (1.0 - index.cfg.lambda_s) * index.sch_syn(sel)
        else:
            syn = 0.0
        gain = alpha[0] * rel + alpha[1] * syn - alpha[2] * cost
        if sel:
            gain[sel] = -np.inf
        v = int(np.argmax(gain))
        sel.append(v)
        sel_ids.append(index.ids[v])
        if n_lightweight is not None:
            n_lightweight.append(int(np.isfinite(gain).sum()))  # candidates scored this step
    return sel_ids


def decode_oracle_nga(index: ToolIndex, q: Query, utility, b: int,
                      candidate_cap: int = 0, budget=None,
                      n_expensive: list | None = None) -> list[str]:
    """Preliminary Alg. 1: NGA with the TRUE marginal gain evaluated by an oracle.

    Delta(v | X) = U(q, X+{v}) - U(q, X). Every candidate evaluation is an
    expensive oracle/LLM call -> O(N*b) expensive evaluations, the exact cost
    ENGA avoids (Method 3.4, Eq. 17-18). Honors an evaluation budget by
    stopping early; candidate_cap caps candidates per step (top-M by relevance).
    """
    rel = index.rel(q)
    sel: list[int] = []
    sel_ids: list[str] = []
    u_prev = 0.0
    for t in range(b):
        remaining = np.argsort(-rel)
        remaining = [int(v) for v in remaining if int(v) not in sel]
        if candidate_cap and candidate_cap > 0:
            remaining = remaining[:candidate_cap]
        if not remaining:
            break
        base = set(sel_ids)
        best_v, best_delta = None, -np.inf
        for v in remaining:
            if budget is not None and budget.exhausted():
                break
            u = utility(q, sorted(base | {index.ids[v]}))
            if n_expensive is not None:
                n_expensive.append(1)
            d = u - u_prev
            if d > best_delta:
                best_delta, best_v = d, v
        if best_v is None or best_delta <= 0:
            break
        sel.append(best_v)
        sel_ids.append(index.ids[best_v])
        u_prev += best_delta
    return sel_ids
