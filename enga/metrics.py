"""Portfolio / single-set metrics (Evaluation section: SR, Best@K, Coverage@K, Diversity)."""
from __future__ import annotations

import numpy as np

from .data import Query
from .evaluator import set_f1, set_recall, set_ndcg
from .features import ToolIndex
from .qd import gamma, comp_distance


def best_at_k(portfolio: list[list[str]], q: Query, metric: str = "f1") -> float:
    fn = {"f1": set_f1, "recall": set_recall, "ndcg": set_ndcg}[metric]
    return max((fn(S, q.gold_ids) for S in portfolio), default=0.0)


def coverage_at_k(portfolio: list[list[str]], q: Query) -> float:
    """Fraction of gold tools covered by the union of the portfolio."""
    if not q.gold_ids:
        return 0.0
    union = set()
    for S in portfolio:
        union |= set(S)
    return len(union & set(q.gold_ids)) / len(q.gold_ids)


def portfolio_diversity(portfolio: list[list[str]], index: ToolIndex) -> float:
    if len(portfolio) < 2:
        return 0.0
    gs = [gamma(index, S) for S in portfolio]
    ds = [comp_distance(gs[i], gs[j]) for i in range(len(gs)) for j in range(i + 1, len(gs))]
    return float(np.mean(ds))


def summarize_single(S: list[str], q: Query) -> dict:
    return {"f1": set_f1(S, q.gold_ids), "recall": set_recall(S, q.gold_ids),
            "ndcg": set_ndcg(S, q.gold_ids)}
