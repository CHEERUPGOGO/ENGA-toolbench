"""Experiment driver shared by all RQs: builds the index once, runs each method
query-by-query with a per-query fresh budget, and aggregates metrics."""
from __future__ import annotations

import numpy as np

from .config import Config
from .data import Query, ToolDoc, build_experiment_set
from .evaluator import (
    EvalBudget, OracleUtility, LLMUtility, LLMJudgeUtility,
    UnsupervisedUtility, set_f1, set_recall
)
from .features import ToolIndex
from .metrics import best_at_k, coverage_at_k, portfolio_diversity, summarize_single
from .nga import decode
from .utils import Logger, seed_everything, save_json
from .baselines import METHODS, PORTFOLIO_METHODS, DEFAULT_RUN_ORDER


def build_index(cfg: Config, synthetic: bool = False):
    """Load data, sample the experiment set, build the ToolIndex."""
    if synthetic:
        from .data import make_synthetic
        tools, queries = make_synthetic(cfg.n_tools, cfg.n_queries, cfg.seed)
        src = "synthetic"
    elif cfg.source == "raw":
        tools, queries = load_raw(cfg)
        src = "raw"
    else:
        from .data import load_toolret
        tools, queries = load_toolret(cfg.data_dir, cfg.source, cfg.library)
        src = f"toolret/{cfg.source}"
    sub, qs, ids = build_experiment_set(tools, queries, cfg.n_queries, cfg.n_tools, cfg.seed)
    index = ToolIndex(sub, ids, cfg, query_texts=[q.text for q in qs])
    return index, qs, src


def load_raw(cfg: Config):
    from .data import load_toolbench_raw
    return load_toolbench_raw(cfg.data_dir, split=cfg.raw_split)


def make_utility(cfg: Config, index: ToolIndex, count_calls: bool = False,
                 budget: EvalBudget | None = None):
    if cfg.eval_mode in ("judge", "llm_judge"):
        return LLMJudgeUtility(cfg, index.tools, budget or EvalBudget())
    elif cfg.eval_mode in ("unsupervised", "zero_shot"):
        return UnsupervisedUtility(cfg, index, count_calls=count_calls, budget=budget)
    elif cfg.eval_mode == "llm":
        return LLMUtility(cfg, index.tools, budget or EvalBudget())
    return OracleUtility(cfg, count_calls=count_calls, budget=budget)


def run_methods(cfg: Config, index: ToolIndex, queries: list[Query],
                method_names: list[str] | None = None) -> dict:
    """Run each method on every query; returns per-method aggregate metrics."""
    log = Logger(cfg.verbose)
    method_names = method_names or DEFAULT_RUN_ORDER
    results: dict = {}

    for name in method_names:
        fn = METHODS[name]
        per_q = []
        for q in queries:
            rng = seed_everything(cfg.seed + stable_int(q.qid))
            budget = EvalBudget(cfg.budget)
            # utility: expensive-counting only for methods that spend evals
            counting = name in ("nga_true",) or (name == "llm_greedy" and cfg.eval_mode == "llm") \
                or name in PORTFOLIO_METHODS
            utility = make_utility(cfg, index, count_calls=counting, budget=budget)
            out = fn(index, q, cfg, rng, utility, budget)
            if name in PORTFOLIO_METHODS:
                archive, rec = out
                portfolio = archive.portfolio()
                S_mu = decode(index, q, rec.get("final_mu") or cfg.init_alpha, cfg.b)
                rec_metrics = {
                    "f1_single": set_f1(S_mu, q.gold_ids),
                    "recall_single": set_recall(S_mu, q.gold_ids),
                    "f1": best_at_k(portfolio, q, "f1"),
                    "recall": best_at_k(portfolio, q, "recall"),
                    "ndcg": best_at_k(portfolio, q, "ndcg"),
                    "coverage": coverage_at_k(portfolio, q),
                    "diversity": portfolio_diversity(portfolio, index),
                    "final_mu": rec.get("final_mu"),
                    "n_expensive": rec.get("n_expensive", 0),
                    "n_lightweight": rec.get("n_lightweight", 0),
                }
            else:
                S, rec = out
                s_dict = summarize_single(S, q)
                rec_metrics = {"f1_single": s_dict["f1"],
                               "recall_single": s_dict["recall"],
                               **s_dict,
                               "coverage": coverage_at_k([S], q),
                               "diversity": 0.0,
                               "n_expensive": rec.get("n_expensive", 0)}
            rec_metrics["budget_used"] = budget.count
            per_q.append({"qid": q.qid, "gold_size": len(q.gold_ids), **rec_metrics})
        agg = aggregate(per_q)
        results[name] = {"per_query": per_q, "aggregate": agg}
        log(f"{name:14s} F1={agg['f1']:.4f} Recall={agg['recall']:.4f} "
            f"Cov@{cfg.K}={agg['coverage']:.4f} Div={agg['diversity']:.4f} "
            f"exp_evals/q={agg['n_expensive']:.1f}")
    return results


def stable_int(s: str) -> int:
    from .utils import stable_hash
    return stable_hash(s) % (2**31)


def aggregate(per_q: list[dict]) -> dict:
    keys = ["f1", "recall", "ndcg", "coverage", "diversity", "n_expensive", "n_lightweight", "budget_used", "f1_single", "recall_single"]
    out = {}
    for k in keys:
        vals = [pq[k] for pq in per_q if pq.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else 0.0
    # RQ5 alpha-pattern: mean learned alpha bucketed by gold size
    mus = [pq["final_mu"] for pq in per_q if pq.get("final_mu")]
    if mus:
        out["mean_alpha"] = [float(np.mean([m[i] for m in mus])) for i in range(3)]
    return out


def alpha_pattern(results: dict, method: str = "enega") -> dict:
    """Mean learned alpha grouped by gold-set size (simple/focused vs multi-step)."""
    if method not in results:
        return {}
    per_q = results[method]["per_query"]
    buckets: dict[str, list] = {}
    for pq in per_q:
        g = pq.get("gold_size", 0)
        key = "1" if g <= 1 else ("2" if g == 2 else "3+")
        buckets.setdefault(key, []).append(pq.get("final_mu") or [0, 0, 0])
    return {k: [float(np.mean([m[i] for m in v])) for i in range(3)]
            for k, v in sorted(buckets.items())}
