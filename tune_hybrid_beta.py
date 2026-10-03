"""Tune hybrid_beta: the BM25/Dense mix weight of the hybrid relevance channel.

Selection is done on TRAIN queries only (gold-labeled, zero LLM cost), so the
test set stays clean for the judge run. Mirrors the main experiment exactly:
same pool construction (build_experiment_set, seed, 3000 tools) as
run_catalog_experiment.py. Also reports weighted-RRF fusion as an alternative.

Run:  python tune_hybrid_beta.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

from enga.config import Config
from enga.data import Query, build_experiment_set, load_toolret
from enga.evaluator import set_f1, set_ndcg, set_recall
from enga.features import ToolIndex
from enga.utils import seed_everything

SEED = 42
B = 5
RRF_K = 60


def norm(s: np.ndarray) -> np.ndarray:
    lo, hi = float(s.min()), float(s.max())
    return ((s - lo) / max(1e-6, hi - lo)).astype(np.float32)


def rrf_scores(ranks: np.ndarray) -> np.ndarray:
    """ranks: 0-based positions of each tool in one ranking."""
    return 1.0 / (RRF_K + ranks)


def topk(index: ToolIndex, scores: np.ndarray) -> list[str]:
    idx = np.argsort(-scores)[:B]
    return [index.ids[int(i)] for i in idx]


def eval_scores(index: ToolIndex, queries, score_fn) -> tuple[float, float, float]:
    f1s, recs, nds = [], [], []
    for q in queries:
        S = topk(index, score_fn(q))
        f1s.append(set_f1(S, q.gold_ids))
        recs.append(set_recall(S, q.gold_ids))
        nds.append(set_ndcg(S, q.gold_ids))
    return float(np.mean(f1s)), float(np.mean(recs)), float(np.mean(nds))


def main():
    seed_everything(SEED)
    root = Path(__file__).resolve().parent
    enga_root = root if (root / "enga").is_dir() else root.parent / "experiments"

    import json
    with open(root / "data" / "catalog_splits.json", encoding="utf-8") as f:
        catalogs = json.load(f)

    all_queries = []
    for cat in catalogs:
        for q_dict in cat["train_queries"][:30] + cat["test_queries"][:10]:
            all_queries.append(Query(qid=q_dict["id"], text=q_dict["query"],
                                     gold_ids=[t["id"] for t in q_dict.get("tools", [])]))
    train_qs = [q for cat in catalogs for q in
                [Query(qid=d["id"], text=d["query"], gold_ids=[t["id"] for t in d.get("tools", [])])
                 for d in cat["train_queries"][:30]]]
    test_qs = [q for cat in catalogs for q in
               [Query(qid=d["id"], text=d["query"], gold_ids=[t["id"] for t in d.get("tools", [])])
                for d in cat["test_queries"][:10]]]

    raw_tools, _ = load_toolret(str(enga_root / "data" / "toolret"), "toolbench", "web")
    sub_tools, _, ids = build_experiment_set(raw_tools, all_queries, len(all_queries),
                                             n_tools=3000, seed=SEED)
    cfg = Config()
    index = ToolIndex(sub_tools, ids, cfg, query_texts=[q.text for q in all_queries])
    print(f"Index built: {index.n} tools | train {len(train_qs)} / test {len(test_qs)} queries\n")

    # per-query channel scores are beta-independent: precompute min-max norms
    dense = {q.qid: norm(index.rel(q)) for q in all_queries}
    bm25 = {q.qid: norm(index.bm25_scores(q.text)) for q in all_queries}
    dense_rank = {q.qid: np.argsort(np.argsort(-index.rel(q))) for q in all_queries}
    bm25_rank = {q.qid: np.argsort(np.argsort(-index.bm25_scores(q.text))) for q in all_queries}

    print(f"{'beta':>5} | {'TRAIN F1@5':>10} {'Rec':>7} {'NDCG':>7} | {'TEST F1@5':>10} {'Rec':>7} {'NDCG':>7}")
    print("-" * 72)
    results = {}
    for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        lin = lambda q, b=beta: b * bm25[q.qid] + (1 - b) * dense[q.qid]
        results[beta] = (eval_scores(index, train_qs, lin), eval_scores(index, test_qs, lin))
        tr, te = results[beta]
        print(f"{beta:>5.1f} | {tr[0]:>10.4f} {tr[1]:>7.4f} {tr[2]:>7.4f} | {te[0]:>10.4f} {te[1]:>7.4f} {te[2]:>7.4f}")

    wrrf = lambda q: 0.5 * rrf_scores(bm25_rank[q.qid]) + 0.5 * rrf_scores(dense_rank[q.qid])
    tr, te = eval_scores(index, train_qs, wrrf), eval_scores(index, test_qs, wrrf)
    print(f"{'RRF':>5} | {tr[0]:>10.4f} {tr[1]:>7.4f} {tr[2]:>7.4f} | {te[0]:>10.4f} {te[1]:>7.4f} {te[2]:>7.4f}")

    best_beta = max(results, key=lambda b: results[b][0][0])
    tr, te = results[best_beta]
    print("-" * 72)
    print(f"SELECTED beta = {best_beta:.1f}  (train F1@5 = {tr[0]:.4f}; test F1@5 = {te[0]:.4f} — reference only)")


if __name__ == "__main__":
    main()
