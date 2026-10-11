"""Comprehensive Offline Ablations on 10-Catalog Test Set (3 Seeds).

Zero LLM calls required. Evaluates:
  1. Cost Channel Ablation: Grounded Cost vs Synthetic Cost vs No Cost (alpha_cost=0)
  2. Synergy Channel Ablation: Full (lambda_s=0.5) vs Semantic-only (1.0) vs Schema-only (0.0) vs No Synergy (alpha_syn=0)
  3. Hybrid Beta Sensitivity on Test Set: beta in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  4. Candidate Pool Scale Sensitivity: N=1000 vs N=3000 tools

Usage:
  python run_ablations_10cat.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
ENGA_ROOT = ROOT if (ROOT / "enga").is_dir() else ROOT.parent / "experiments"
sys.path.insert(0, str(ENGA_ROOT))

from enga.config import Config
from enga.data import Query, build_experiment_set, load_toolret
from enga.evaluator import set_f1, set_ndcg, set_recall
from enga.features import ToolIndex
from enga.nga import decode


def project_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w.astype(np.float32)


def main():
    splits_path = ROOT / "data" / "catalog_splits10.json"
    catalogs = json.load(open(splits_path, encoding="utf-8"))
    cat_ids = [str(c["catalog_id"]) for c in catalogs]

    # Load learned priors for seeds 42, 43, 44
    seeds = [42, 43, 44]
    priors = {}
    for s in seeds:
        tag = "" if s == 42 else f"_s{s}"
        p = ROOT / "results" / f"catalog_learned_alphas_10cat{tag}.json"
        if p.exists():
            priors[s] = json.load(open(p, encoding="utf-8"))
        else:
            print(f"[warn] missing {p}")

    all_raw_tools, _ = load_toolret(str(ENGA_ROOT / "data" / "toolret"), "toolbench", "web")
    all_queries = [Query(qid=q["id"], text=q["query"], gold_ids=[t["id"] for t in q.get("tools", [])])
                   for cat in catalogs for q in cat["train_queries"] + cat["test_queries"]]

    # Pre-encode centroids
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    centroids = []
    for cat in catalogs:
        train_texts = [q["query"] for q in cat["train_queries"]]
        embs = st.encode(train_texts, normalize_embeddings=True, show_progress_bar=False)
        c = np.mean(embs, axis=0)
        centroids.append(c / np.linalg.norm(c))
    centroids = np.vstack(centroids)

    test_queries = []
    for cat in catalogs:
        true_cid = str(cat["catalog_id"])
        for q_dict in cat["test_queries"]:
            q_emb = st.encode([q_dict["query"]], normalize_embeddings=True, show_progress_bar=False)[0]
            matched_cid = str(cat_ids[int(np.argmax(centroids @ q_emb))])
            q_obj = Query(qid=q_dict["id"], text=q_dict["query"],
                          gold_ids=[t["id"] for t in q_dict.get("tools", [])])
            test_queries.append((true_cid, matched_cid, q_obj))

    print(f"Loaded {len(test_queries)} test queries across {len(catalogs)} catalogs.")

    # =========================================================================
    # 1. Cost Channel Ablation (N=3000, pool_seed=42)
    # =========================================================================
    print("\n--- Running Cost Channel Ablation (Grounded vs Synthetic vs None) ---")
    cost_results = {"grounded": [], "synthetic": [], "no_cost": []}

    for mode in ("grounded", "synthetic", "no_cost"):
        cfg = Config()
        cfg.cost_mode = "synthetic" if mode == "synthetic" else "grounded"
        use_cost = (mode != "no_cost")

        sub_tools, qs, ids = build_experiment_set(all_raw_tools, all_queries, len(all_queries),
                                                  n_tools=3000, seed=42)
        index = ToolIndex(sub_tools, ids, cfg, query_texts=[q.text for q in qs])

        for s in seeds:
            mean_alphas = {k: np.array(v) for k, v in priors[s]["learned_alphas_polyak"].items()}
            f1s = []
            for true_cid, matched_cid, q_obj in test_queries:
                alpha = project_simplex(mean_alphas[matched_cid])
                if mode == "no_cost":
                    # renormalize [rel, syn] to sum=1, cost weight=0
                    a_2d = alpha[:2] / max(1e-6, alpha[:2].sum())
                    alpha_eff = np.array([a_2d[0], a_2d[1], 0.0], dtype=np.float32)
                else:
                    alpha_eff = alpha
                S = decode(index, q_obj, alpha_eff, cfg.b, use_syn=True, use_cost=use_cost)
                f1s.append(set_f1(S, q_obj.gold_ids))
            cost_results[mode].append(float(np.mean(f1s)))

    # =========================================================================
    # 2. Synergy Channel Ablation (N=3000, pool_seed=42)
    # =========================================================================
    print("\n--- Running Synergy Channel Ablation ---")
    syn_results = {"full_0.5": [], "semantic_only": [], "schema_only": [], "no_syn": []}

    cfg = Config()
    sub_tools, qs, ids = build_experiment_set(all_raw_tools, all_queries, len(all_queries),
                                              n_tools=3000, seed=42)

    for syn_mode in ("full_0.5", "semantic_only", "schema_only", "no_syn"):
        cfg_syn = Config()
        if syn_mode == "semantic_only":
            cfg_syn.lambda_s = 1.0
        elif syn_mode == "schema_only":
            cfg_syn.lambda_s = 0.0
        else:
            cfg_syn.lambda_s = 0.5
        use_syn = (syn_mode != "no_syn")

        index_syn = ToolIndex(sub_tools, ids, cfg_syn, query_texts=[q.text for q in qs])

        for s in seeds:
            mean_alphas = {k: np.array(v) for k, v in priors[s]["learned_alphas_polyak"].items()}
            f1s = []
            for true_cid, matched_cid, q_obj in test_queries:
                alpha = project_simplex(mean_alphas[matched_cid])
                if syn_mode == "no_syn":
                    a_nosyn = np.array([alpha[0], 0.0, alpha[2]])
                    alpha_eff = a_nosyn / max(1e-6, a_nosyn.sum())
                else:
                    alpha_eff = alpha
                S = decode(index_syn, q_obj, alpha_eff, cfg_syn.b, use_syn=use_syn, use_cost=True)
                f1s.append(set_f1(S, q_obj.gold_ids))
            syn_results[syn_mode].append(float(np.mean(f1s)))

    # =========================================================================
    # 3. Hybrid Beta Sensitivity on Test Set
    # =========================================================================
    print("\n--- Running Hybrid Beta Sensitivity on Test Set ---")
    beta_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    beta_results = {b: [] for b in beta_vals}

    index_base = ToolIndex(sub_tools, ids, Config(), query_texts=[q.text for q in qs])
    for b in beta_vals:
        f1s_hybrid = []
        for _, _, q_obj in test_queries:
            scores = index_base.hybrid_norm(q_obj, beta=b)
            idx = np.argsort(-scores)[:5]
            S = [index_base.ids[int(i)] for i in idx]
            f1s_hybrid.append(set_f1(S, q_obj.gold_ids))
        beta_results[b] = float(np.mean(f1s_hybrid))

    # =========================================================================
    # 4. Candidate Pool Scale Sensitivity: N=1000 vs N=3000
    # =========================================================================
    print("\n--- Running Candidate Pool Scale Sensitivity (N=1000 vs N=3000) ---")
    scale_results = {}
    for N in (1000, 3000):
        sub_tools_N, qs_N, ids_N = build_experiment_set(all_raw_tools, all_queries, len(all_queries),
                                                        n_tools=N, seed=42)
        index_N = ToolIndex(sub_tools_N, ids_N, Config(), query_texts=[q.text for q in qs_N])
        f1s_bm25, f1s_dense, f1s_hybrid, f1s_ours = [], [], [], []
        mean_alphas = {k: np.array(v) for k, v in priors[42]["learned_alphas_polyak"].items()}
        for _, matched_cid, q_obj in test_queries:
            # BM25
            s_bm25 = index_N.bm25_scores(q_obj.text)
            f1s_bm25.append(set_f1([ids_N[i] for i in np.argsort(-s_bm25)[:5]], q_obj.gold_ids))
            # Dense
            s_dense = index_N.rel(q_obj)
            f1s_dense.append(set_f1([ids_N[i] for i in np.argsort(-s_dense)[:5]], q_obj.gold_ids))
            # Hybrid
            s_hyb = index_N.hybrid_norm(q_obj, beta=0.6)
            f1s_hybrid.append(set_f1([ids_N[i] for i in np.argsort(-s_hyb)[:5]], q_obj.gold_ids))
            # Ours (Amortized Decode)
            S_ours = decode(index_N, q_obj, project_simplex(mean_alphas[matched_cid]), 5)
            f1s_ours.append(set_f1(S_ours, q_obj.gold_ids))
        scale_results[N] = {
            "bm25": float(np.mean(f1s_bm25)),
            "dense": float(np.mean(f1s_dense)),
            "hybrid": float(np.mean(f1s_hybrid)),
            "amortized": float(np.mean(f1s_ours)),
        }

    # =========================================================================
    # Output Report
    # =========================================================================
    report = [
        "# Offline Ablation Study Report (10-Catalog Test Set)",
        "",
        "## 1. 成本通道消融 (Cost Channel Ablation, Mean ± Std over 3 seeds)",
        "| Cost Mode | Description | F1@5 (Mean ± Std) | Per-Seed F1 (42, 43, 44) |",
        "| :--- | :--- | :---: | :---: |",
    ]
    for mode, desc in [("grounded", "Grounded Cost (Physical Token/Method/Risk metadata)"),
                       ("synthetic", "Synthetic Cost (Hash-based pseudo-random)"),
                       ("no_cost", "No Cost Channel (alpha_cost = 0)")]:
        vals = np.array(cost_results[mode])
        report.append(f"| {mode} | {desc} | {vals.mean():.4f} ± {vals.std(ddof=1):.4f} | "
                      f"({', '.join(f'{x:.4f}' for x in vals)}) |")

    report += [
        "",
        "## 2. 协同通道消融 (Synergy Disentanglement, Mean ± Std over 3 seeds)",
        "| Synergy Setting | Description | F1@5 (Mean ± Std) | Per-Seed F1 (42, 43, 44) |",
        "| :--- | :--- | :---: | :---: |",
    ]
    for syn_mode, desc in [("full_0.5", "Full Synergy (0.5 Semantic + 0.5 Schema)"),
                           ("semantic_only", "Semantic Synergy Only (lambda_s = 1.0)"),
                           ("schema_only", "Schema Compatibility Only (lambda_s = 0.0)"),
                           ("no_syn", "No Synergy Channel (alpha_syn = 0)")]:
        vals = np.array(syn_results[syn_mode])
        report.append(f"| {syn_mode} | {desc} | {vals.mean():.4f} ± {vals.std(ddof=1):.4f} | "
                      f"({', '.join(f'{x:.4f}' for x in vals)}) |")

    report += [
        "",
        "## 3. 混合通道 Beta 敏感性 (Hybrid Beta Sweep on 100 Test Queries)",
        "| Beta (BM25 weight) | 1 - Beta (Dense weight) | Test Retrieval F1@5 |",
        "| :---: | :---: | :---: |",
    ]
    for b in beta_vals:
        report.append(f"| {b:.1f} | {1.0 - b:.1f} | {beta_results[b]:.4f} |")

    report += [
        "",
        "## 4. 候选库规模扩展性 (Scale Sensitivity: N=1000 vs N=3000)",
        "| Tool Pool Scale N | BM25 | Dense | Hybrid (beta=0.6) | Amortized Decode (Ours) |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ]
    for N in (1000, 3000):
        r = scale_results[N]
        report.append(f"| N={N} | {r['bm25']:.4f} | {r['dense']:.4f} | {r['hybrid']:.4f} | {r['amortized']:.4f} |")

    out_text = "\n".join(report)
    print("\n" + "=" * 60)
    print(out_text)
    print("=" * 60)

    out_path = ROOT / "results" / "offline_ablations_10cat.md"
    out_path.write_text(out_text, encoding="utf-8")
    print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()
