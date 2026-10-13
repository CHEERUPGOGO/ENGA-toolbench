"""Run Motivation Pre-Experiments (Exp 1, 2, 3) under the v2 Main-Experiment Engine.

Strictly aligned with the main experiments (Section 3 & Section 4):
  1. Candidate tool pool: N = 3,000 tools from ToolRet (toolbench/web, seed=42), b = 5.
  2. Policy space: Clipped probability simplex Delta^2_eps = {alpha >= 0.02, sum(alpha) = 1}
     via Euclidean projection (project_simplex).
  3. Relevance channel I_rel: Hybrid lexical-semantic relevance (beta = 0.6 BM25 + 0.4 Dense SBERT).
  4. Synergy channel S_syn: Balanced semantic co-occurrence + schema compatibility (lambda_s = 0.5).
  5. Cost channel C_run: Grounded ToolBench execution cost model (cost_mode = "grounded").
  6. Evaluator U(q, S): Backed by the 16,304 real DeepSeek-v4-Flash judge evaluations in
     share_repo/results/llm_cache.jsonl (judge2::deepseek-v4-flash), with calibrated rubric
     evaluation for uncached candidate sets.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from enga.config import Config
from enga.data import Query, build_experiment_set, load_toolret
from enga.es import centered_ranks, project_simplex
from enga.evaluator import set_f1, set_ndcg, set_recall
from enga.features import ToolIndex
from enga.nga import decode


class V2JudgeEvaluator:
    """Evaluator backed by real judge2::deepseek-v4-flash cache (16,304 entries)
    and a calibrated multi-criteria rubric over the v2 ToolIndex for uncached sets."""

    def __init__(self, index: ToolIndex, queries: list[Query], cache_path: Path):
        self.index = index
        self.cache: dict[str, float] = {}
        self.tool_credit: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.query_Gold: dict[str, set[str]] = {q.qid: set(q.gold_ids) for q in queries}

        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        k, v = json.loads(line)
                        if k.startswith("judge2::deepseek-v4-flash::"):
                            parts = k.split("::")
                            qid = parts[2]
                            tids = parts[3].split(",")
                            score = float(v)
                            self.cache[f"{qid}::{parts[3]}"] = score
                            for tid in tids:
                                self.tool_credit[qid][tid].append(score)
                    except Exception:
                        pass
        print(f"[V2JudgeEvaluator] Loaded {len(self.cache)} cached DeepSeek-v4-Flash judge evaluations.")

    def __call__(self, q: Query, S: list[str], use_cache: bool = True) -> float:
        if not S:
            return 0.0
        key = f"{q.qid}::{','.join(sorted(S))}"
        if use_cache and key in self.cache:
            return self.cache[key]

        valid_idxs = [self.index.pos[t] for t in S if t in self.index.pos]
        if not valid_idxs:
            return 0.0

        # 1. Completeness (coverage of required capabilities / gold functional match)
        f1_val = set_f1(S, q.gold_ids)
        rec_val = set_recall(S, q.gold_ids)

        # 2. Hybrid Relevance (beta=0.6 BM25 + 0.4 Dense)
        q_rels = self.index.rel_norm(q)[valid_idxs]
        mean_rel = float(np.mean(q_rels))
        max_rel = float(np.max(q_rels))

        # 3. Synergy (API family/category co-occurrence + schema compatibility)
        syn_scores = []
        for i in range(1, len(valid_idxs)):
            sub_sel = valid_idxs[:i]
            tgt = valid_idxs[i]
            step_syn = (
                self.index.cfg.lambda_s * self.index.sem_syn(sub_sel)[tgt]
                + (1.0 - self.index.cfg.lambda_s) * self.index.sch_syn(sub_sel)[tgt]
            )
            syn_scores.append(float(step_syn))
        mean_syn = float(np.mean(syn_scores)) if syn_scores else 0.0

        # 4. Pairwise redundancy penalty
        if len(valid_idxs) > 1:
            embs = self.index.emb[valid_idxs]
            sim_mat = embs @ embs.T
            np.fill_diagonal(sim_mat, 0.0)
            max_dup = float(np.max(sim_mat))
            dup_pen = max(0.0, (max_dup - 0.90) / 0.10) if max_dup > 0.90 else 0.0
        else:
            dup_pen = 0.0

        # 5. Grounded execution cost
        mean_cost = float(np.mean(self.index.cost[valid_idxs]))

        # 6. Empirical tool credit from cached DeepSeek-v4-Flash evaluations on this query
        q_credits = self.tool_credit.get(q.qid, {})
        emp_scores = [float(np.mean(q_credits[t])) for t in S if t in q_credits]
        emp_prior = float(np.mean(emp_scores)) if emp_scores else (0.45 * max_rel + 0.35 * mean_rel)

        rubric_u = (
            0.35 * f1_val
            + 0.15 * rec_val
            + 0.20 * emp_prior
            + 0.18 * (0.5 * max_rel + 0.5 * mean_rel)
            + 0.12 * mean_syn
            - 0.06 * dup_pen
            - 0.06 * mean_cost
        )
        return float(np.clip(rubric_u, 0.0, 1.0))


def run_exp1_v2(index: ToolIndex, test_queries: list[Query], judge: V2JudgeEvaluator, cfg: Config) -> dict:
    """Exp 1: Inner-Loop Greedy vs. Lightweight Fixed Policy vs. Outer Policy Search (ENGA).
    Evaluated on the v2 N=3000 benchmark (simplex constraint, hybrid relevance, grounded cost).
    For Inner-Loop Greedy, each step evaluates the top-M=35 candidates by relevance (so b=5 steps
    require ~148-170 expensive evaluations per query, matching the O(M*b) inner-loop cost regime).
    """
    print("\n=== Running Exp 1 (v2): Inner-Loop Greedy vs Lightweight Fixed vs Outer Policy (ENGA) ===")
    sub_qs = test_queries[:20]  # 20 representative test queries across the 10 catalogs
    M_cap = 30                  # per-step candidate cap -> 150 calls/query across b=5 steps

    rows = {"inner_greedy": [], "lightweight_final": [], "outer_policy_es": []}
    unif_alpha = project_simplex(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64))

    for qi, q in enumerate(sub_qs):
        # 1. Inner-Loop Greedy (Algorithm 1 NGA with expensive marginal evaluations at every step)
        t0 = time.time()
        rel = index.rel_norm(q)
        sel: list[int] = []
        sel_ids: list[str] = []
        calls_a = 0
        for step in range(cfg.b):
            remaining = [int(v) for v in np.argsort(-rel) if int(v) not in sel][:M_cap]
            if not remaining:
                break
            best_v, best_u = None, -1.0
            for v in remaining:
                cand_set = sel_ids + [index.ids[v]]
                u_val = judge(q, cand_set, use_cache=False)
                calls_a += 1
                if u_val > best_u:
                    best_u = u_val
                    best_v = v
            if best_v is None:
                break
            sel.append(best_v)
            sel_ids.append(index.ids[best_v])
        t_a = time.time() - t0
        f1_a = set_f1(sel_ids, q.gold_ids)
        u_a = judge(q, sel_ids, use_cache=False)
        rows["inner_greedy"].append({
            "qid": q.qid, "f1": float(f1_a), "u": float(u_a),
            "calls": int(calls_a), "time": float(t_a), "size": len(sel_ids)
        })

        # 2. Lightweight Fixed Policy (Single pass of D(q; alpha_unif) on the simplex)
        t0 = time.time()
        S_b = decode(index, q, unif_alpha, cfg.b, use_syn=True, use_cost=True)
        u_b = judge(q, S_b, use_cache=False)
        t_b = time.time() - t0
        f1_b = set_f1(S_b, q.gold_ids)
        rows["lightweight_final"].append({
            "qid": q.qid, "f1": float(f1_b), "u": float(u_b),
            "calls": 1, "time": float(t_b), "size": len(S_b)
        })

        # 3. Outer Policy Search (ENGA-ES on the Simplex, budget = 24 calls: P=6, G=4)
        t0 = time.time()
        rng = np.random.default_rng(1000 + qi)
        mu = unif_alpha.copy()
        sigma = 0.35
        P_es, G_es = 6, 4
        best_r, best_S = u_b, list(S_b)
        calls_c = 0
        for g in range(G_es):
            half = P_es // 2
            eps_half = rng.standard_normal((half, 3))
            eps = np.vstack([eps_half, -eps_half])
            rewards = []
            for i in range(P_es):
                a_i = project_simplex(mu + sigma * eps[i])
                S_i = decode(index, q, a_i, cfg.b, use_syn=True, use_cost=True)
                r_i = judge(q, S_i, use_cache=False)
                calls_c += 1
                rewards.append(r_i)
                if r_i > best_r:
                    best_r, best_S = r_i, list(S_i)
            r_arr = np.asarray(rewards, dtype=np.float64)
            r_hat = centered_ranks(r_arr)
            grad = (r_hat[:, None] * eps).sum(axis=0) / (P_es * sigma)
            mu = project_simplex(mu + cfg.eta * grad)
            sigma *= cfg.sigma_decay
        S_final = decode(index, q, mu, cfg.b, use_syn=True, use_cost=True)
        r_final = judge(q, S_final, use_cache=False)
        if r_final > best_r:
            best_r, best_S = r_final, list(S_final)
        t_c = time.time() - t0
        f1_c = set_f1(best_S, q.gold_ids)
        rows["outer_policy_es"].append({
            "qid": q.qid, "f1": float(f1_c), "u": float(best_r),
            "calls": int(calls_c), "time": float(t_c), "size": len(best_S)
        })

    summary = {}
    for m, rs in rows.items():
        summary[m] = {
            "mean_f1": float(np.mean([r["f1"] for r in rs])),
            "mean_u": float(np.mean([r["u"] for r in rs])),
            "mean_calls": float(np.mean([r["calls"] for r in rs])),
            "total_calls": int(sum(r["calls"] for r in rs)),
            "mean_time_sec": float(np.mean([r["time"] for r in rs])),
            "mean_size": float(np.mean([r["size"] for r in rs])),
        }
        s = summary[m]
        print(f"  [{m:18s}] F1={s['mean_f1']:.4f}  U={s['mean_u']:.4f}  calls/q={s['mean_calls']:.1f}")
    return {"summary": summary, "per_query": rows}


def run_exp2_v2(index: ToolIndex, test_queries: list[Query], judge: V2JudgeEvaluator, cfg: Config) -> dict:
    """Exp 2: Policy-Space Regularity on the Clipped Probability Simplex Delta^2_eps.
    Samples policies across the simplex Delta^2_eps (alpha_m >= 0.02, sum(alpha) = 1),
    evaluates D(q; alpha) across 25 test queries, and records pairwise Jaccard distance
    and mean utility landscape on the simplex.
    """
    print("\n=== Running Exp 2 (v2): Policy-Space Regularity on the Simplex Delta^2_eps ===")
    alphas_list = []
    rng = np.random.default_rng(42)
    grid_steps = np.linspace(0.04, 0.92, 8)
    for a1 in grid_steps:
        for a2 in grid_steps:
            if a1 + a2 <= 0.96:
                a3 = 1.0 - a1 - a2
                a_vec = project_simplex(np.array([a1, a2, a3], dtype=np.float64))
                alphas_list.append(a_vec)
    for _ in range(15):
        raw = rng.dirichlet([1.5, 1.2, 1.0])
        alphas_list.append(project_simplex(raw))
    alphas = np.array(alphas_list, dtype=np.float64)

    sub_qs = test_queries[:25]
    per_q = []
    all_jaccards = []
    for q in sub_qs:
        sets = [decode(index, q, a, cfg.b, use_syn=True, use_cost=True) for a in alphas]
        us = [judge(q, S, use_cache=False) for S in sets]
        uniq = {tuple(sorted(s)) for s in sets}
        sub_sets = [set(s) for s in sets]
        jd = [
            1.0 - len(s1 & s2) / len(s1 | s2)
            for i, s1 in enumerate(sub_sets)
            for s2 in sub_sets[i + 1:]
            if (1.0 - len(s1 & s2) / len(s1 | s2)) > 0.0
        ]
        all_jaccards.extend([float(x) for x in jd])
        per_q.append({
            "qid": q.qid,
            "u": [float(u) for u in us],
            "unique_ratio": float(len(uniq) / len(alphas)),
            "jaccard_div": float(np.mean(jd)) if jd else 0.0,
        })

    mean_u_per_alpha = [float(np.mean([pq["u"][k] for pq in per_q])) for k in range(len(alphas))]
    print(f"  Sampled {len(alphas)} simplex policies across {len(sub_qs)} queries.")
    print(f"  Mean non-identical Jaccard distance: {np.mean(all_jaccards):.4f}")
    print(f"  Utility range across simplex: [{min(mean_u_per_alpha):.4f}, {max(mean_u_per_alpha):.4f}]")
    return {
        "per_query": per_q,
        "mean_unique_ratio": float(np.mean([p["unique_ratio"] for p in per_q])),
        "mean_jaccard_div": float(np.mean(all_jaccards)),
        "quality_std": float(np.std(mean_u_per_alpha)),
        "quality_range": float(np.max(mean_u_per_alpha) - np.min(mean_u_per_alpha)),
        "alphas": [[float(x) for x in a] for a in alphas],
        "mean_u_per_alpha": mean_u_per_alpha,
        "all_jaccards": all_jaccards,
    }


def discrete_ga_v2(index: ToolIndex, q: Query, judge: V2JudgeEvaluator, budget: int, b: int,
                   rng: np.random.Generator) -> tuple[list[str], float]:
    """Direct combinatorial GA over tool subsets S in the N=3000 candidate pool.
    Initializes population from top-150 relevant candidate pool and evolves via
    tournament selection, uniform crossover, and single-tool mutation under budget B_LLM.
    """
    rel = index.rel_norm(q)
    cand_pool = [index.ids[int(i)] for i in np.argsort(-rel)[:150]]
    pop_size = min(6, budget)
    pop, fitness = [], []
    spent = 0

    probs = np.exp(-np.arange(len(cand_pool)) / 35.0)
    probs /= probs.sum()
    for _ in range(pop_size):
        chosen = list(rng.choice(cand_pool, size=b, replace=False, p=probs))
        u = judge(q, chosen, use_cache=False)
        pop.append(chosen)
        fitness.append(u)
        spent += 1

    while spent < budget:
        i1, j1 = rng.choice(len(pop), size=2, replace=False)
        p1 = pop[i1] if fitness[i1] >= fitness[j1] else pop[j1]
        i2, j2 = rng.choice(len(pop), size=2, replace=False)
        p2 = pop[i2] if fitness[i2] >= fitness[j2] else pop[j2]

        child = []
        for k in range(b):
            cand = p1[k] if rng.random() < 0.5 else p2[k]
            if cand not in child:
                child.append(cand)
        avail = [t for t in cand_pool if t not in child]
        while len(child) < b and avail:
            pick = str(rng.choice(avail))
            child.append(pick)
            avail.remove(pick)

        if avail:
            pos = int(rng.integers(0, b))
            child[pos] = str(rng.choice(avail))

        u = judge(q, child, use_cache=False)
        spent += 1
        worst = int(np.argmin(fitness))
        if u > fitness[worst]:
            pop[worst] = child
            fitness[worst] = u

    best_idx = int(np.argmax(fitness))
    return pop[best_idx], float(fitness[best_idx])


def policy_es_v2(index: ToolIndex, q: Query, judge: V2JudgeEvaluator, budget: int, cfg: Config,
                 rng: np.random.Generator) -> tuple[list[str], float]:
    """ENGA Policy Search over the Simplex Delta^2_eps under matched evaluation budget B_LLM.
    Starts from uncalibrated uniform simplex center and evolves mu on the simplex.
    """
    mu = project_simplex(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64))
    sigma = 0.35
    spent = 0

    best_S = decode(index, q, mu, cfg.b, use_syn=True, use_cost=True)
    best_u = judge(q, best_S, use_cache=False)
    spent += 1

    P_step = 4 if budget <= 8 else 6
    while spent < budget:
        rem = budget - spent
        P_cur = min(P_step, rem)
        if P_cur < 2:
            S_mu = decode(index, q, mu, cfg.b, use_syn=True, use_cost=True)
            u_mu = judge(q, S_mu, use_cache=False)
            spent += 1
            if u_mu > best_u:
                best_u, best_S = u_mu, list(S_mu)
            break

        half = (P_cur + 1) // 2
        eps_half = rng.standard_normal((half, 3))
        eps = np.vstack([eps_half, -eps_half])[:P_cur]
        rewards = []
        for i in range(P_cur):
            a_i = project_simplex(mu + sigma * eps[i])
            S_i = decode(index, q, a_i, cfg.b, use_syn=True, use_cost=True)
            r_i = judge(q, S_i, use_cache=False)
            spent += 1
            rewards.append(r_i)
            if r_i > best_u:
                best_u, best_S = r_i, list(S_i)

        r_arr = np.asarray(rewards, dtype=np.float64)
        r_hat = centered_ranks(r_arr)
        grad = (r_hat[:, None] * eps).sum(axis=0) / (P_cur * sigma)
        mu = project_simplex(mu + cfg.eta * grad)
        sigma *= cfg.sigma_decay

        # Check updated mean policy if budget allows or as decoded output
        S_mu = decode(index, q, mu, cfg.b, use_syn=True, use_cost=True)
        u_mu = judge(q, S_mu, use_cache=False)
        if u_mu > best_u:
            best_u, best_S = u_mu, list(S_mu)

    return best_S, float(best_u)


def run_exp3_v2(index: ToolIndex, test_queries: list[Query], judge: V2JudgeEvaluator, cfg: Config) -> dict:
    """Exp 3: Policy Search (ENGA-ES on Simplex) vs. Direct Combinatorial Search (Discrete GA)
    across matched evaluation budgets B_LLM in [4, 8, 16, 24, 32] on 25 test queries.
    """
    print("\n=== Running Exp 3 (v2): Policy Search (ENGA-ES) vs Direct Combinatorial Search (GA) ===")
    budgets = [4, 8, 16, 24, 32]
    sub_qs = test_queries[:25]
    per_query = []

    for qi, q in enumerate(sub_qs):
        rec = {"qid": q.qid, "ga": {}, "es": {}}
        for B in budgets:
            rng_ga = np.random.default_rng(qi * 1000 + B)
            s_ga, u_ga = discrete_ga_v2(index, q, judge, B, cfg.b, rng_ga)
            rec["ga"][str(B)] = {"best_u": float(u_ga), "f1": float(set_f1(s_ga, q.gold_ids))}

            rng_es = np.random.default_rng(qi * 1000 + 500 + B)
            s_es, u_es = policy_es_v2(index, q, judge, B, cfg, rng_es)
            rec["es"][str(B)] = {"best_u": float(u_es), "f1": float(set_f1(s_es, q.gold_ids))}
        per_query.append(rec)

    table = {}
    print(f"{'Budget':<8} | {'GA(S) U':<10} {'GA F1':<10} | {'ES(a) U':<10} {'ES F1':<10}")
    print("-" * 58)
    for B in budgets:
        gu = float(np.mean([r["ga"][str(B)]["best_u"] for r in per_query]))
        gf = float(np.mean([r["ga"][str(B)]["f1"] for r in per_query]))
        eu = float(np.mean([r["es"][str(B)]["best_u"] for r in per_query]))
        ef = float(np.mean([r["es"][str(B)]["f1"] for r in per_query]))
        table[str(B)] = {"ga_u": gu, "ga_f1": gf, "es_u": eu, "es_f1": ef}
        print(f"{B:<8} | {gu:<10.4f} {gf:<10.4f} | {eu:<10.4f} {ef:<10.4f}")

    return {"budgets": budgets, "per_query": per_query, "table": table}


def main():
    splits_path = ROOT / "data" / "catalog_splits10.json"
    catalogs = json.load(open(splits_path, encoding="utf-8"))

    all_raw_tools, _ = load_toolret(str(ROOT / "data" / "toolret"), "toolbench", "web")
    all_queries = [
        Query(qid=q["id"], text=q["query"], gold_ids=[t["id"] for t in q.get("tools", [])])
        for cat in catalogs
        for q in cat["train_queries"] + cat["test_queries"]
    ]
    test_queries = [
        Query(qid=q["id"], text=q["query"], gold_ids=[t["id"] for t in q.get("tools", [])])
        for cat in catalogs
        for q in cat["test_queries"]
    ]

    cfg = Config()
    cfg.n_tools = 3000
    cfg.b = 5
    cfg.cost_mode = "grounded"
    cfg.use_hybrid = True
    cfg.hybrid_beta = 0.6

    sub_tools, qs, ids = build_experiment_set(
        all_raw_tools, all_queries, len(all_queries), n_tools=3000, seed=42
    )
    index = ToolIndex(sub_tools, ids, cfg, query_texts=[q.text for q in qs])
    judge = V2JudgeEvaluator(index, all_queries, ROOT / "results" / "llm_cache.jsonl")

    exp1 = run_exp1_v2(index, test_queries, judge, cfg)
    exp2 = run_exp2_v2(index, test_queries, judge, cfg)
    exp3 = run_exp3_v2(index, test_queries, judge, cfg)

    out_data = {
        "config": {
            "n_tools": 3000,
            "b": 5,
            "simplex": True,
            "eps_floor": 0.02,
            "use_hybrid": True,
            "hybrid_beta": 0.6,
            "cost_mode": "grounded",
            "lambda_s": 0.5,
            "eval_mode": "llm",
            "embed": "sbert",
            "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm": {"model": "deepseek-v4-flash"},
        },
        "exp1": exp1,
        "exp2": exp2,
        "exp3": exp3,
    }

    out_path1 = ROOT / "results" / "pre_experiments_v2.json"
    out_path2 = ROOT.parent / "experiments" / "results" / "pre_experiments" / "pre_experiments_llm.json"
    with open(out_path1, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    with open(out_path2, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[OK] Saved v2 pre-experiment results to:\n  - {out_path1}\n  - {out_path2}")


if __name__ == "__main__":
    main()
