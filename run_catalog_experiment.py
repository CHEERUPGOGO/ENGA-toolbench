"""Continuous Domain-Amortized Evolution with Catalog Routing Experiment (try_ex, v2).

v2 fixes over v1:
1. Simplex constraint: alpha lives on {alpha >= 0, sum(alpha) = 1} throughout -- initial
   sampling, ES perturbations, updates and amortized priors are all Euclidean-projected
   onto the simplex. This kills the norm drift of v1 (e.g. alpha -> [8.3, 2.3, 0.0]) and
   makes warm/cold naturally step-size comparable.
2. Amortized prior = Polyak mean over the training trajectory (the tail a_30 was an
   arbitrary, order-dependent quantity). Tail is kept only as an ablation arm.
3. Test phase contrasts DIRECT DECODE (0 LLM cost) vs RE-EVOLUTION (P*G+G judge calls):
     decode: global default | cold a0 | tail a30 | amortized a_mean (ours)
     evolve: global default | cold a0 | amortized a_mean (ours)
4. Reproducibility: per-(query, arm) deterministic RNG spawn keys -- results are
   identical no matter how often the run is interrupted/resumed (no RNG state saved).
   The checkpoint stores a run signature and refuses to mix incompatible configs.
5. Parallelism: the only sequential dependency is query t starting from alpha_{t-1}
   WITHIN a catalog. Catalogs are trained concurrently and test queries are evaluated
   concurrently (--catalog-parallel / --query-parallel); the checkpoint is guarded by a
   lock and training saves partial per-catalog state after every query.

Pipeline:
1. Top 5 most cohesive catalogs (30 train + 10 test queries each, data/catalog_splits.json).
2. Training: per catalog, sequentially evolve alpha across train queries under the
   LLM Judge reward (zero gold access); keep the full trajectory, alpha* = Polyak mean.
3. Testing: route each test query to the nearest catalog centroid (MiniLM), then
   compare all decode/evolve arms against gold (F1 / Recall / NDCG@5) and judge reward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

# Dual layout: self-contained repo (enga/ sits next to this file) or the original
# workspace layout (../experiments/enga). Both must keep working.
ROOT = Path(__file__).resolve().parent
ENGA_ROOT = ROOT if (ROOT / "enga").is_dir() else ROOT.parent / "experiments"
sys.path.insert(0, str(ENGA_ROOT))

from enga.config import Config
from enga.data import Query, build_experiment_set, load_toolret
from enga.baselines import method_bm25, method_dense, method_hybrid
from enga.evaluator import EvalBudget, LLMJudgeUtility, UnsupervisedUtility, set_f1, set_recall, set_ndcg
from enga.features import ToolIndex
from enga.nga import decode
from enga.es import centered_ranks
from enga.utils import seed_everything, stable_hash

ALL_DECODE_ARMS = ["default", "cold", "tail", "warm"]
DECODE_ARMS = ["cold", "tail", "warm"]             # warm = amortized Polyak mean (ours)
ALL_EVO_ARMS = ["default", "cold", "warm"]
EVO_ARMS = ["cold", "warm"]                        # warm = amortized Polyak mean (ours)
RETRIEVAL_ARMS = ["bm25", "dense", "hybrid"]        # classic & hybrid retrieval baselines, no alpha


def project_simplex(v: np.ndarray, eps_floor: float = 0.02) -> np.ndarray:
    """Euclidean projection onto the probability simplex {x >= 0, sum(x) = 1} (Duchi et al.).
    
    When eps_floor > 0, enforces a strictly positive lower bound (x_i >= eps_floor)
    to prevent dimension collapse and artificial truncation of small weights (e.g. cost).
    """
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    res = np.maximum(v - theta, 0.0)
    if eps_floor > 0 and len(v) * eps_floor < 1.0:
        res = np.maximum(res, eps_floor)
        res = res / res.sum()
    return res


def spawn_rng(seed: int, key: str) -> np.random.Generator:
    """Deterministic per-key RNG: identical draws regardless of execution order/interruption."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stable_hash(key) % (2**32),)))


def parallel_es_step(index: ToolIndex, q: Query, utility, mu: np.ndarray, sigma: float,
                     P: int, dim: int, rng: np.random.Generator, cfg: Config,
                     executor: ThreadPoolExecutor) -> tuple[np.ndarray, float, list[str]]:
    """Runs one ES generation with parallel evaluation across P candidate policies."""
    half = P // 2
    eps_half = rng.standard_normal((half, dim))
    eps = np.vstack([eps_half, -eps_half])

    # Perturbed candidates live on the simplex (projection subsumes the >=0 clip).
    alphas = [project_simplex(mu + sigma * eps[i]) for i in range(P)]
    comps = [decode(index, q, a, cfg.b, use_syn=True, use_cost=True) for a in alphas]

    rewards = np.asarray(list(executor.map(lambda S: utility(q, S), comps)), dtype=np.float64)

    # Elitist evaluation of mu
    S_mu = decode(index, q, project_simplex(mu), cfg.b, use_syn=True, use_cost=True)
    r_mu = utility(q, S_mu)

    # Rank-based natural gradient update, projected back onto the simplex
    ranks = centered_ranks(rewards)
    grad = (eps * ranks[:, None]).sum(axis=0) / (P * sigma)
    new_mu = project_simplex(mu + cfg.eta * grad)

    best_idx = int(np.argmax(rewards))
    if r_mu >= rewards[best_idx]:
        best_comp, best_r = S_mu, r_mu
    else:
        best_comp, best_r = comps[best_idx], float(rewards[best_idx])

    return new_mu, best_r, best_comp


def run_query_evolution(index: ToolIndex, q: Query, utility, init_mu: np.ndarray,
                        cfg: Config, rng: np.random.Generator, G: int, P: int,
                        executor: ThreadPoolExecutor) -> tuple[np.ndarray, float, list[str]]:
    """Evolve alpha for a single query starting from init_mu over G generations.

    mu stays on the probability simplex; sigma scales with ||mu|| (norm bounded in
    [1/sqrt(3), 1] on the simplex, so all arms explore with comparable steps)."""
    mu = np.array(init_mu, dtype=np.float64)
    sigma = float(cfg.sigma0) * float(np.linalg.norm(mu))
    dim = len(mu)
    best_r_overall, best_comp_overall = -1.0, []

    for _ in range(G):
        mu, best_r, best_comp = parallel_es_step(index, q, utility, mu, sigma, P, dim, rng, cfg, executor)
        sigma *= cfg.sigma_decay
        if best_r > best_r_overall:
            best_r_overall, best_comp_overall = best_r, best_comp

    return mu, best_r_overall, best_comp_overall


def polyak_mean(traj: list[list[float]]) -> np.ndarray:
    """Amortized prior: average of post-update alphas along the training trajectory."""
    return np.mean(np.asarray(traj[1:], dtype=np.float64), axis=0)


def arm_stats(test_results: list[dict], prefix: str, arm: str) -> dict:
    rows = [r[f"{prefix}_{arm}"] for r in test_results]
    return {m: float(np.mean([x[m] for x in rows])) for m in ("f1", "recall", "ndcg", "reward")}


def paired(test_results: list[dict], a: str, b: str, metric: str = "f1") -> dict:
    """Paired stats a minus b over test queries (win/tie/lose + t statistic)."""
    d = np.array([r[a][metric] - r[b][metric] for r in test_results], dtype=np.float64)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return {"mean_diff": float(d.mean()), "win": int((d > 0).sum()), "tie": int((d == 0).sum()),
            "lose": int((d < 0).sum()), "t": float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-mode", choices=["judge", "unsupervised"], default="judge",
                        help="judge: LLM-as-a-Judge (DeepSeek) | unsupervised: local zero-shot proxy")
    parser.add_argument("--P", type=int, default=8, help="Population size (default 8)")
    parser.add_argument("--G", type=int, default=5, help="Number of generations (default 5)")
    parser.add_argument("--train-limit", type=int, default=30, help="Train queries per catalog (up to 30)")
    parser.add_argument("--test-limit", type=int, default=10, help="Test queries per catalog (up to 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for LLM Judge calls")
    parser.add_argument("--catalog-parallel", type=int, default=5, help="Catalogs trained concurrently")
    parser.add_argument("--query-parallel", type=int, default=5, help="Test queries evaluated concurrently")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix for checkpoint/result files (e.g. 'smoke'); keeps runs isolated")
    parser.add_argument("--hybrid-beta", type=float, default=0.6,
                        help="Weight for BM25 in hybrid relevance (default 0.6)")
    parser.add_argument("--cost-mode", choices=["grounded", "synthetic", "schema"], default="grounded",
                        help="Cost calculation mode (grounded: ToolBench prompt/payload/risk; synthetic: hash)")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="Disable hybrid retrieval (fallback to pure Dense embedding)")
    parser.add_argument("--splits", type=str, default="data/catalog_splits.json",
                        help="Catalog splits file, relative to repo root (default data/catalog_splits.json)")
    parser.add_argument("--llm-base-url", type=str, default=None,
                        help="Override LLM base_url (e.g. http://127.0.0.1:8046/v1)")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Override LLM model name (e.g. gemini-3.8-flash)")
    parser.add_argument("--llm-api-key", type=str, default=None,
                        help="Override LLM api_key")
    parser.add_argument("--pool-seed", type=int, default=None,
                        help="Seed for sub-library sampling (defaults to --seed). Fix it across seeds "
                             "to hold the pool fixed while varying only training randomness.")
    parser.add_argument("--decode-arms", type=str, default=",".join(DECODE_ARMS),
                        help=f"Comma list of decode arms to evaluate (default all: {','.join(DECODE_ARMS)})")
    parser.add_argument("--evo-arms", type=str, default=",".join(EVO_ARMS),
                        help=f"Comma list of re-evolution arms to evaluate (default all: {','.join(EVO_ARMS)})")
    parser.add_argument("--retrieval-arms", type=str, default=",".join(RETRIEVAL_ARMS),
                        help=f"Comma list of retrieval baselines (default all: {','.join(RETRIEVAL_ARMS)})")
    args = parser.parse_args()

    def _parse_arms(flag: str, allowed: list[str]) -> list[str]:
        arms = [a.strip() for a in flag.split(",") if a.strip()]
        bad = [a for a in arms if a not in allowed]
        if bad or not arms:
            parser.error(f"invalid arm list '{flag}' (unknown: {bad}); allowed: {allowed}")
        return arms

    decode_arms = _parse_arms(args.decode_arms, ALL_DECODE_ARMS)
    evo_arms = _parse_arms(args.evo_arms, ALL_EVO_ARMS)
    retrieval_arms = _parse_arms(args.retrieval_arms, RETRIEVAL_ARMS)
    pool_seed = args.pool_seed if args.pool_seed is not None else args.seed

    root = Path(__file__).resolve().parent
    splits_path = root / args.splits
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    ckpt_path = results_dir / f"checkpoint_{args.eval_mode}{suffix}.json"

    if not splits_path.exists():
        print(f"Splits not found at {splits_path}. Run split_catalogs.py first.")
        return

    with open(splits_path, "r", encoding="utf-8") as f:
        catalogs = json.load(f)

    cfg = Config()
    cfg.eval_mode = args.eval_mode
    cfg.cost_mode = args.cost_mode
    cfg.hybrid_beta = args.hybrid_beta
    cfg.use_hybrid = not args.no_hybrid
    cfg.P, cfg.G, cfg.seed = args.P, args.G, args.seed
    if args.llm_base_url:
        cfg.llm.base_url = args.llm_base_url
    if args.llm_model:
        cfg.llm.model = args.llm_model
    if args.llm_api_key:
        cfg.llm.api_key = args.llm_api_key
    seed_everything(args.seed)

    # Run signature: checkpoint refuses to mix incompatible configs (v1 flaw).
    # splits_hash pins the exact catalog splits (5-cat vs 10-cat differ here even
    # when all numeric hyperparameters match); arm lists pin the evaluated subset.
    splits_hash = hashlib.sha1(splits_path.read_bytes()).hexdigest()[:12]
    signature = {
        "eval_mode": args.eval_mode, "model": cfg.llm.model, "P": args.P, "G": args.G,
        "train_limit": args.train_limit, "test_limit": args.test_limit, "seed": args.seed,
        "pool_seed": pool_seed, "n_catalogs": len(catalogs), "splits_hash": splits_hash,
        "decode_arms": decode_arms, "evo_arms": evo_arms, "retrieval_arms": retrieval_arms,
        "simplex": True, "relative_sigma": True, "b": cfg.b, "eta": cfg.eta,
        "sigma0": cfg.sigma0, "sigma_decay": cfg.sigma_decay, "judge_prompt": "judge2",
        "cost_mode": args.cost_mode, "hybrid_beta": args.hybrid_beta, "use_hybrid": not args.no_hybrid,
    }

    print("=" * 100)
    print("CONTINUOUS DOMAIN-AMORTIZED EVOLUTION WITH CATALOG ROUTING (v2, simplex + parallel)")
    print(f"Evaluator: {args.eval_mode} ({cfg.llm.model if args.eval_mode == 'judge' else 'local proxy'}) | "
          f"P={args.P}, G={args.G}, alpha on simplex (sum=1) | train/test per catalog: {args.train_limit}/{args.test_limit}")
    print(f"Parallel: {args.catalog_parallel} catalogs x {args.query_parallel} test queries, {args.workers} LLM workers")
    print(f"Checkpoint: {ckpt_path.name}")
    print("=" * 100)

    # -------------------------------------------------------------
    # Shared index / evaluator
    # -------------------------------------------------------------
    print("\n[1/4] Loading Tool Library and building ToolIndex...")
    all_raw_tools, _ = load_toolret(str(ENGA_ROOT / "data" / "toolret"), "toolbench", "web")

    all_queries = []
    for cat in catalogs:
        for q_dict in cat["train_queries"][:args.train_limit] + cat["test_queries"][:args.test_limit]:
            all_queries.append(Query(qid=q_dict["id"], text=q_dict["query"],
                                     gold_ids=[t["id"] for t in q_dict.get("tools", [])]))

    sub_tools, qs, ids = build_experiment_set(all_raw_tools, all_queries, len(all_queries),
                                              n_tools=3000, seed=pool_seed)
    index = ToolIndex(sub_tools, ids, cfg, query_texts=[q.text for q in qs])
    print(f"Index built: {index.n} tools available for selection.")

    budget = EvalBudget()
    if args.eval_mode == "judge":
        utility = LLMJudgeUtility(cfg, index.tools, budget)
    else:
        utility = UnsupervisedUtility(cfg, index, count_calls=True, budget=budget)

    executor = ThreadPoolExecutor(max_workers=args.workers)
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Thread-safe checkpoint: one lock guards both mutation and file writes.
    ckpt_lock = threading.Lock()
    print_lock = threading.Lock()

    def plog(msg: str):
        with print_lock:
            print(msg, flush=True)

    def ckpt_mutate(mutator):
        with ckpt_lock:
            mutator()
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------
    # Phase 2: Training per Catalog -- catalogs run concurrently (independent chains)
    # -------------------------------------------------------------
    plog("\n[2/4] Phase 1: Evolution Training per Catalog (parallel)...")
    checkpoint: dict = {"signature": signature, "trained_catalogs": {}, "initial_alphas": {},
                        "trajectories": {}, "partial_catalogs": {}, "test_results": []}
    if ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if loaded.get("signature") != signature:
            plog(f"Checkpoint {ckpt_path.name} has an incompatible signature "
                 f"(existing: {loaded.get('signature')}).\nUse a different --tag or move the file.")
            return
        checkpoint = loaded
        checkpoint.setdefault("partial_catalogs", {})
        plog(f"Resumed checkpoint: {len(checkpoint['trained_catalogs'])}/{len(catalogs)} catalogs trained, "
             f"{len(checkpoint['test_results'])} test queries finished.")

    catalog_alphas = checkpoint["trained_catalogs"]
    catalog_trajectories = checkpoint["trajectories"]
    initial_alphas = checkpoint["initial_alphas"]

    def train_catalog(cat: dict):
        """Sequential evolution chain across one catalog's train queries."""
        cid = str(cat["catalog_id"])
        with ckpt_lock:
            if cid in catalog_alphas:
                return
            partial = checkpoint["partial_catalogs"].get(cid)

        if partial:
            cur_alpha = np.array(partial["alpha"], dtype=np.float64)
            trajectory = partial["traj"]
            start_step = partial["next_step"]
            plog(f"[cat {cid}] resuming at query {start_step + 1}/{args.train_limit}")
        else:
            raw = spawn_rng(args.seed, f"a0:{cid}").uniform(0.3, 1.2, size=3)
            cur_alpha = raw / raw.sum()          # initial alpha_0 on the simplex
            trajectory = [cur_alpha.tolist()]
            start_step = 0
            ckpt_mutate(lambda: initial_alphas.__setitem__(cid, cur_alpha.tolist()))
            plog(f"[cat {cid}] {cat['title_zh']} start alpha_0 = {cur_alpha.round(4).tolist()}")

        train_qs = cat["train_queries"][:args.train_limit]
        t0 = time.time()
        for step in range(start_step + 1, len(train_qs) + 1):
            q_dict = train_qs[step - 1]
            q_obj = Query(qid=q_dict["id"], text=q_dict["query"],
                          gold_ids=[t["id"] for t in q_dict.get("tools", [])])
            rng_q = spawn_rng(args.seed, f"train:{cid}:{step}")
            cur_alpha, best_r, _ = run_query_evolution(
                index, q_obj, utility, cur_alpha, cfg, rng_q, args.G, args.P, executor)
            trajectory.append(cur_alpha.tolist())
            # partial state persists after EVERY query -> crash-safe resume mid-chain
            _a, _t, _s = cur_alpha.tolist(), trajectory, step
            ckpt_mutate(lambda: checkpoint["partial_catalogs"].__setitem__(
                cid, {"alpha": _a, "traj": _t, "next_step": _s}))
            if step % 5 == 0 or step == len(train_qs):
                plog(f"[cat {cid}] [Query {step:2d}/{len(train_qs)}] best_reward={best_r:.3f} | "
                     f"alpha={cur_alpha.round(4).tolist()}")

        _a, _t = cur_alpha.tolist(), trajectory
        ckpt_mutate(lambda: (catalog_alphas.__setitem__(cid, _a),
                             catalog_trajectories.__setitem__(cid, _t),
                             checkpoint["partial_catalogs"].pop(cid, None)))
        plog(f"[cat {cid}] DONE in {time.time() - t0:.1f}s -> tail={cur_alpha.round(3).tolist()} | "
             f"polyak_mean={polyak_mean(trajectory).round(3).tolist()}")

    with ThreadPoolExecutor(max_workers=args.catalog_parallel) as cat_pool:
        list(cat_pool.map(train_catalog, catalogs))

    # Derived amortized priors
    mean_alphas = {cid: polyak_mean(traj) for cid, traj in catalog_trajectories.items()}
    with open(results_dir / f"catalog_learned_alphas{suffix}.json", "w", encoding="utf-8") as f:
        json.dump({"initial_alphas": initial_alphas, "learned_alphas_tail": catalog_alphas,
                   "learned_alphas_polyak": {k: v.tolist() for k, v in mean_alphas.items()},
                   "trajectories": catalog_trajectories}, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------
    # Phase 3: Testing -- routing, then decode vs re-evolution (queries run concurrently)
    # -------------------------------------------------------------
    plog("\n[3/4] Phase 2: Test Routing + Decode/Evolve Comparison (parallel)...")
    centroids = np.array([cat["centroid_vector"] for cat in catalogs])
    cat_ids = [str(cat["catalog_id"]) for cat in catalogs]

    test_items = [(str(cat["catalog_id"]), q_dict)
                  for cat in catalogs for q_dict in cat["test_queries"][:args.test_limit]]
    texts = [q["query"] for _, q in test_items]
    plog(f"Encoding {len(texts)} test queries for routing...")
    q_embs = st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    test_results = checkpoint["test_results"]
    tested_qids = {r["qid"] for r in test_results}

    def eval_test_item(item) -> None:
        (true_cid, q_dict), q_emb = item
        qid, q_text = q_dict["id"], q_dict["query"]
        gold_ids = [t["id"] for t in q_dict.get("tools", [])]
        if qid in tested_qids:      # already done in a previous (interrupted) run
            return
        q_obj = Query(qid=qid, text=q_text, gold_ids=gold_ids)

        # Routing: nearest catalog centroid
        matched_cid = str(cat_ids[int(np.argmax(centroids @ q_emb))])
        is_match = (matched_cid == true_cid)

        # All priors live on the simplex. The default keeps its RATIOS (uniform
        # normalization, decode-equivalent to [1.0, 0.2, 0.05]); Euclidean projection
        # would distort it to [0.9, 0.1, 0.0] and kill the cost weight.
        arm_alphas = {
            "default": np.array([1.0, 0.2, 0.05]) / 1.25,               # hand-tuned global default
            "cold": project_simplex(initial_alphas[matched_cid]),        # random a0, no amortization
            "tail": project_simplex(catalog_alphas[matched_cid]),        # v1-style tail prior (ablation)
            "warm": project_simplex(mean_alphas[matched_cid]),           # amortized Polyak mean (ours)
        }

        rec: dict = {"qid": qid, "query": q_text, "gold_size": len(gold_ids),
                     "true_catalog": true_cid, "matched_catalog": matched_cid,
                     "match_success": is_match}

        # --- Category A: direct decode (no test-time search; 1 judge call only for reward) ---
        for arm in decode_arms:
            S = decode(index, q_obj, arm_alphas[arm], cfg.b, use_syn=True, use_cost=True)
            rec[f"decode_{arm}"] = {"f1": set_f1(S, gold_ids), "recall": set_recall(S, gold_ids),
                                    "ndcg": set_ndcg(S, gold_ids), "reward": float(utility(q_obj, S)),
                                    "tools": S}

        # --- Category A2: classic & hybrid retrieval baselines (BM25 / dense cosine / hybrid; 0 search) ---
        retrieval_fns = {"bm25": method_bm25, "dense": method_dense, "hybrid": method_hybrid}
        for arm in retrieval_arms:
            S, _ = retrieval_fns[arm](index, q_obj, cfg, None, utility, budget)
            rec[f"decode_{arm}"] = {"f1": set_f1(S, gold_ids), "recall": set_recall(S, gold_ids),
                                    "ndcg": set_ndcg(S, gold_ids), "reward": float(utility(q_obj, S)),
                                    "tools": S}

        # --- Category B: re-evolution ((P+1)*G judge calls per arm) ---
        for arm in evo_arms:
            rng_arm = spawn_rng(args.seed, f"test:{qid}:{arm}")
            _, r_best, S_best = run_query_evolution(
                index, q_obj, utility, arm_alphas[arm], cfg, rng_arm, args.G, args.P, executor)
            rec[f"evo_{arm}"] = {"f1": set_f1(S_best, gold_ids), "recall": set_recall(S_best, gold_ids),
                                 "ndcg": set_ndcg(S_best, gold_ids), "reward": float(r_best),
                                 "tools": S_best}

        ckpt_mutate(lambda: (test_results.append(rec), tested_qids.add(qid)))
        prog = " ".join(f"{k.replace('decode_', 'd').replace('evo_', 'e')}={rec[k]['f1']:.3f}"
                        for k in ("decode_warm", "decode_default", "decode_cold",
                                  "evo_warm", "evo_default", "evo_cold") if k in rec)
        plog(f"cat {true_cid} {qid:<24s} {'OK' if is_match else 'XX':<3s} | {prog}")

    with ThreadPoolExecutor(max_workers=args.query_parallel) as q_pool:
        list(q_pool.map(eval_test_item, list(zip(test_items, q_embs))))

    executor.shutdown(wait=True)

    # -------------------------------------------------------------
    # Phase 4: Final Summary (derived entirely from checkpointed test results)
    # -------------------------------------------------------------
    total_test = len(test_results)
    correct_matches = sum(1 for r in test_results if r["match_success"])
    match_acc = correct_matches / max(total_test, 1)
    n_evals_per_evo = args.P * args.G + args.G

    print("\n" + "=" * 100)
    print("FINAL EXPERIMENTAL SUMMARY REPORT")
    print(f"Catalog Routing Accuracy: {correct_matches}/{total_test} ({match_acc * 100:.1f}%)  |  "
          f"N test queries: {total_test}")
    print("=" * 100)
    print(f"{'Category':<8s} {'Method':<40s} {'F1@5':<8s} {'Recall@5':<9s} {'NDCG@5':<8s} {'J-Reward':<9s} Cost")
    print("-" * 100)

    ARM_LABELS = {
        "bm25":    "BM25 top-5 (lexical baseline)",
        "dense":   "Dense cosine top-5 (dense baseline)",
        "hybrid":  "Hybrid BM25+Dense top-5 (retrieval)",
        "default": "Global Default (norm. [0.80, 0.16, 0.04])",
        "cold":    "Cold a0 (random init)",
        "tail":    "Tail a30 (v1 prior, ablation)",
        "warm":    "Amortized a_mean (Ours, Polyak)",
    }
    arm_labels = ([("decode", a, ARM_LABELS[a], "0 LLM") for a in retrieval_arms]
                  + [("decode", a, ARM_LABELS[a], "0 LLM") for a in decode_arms]
                  + [("evo", a, f"{ARM_LABELS[a]} -> ES", f"{n_evals_per_evo} LLM") for a in evo_arms])
    available = set(test_results[0].keys()) if test_results else set()
    aggregate = {}
    for prefix, arm, label, cost in arm_labels:
        s = arm_stats(test_results, prefix, arm)
        aggregate[f"{prefix}_{arm}"] = s
        cat_label = "decode" if prefix == "decode" else "evo"
        print(f"{cat_label:<8s} {label:<40s} {s['f1']:<8.4f} {s['recall']:<9.4f} {s['ndcg']:<8.4f} {s['reward']:<9.4f} {cost}")

    paired_specs = {
        "evo_warm_vs_cold": ("evo_warm", "evo_cold"),
        "evo_warm_vs_default": ("evo_warm", "evo_default"),
        "decode_warm_vs_default": ("decode_warm", "decode_default"),
        "decode_warm_vs_tail": ("decode_warm", "decode_tail"),
        "decode_warm_vs_bm25": ("decode_warm", "decode_bm25"),
        "decode_warm_vs_hybrid": ("decode_warm", "decode_hybrid"),
        "evo_warm_vs_decode_warm": ("evo_warm", "decode_warm"),
    }
    stats = {name: paired(test_results, a, b) for name, (a, b) in paired_specs.items()
             if a in available and b in available}
    rews = [r[f"evo_{a}"]["reward"] for r in test_results for a in evo_arms]
    f1s_ = [r[f"evo_{a}"]["f1"] for r in test_results for a in evo_arms]
    corr = float(np.corrcoef(rews, f1s_)[0, 1]) if len(set(rews)) > 1 else float("nan")

    print("-" * 100)
    for name, p in stats.items():
        print(f"paired {name}: diff={p['mean_diff']:+.4f}  W/T/L={p['win']}/{p['tie']}/{p['lose']}  t={p['t']:+.2f}")
    print(f"corr(judge reward, gold F1) over evolution methods: {corr:.3f}")
    print("=" * 100)

    print(f"\nPER-CATALOG F1@5 ({' / '.join(k for k in ('decode_warm', 'evo_warm', 'evo_cold') if k in available)})")
    for cat in catalogs:
        cid = str(cat["catalog_id"])
        rows = [r for r in test_results if r["true_catalog"] == cid]
        if not rows:
            continue
        cols = []
        for k in ("decode_warm", "evo_warm", "evo_cold"):
            if k in available:
                cols.append(f"{np.mean([r[k]['f1'] for r in rows]):.4f}")
        print(f"  Catalog {cid} {cat['title_en'][:34]:<36s} (n={len(rows)}): {' / '.join(cols)}")

    # Save JSON + Markdown report
    full_output = {"config": signature, "catalog_routing_accuracy": match_acc,
                   "aggregate": aggregate, "paired_stats": stats,
                   "judge_gold_corr": corr, "per_query_results": test_results}
    with open(results_dir / f"comparison_results{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    md = [
        "# Continuous Domain-Amortized Evolution 实验对比报告 (v2)",
        "",
        "## 实验设定",
        f"- **评估模式**: `{args.eval_mode}` (`{cfg.llm.model if args.eval_mode == 'judge' else 'local'}`, 训练/测试零 gold 泄露)",
        f"- **演化超参数**: P={args.P}, G={args.G}, alpha 约束于单纯形 (sum=1), sigma=sigma0*||alpha||",
        f"- **摊销先验**: 训练轨迹 Polyak 均值 a_mean (v1 使用尾部 a30, 仅作消融)",
        f"- **数据**: {len(catalogs)} Catalog, 每类训练 {args.train_limit} 条 / 测试 {args.test_limit} 条; 路由准确率 **{match_acc*100:.1f}%** ({correct_matches}/{total_test})",
        f"- **judge-gold 相关性**: {corr:.3f}",
        "",
        "## 核心指标对比表",
        "",
        "| 类别 | 方法 | F1@5 | Recall@5 | NDCG@5 | Judge Reward | 测试开销 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for prefix, arm, label, cost in arm_labels:
        s = aggregate[f"{prefix}_{arm}"]
        md.append(f"| {'直接解码' if prefix == 'decode' else '再演化'} | {label} | {s['f1']:.4f} | "
                  f"{s['recall']:.4f} | {s['ndcg']:.4f} | {s['reward']:.4f} | {cost} |")
    md += ["", "## 配对检验 (F1@5)"]
    md_labels = {
        "evo_warm_vs_cold": "再演化 amortized vs cold",
        "evo_warm_vs_default": "再演化 amortized vs default",
        "decode_warm_vs_default": "解码 amortized vs global default",
        "decode_warm_vs_tail": "解码 amortized (mean) vs tail",
        "decode_warm_vs_bm25": "解码 amortized vs BM25",
        "decode_warm_vs_hybrid": "解码 amortized vs Hybrid",
        "evo_warm_vs_decode_warm": "再演化 vs 直接解码 (amortized)",
    }
    for name, p in stats.items():
        line = f"- {md_labels.get(name, name)}: diff={p['mean_diff']:+.4f}"
        if name == "evo_warm_vs_cold":
            line += f" (W/T/L = {p['win']}/{p['tie']}/{p['lose']}, t={p['t']:+.2f})"
        md.append(line)
    with open(results_dir / f"EXPERIMENT_REPORT{suffix}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nSaved: comparison_results{suffix}.json, EXPERIMENT_REPORT{suffix}.md, catalog_learned_alphas{suffix}.json")


if __name__ == "__main__":
    main()
