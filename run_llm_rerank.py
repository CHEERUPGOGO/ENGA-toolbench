"""LLM-rerank baseline: hybrid top-50 candidates -> one LLM selection call picks 5.

Complements the zero-cost retrieval baselines and the 45-call re-evolution arm:
this baseline spends exactly 1 selection LLM call per query (plus 1 blind judge
scoring call for the reward column, same as every decode arm).

Run AFTER any judge-mode experiment driver finishes (it appends to the shared
results/llm_cache.jsonl via LLMJudgeUtility; two processes writing it concurrently
are not coordinated).

Usage:
    python run_llm_rerank.py --splits data/catalog_splits10.json --tag 10cat
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ENGA_ROOT = ROOT if (ROOT / "enga").is_dir() else ROOT.parent / "experiments"
sys.path.insert(0, str(ENGA_ROOT))

from enga.config import Config
from enga.data import Query, build_experiment_set, load_toolret
from enga.evaluator import EvalBudget, LLMJudgeUtility, set_f1, set_ndcg, set_recall
from enga.features import ToolIndex

SELECTION_CACHE = ROOT / "results" / "rerank_selection_cache.jsonl"
_cache_lock = threading.Lock()
_selection_cache: dict[str, str] = {}
if SELECTION_CACHE.exists():
    with open(SELECTION_CACHE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                p_list = json.loads(row["picked"])
                if len(p_list) == 5:
                    _selection_cache[row["key"]] = row["picked"]
            except Exception:
                pass


def selection_prompt(q: Query, cands: list[str], tools: dict) -> str:
    lines = [
        "You are selecting tools for an LLM agent. From the candidate tools below,",
        "pick the FIVE that together best solve the user query.",
        "Output ONLY a valid JSON object without markdown fences, formatted exactly as:",
        '{"picked": ["id1", "id2", "id3", "id4", "id5"]}',
        "",
        f"User query: {q.text}",
        "",
        "Candidate tools:",
    ]
    for tid in cands:  # alphabetical -> position-bias neutral, cache-stable
        t = tools[tid]
        req = ", ".join(p.name for p in t.required_params)
        desc = (t.description or "")[:200].replace("\n", " ")
        lines.append(f"- {tid} | {t.name} | category: {t.category or 'n/a'} | required: {req} | {desc}")
    return "\n".join(lines)


def parse_picked_tools(raw: str, cands: list[str]) -> list[str]:
    text = raw
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    cand_set = set(cands)
    m_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate_json_str = m_block.group(1) if m_block else None
    if not candidate_json_str:
        m_json = re.search(r"(\{.*\"picked\".*?\})", text, re.S)
        candidate_json_str = m_json.group(1) if m_json else text
    picked = []
    try:
        data = json.loads(candidate_json_str)
        if isinstance(data, dict) and "picked" in data:
            picked = [str(x) for x in data["picked"] if str(x) in cand_set]
    except Exception:
        pass
    if len(picked) < 5:
        found = re.findall(r'["\']([a-zA-Z0-9_\-]+)["\']', text)
        seen = set()
        picked = []
        for x in found:
            if x in cand_set and x not in seen:
                seen.add(x)
                picked.append(x)
                if len(picked) == 5:
                    break
    return picked[:5]


def pick_five(client, cfg: Config, q: Query, cands: list[str], tools: dict) -> tuple[list[str], bool, bool]:
    """One selection call; returns (tool ids, from_cache, used_fallback).
    Falls back to hybrid top-5 on parse failure or a short/invalid list."""
    key = f"rerank::{cfg.llm.model}::{q.qid}::{','.join(cands)}"
    with _cache_lock:
        if key in _selection_cache:
            raw = _selection_cache[key]
            cached = True
        else:
            raw = None
            cached = False
    if raw is None:
        resp = client.chat.completions.create(
            model=cfg.llm.model,
            messages=[{"role": "user", "content": selection_prompt(q, cands, tools)}],
            temperature=0, max_tokens=2048, timeout=cfg.llm.timeout,
        )
        raw = resp.choices[0].message.content or ""
        picked = parse_picked_tools(raw, cands)
        raw = json.dumps(picked)  # store the cleaned list
        if len(picked) == 5:
            with _cache_lock:
                _selection_cache[key] = raw
                with open(SELECTION_CACHE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "picked": raw}, ensure_ascii=False) + "\n")
        cached = False
    else:
        picked = json.loads(raw)
    if len(picked) < 5:                      # parse failure / short list -> fallback
        return list(cands[:5]), cached, True
    return picked, cached, False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=str, default="data/catalog_splits.json")
    ap.add_argument("--tag", type=str, default="", help="suffix for the output file")
    ap.add_argument("--pool-seed", type=int, default=42,
                    help="must match the compared run (same sub-library)")
    ap.add_argument("--top-k", type=int, default=50, help="candidate pool size for rerank")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    with open(ROOT / args.splits, "r", encoding="utf-8") as f:
        catalogs = json.load(f)

    cfg = Config()
    print(f"endpoint : {cfg.llm.base_url} | model: {cfg.llm.model}")

    all_raw_tools, _ = load_toolret(str(ENGA_ROOT / "data" / "toolret"), "toolbench", "web")
    all_queries = [Query(qid=q["id"], text=q["query"], gold_ids=[t["id"] for t in q.get("tools", [])])
                   for cat in catalogs for q in cat["train_queries"] + cat["test_queries"]]
    sub_tools, qs, ids = build_experiment_set(all_raw_tools, all_queries, len(all_queries),
                                              n_tools=3000, seed=args.pool_seed)
    index = ToolIndex(sub_tools, ids, cfg, query_texts=[q.text for q in qs])
    print(f"Index built: {index.n} tools")

    budget = EvalBudget()
    judge = LLMJudgeUtility(cfg, index.tools, budget)

    test_items = [(str(cat["catalog_id"]), q) for cat in catalogs for q in cat["test_queries"]]
    print_lock = threading.Lock()

    def run_one(item):
        _, q_dict = item
        q = Query(qid=q_dict["id"], text=q_dict["query"],
                  gold_ids=[t["id"] for t in q_dict.get("tools", [])])
        scores = index.hybrid_norm(q)          # aligned with index.ids (same as method_hybrid)
        order = [ids[int(i)] for i in np.argsort(-scores)]
        cands = order[:args.top_k]
        picked, cached, fallback = pick_five(judge._client_(), cfg, q, cands, index.tools)
        rec = {"qid": q.qid, "gold_size": len(q.gold_ids), "from_cache": cached,
               "fallback": fallback,
               "f1": set_f1(picked, q.gold_ids), "recall": set_recall(picked, q.gold_ids),
               "ndcg": set_ndcg(picked, q.gold_ids), "reward": float(judge(q, picked)),
               "tools": picked}
        with print_lock:
            print(f"{q.qid:<26s} cache={int(cached)} fb={int(fallback)} F1={rec['f1']:.3f} "
                  f"R={rec['recall']:.3f} N={rec['ndcg']:.3f} rew={rec['reward']:.3f}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(run_one, test_items))

    agg = {m: float(np.mean([r[m] for r in records])) for m in ("f1", "recall", "ndcg", "reward")}
    n_cached = sum(1 for r in records if r["from_cache"])
    n_fb = sum(1 for r in records if r["fallback"])
    print("\nLLM-rerank (hybrid top-%d -> LLM picks 5) over %d queries" % (args.top_k, len(records)))
    print(f"F1@5={agg['f1']:.4f}  Recall@5={agg['recall']:.4f}  NDCG@5={agg['ndcg']:.4f}  "
          f"JudgeReward={agg['reward']:.4f}  selection calls: {len(records)-n_cached} new / "
          f"{n_cached} cached | parse fallbacks: {n_fb}")
    out = {"config": {"splits": args.splits, "pool_seed": args.pool_seed, "top_k": args.top_k,
                      "model": cfg.llm.model, "selection_prompt": "rerank1"},
           "aggregate": agg, "n_new_selection_calls": len(records) - n_cached,
           "per_query": records}
    with open(ROOT / "results" / f"llm_rerank{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved: results/llm_rerank{suffix}.json | judge budget: {budget.count} calls")


if __name__ == "__main__":
    main()
