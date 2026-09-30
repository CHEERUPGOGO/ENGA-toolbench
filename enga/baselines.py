"""All selection methods, exposed through one dispatcher used by run_experiment.

Methods
  random        uniform sample of b tools
  bm25          BM25 top-b (classic lexical retrieval)
  dense         embedding cosine top-b  (== NGA with alpha = [1, 0, 0])
  nga_fixed     parameterized NGA with the default alpha (no search)   [ablation]
  nga_true      Preliminary Alg. 1: greedy with TRUE oracle marginal gain
                -> O(N*b) expensive evaluations, the cost ENGA avoids
  llm_greedy    LLM picks the next tool given query + already-selected tools
                (expensive inner loop; in oracle mode aliases to nga_true)
  enega         full method: ES over alpha + NGA decoding + QD archive
  + ablations   enega_noES / enega_noQD / enega_nosyn / enega_nocost
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .data import Query, ToolDoc
from .es import enaga_search
from .evaluator import EvalBudget, OracleUtility, LLMUtility
from .features import ToolIndex
from .nga import decode, decode_oracle_nga
from .qd import Archive


def _topk(index: ToolIndex, scores: np.ndarray, b: int) -> list[str]:
    idx = np.argsort(-scores)[:b]
    return [index.ids[int(i)] for i in idx]


def method_random(index: ToolIndex, q: Query, cfg: Config, rng, utility, budget):
    sel = rng.choice(index.n, size=min(cfg.b, index.n), replace=False)
    return [index.ids[int(i)] for i in sel], {"n_expensive": 0}


def method_dense(index: ToolIndex, q: Query, cfg: Config, rng, utility, budget):
    return _topk(index, index.rel(q), cfg.b), {"n_expensive": 0}


def method_bm25(index: ToolIndex, q: Query, cfg: Config, rng, utility, budget):
    return _topk(index, index.bm25_scores(q.text), cfg.b), {"n_expensive": 0}


def method_nga_fixed(index: ToolIndex, q: Query, cfg: Config, rng, utility, budget):
    S = decode(index, q, cfg.init_alpha, cfg.b)
    return S, {"n_expensive": 0}


def method_nga_true(index: ToolIndex, q: Query, cfg: Config, rng, utility, budget):
    """Alg. 1 with the real oracle; each candidate scoring spends budget."""
    counter: list = []
    S = decode_oracle_nga(index, q, utility, cfg.b,
                          candidate_cap=cfg.candidate_cap, budget=budget,
                          n_expensive=counter)
    return S, {"n_expensive": len(counter)}


def _greedy_step_cache():
    """Persistent cache for llm_greedy selection steps, so crashes/re-runs are free."""
    import json as _json
    from pathlib import Path as _Path
    path = _Path("results/llm_greedy_cache.jsonl")
    cache: dict[str, int] = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                k, v = _json.loads(line)
                cache[k] = v
    return path, cache


def method_llm_greedy(index: ToolIndex, q: Query, cfg: Config, rng, utility, budget):
    """LLM inside the greedy loop: 'which of these candidates should be added next?'"""
    if not isinstance(utility, LLMUtility):
        return method_nga_true(index, q, cfg, rng, utility, budget)
    import hashlib
    import json as jsonlib
    from openai import OpenAI
    client = utility._client_()
    rel = index.rel(q)
    sel: list[str] = []
    n_calls = 0
    cache_path, step_cache = _greedy_step_cache()
    for t in range(cfg.b):
        if budget.exhausted():
            break
        cand = [i for i in np.argsort(-rel) if index.ids[int(i)] not in sel][:cfg.candidate_cap]
        cand = [int(c) for c in cand]
        if not cand:
            break
        lines = [f"[{k}] id={index.ids[i]} | {index.doc_texts[i][:160]}" for k, i in enumerate(cand)]
        chosen_prev = "\n".join(f"- {s}" for s in sel) or "(none yet)"
        key = hashlib.sha1(("::".join(
            [cfg.llm.model, q.qid, str(t)] + list(sel) + [index.ids[i] for i in cand])
        ).encode()).hexdigest()
        if key in step_cache:
            k = step_cache[key]
            if not (0 <= k < len(cand)):
                break
            sel.append(index.ids[cand[k]])
            continue
        prompt = (
            f"Query: {q.text}\n\nAlready selected tools:\n{chosen_prev}\n\n"
            f"Which ONE candidate tool should be added next to best complete the task?\n"
            + "\n".join(lines)
            + "\n\nAnswer with ONLY the bracket index, e.g. [3]. No explanation."
        )
        resp = client.chat.completions.create(
            model=cfg.llm.model,
            messages=[
                # keeps the reasoning block short on thinking models (e.g. MiniMax-M2)
                {"role": "system", "content": "Answer immediately with the final result only. Minimal reasoning."},
                {"role": "user", "content": prompt},
            ],
            temperature=cfg.llm.temperature,
            max_tokens=4096,  # reasoning models: the <think> block alone can exceed 1024
            timeout=cfg.llm.timeout,
        )
        budget.spend()
        n_calls += 1
        raw = (resp.choices[0].message.content or "").strip()
        text = raw.rsplit("</think>", 1)[1].strip() if "</think>" in raw else raw
        import re
        # post-think answer first; if truncated before an answer appeared, fall back to
        # the last bracket index anywhere in the text (conclusions sit at the end)
        m = re.search(r"\[?(-?\d+)\]?", text) or re.search(r"\[(\d+)\]", raw)
        if not m:
            break
        k = int(m.group(1))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(jsonlib.dumps([key, k]) + "\n")
            step_cache[key] = k
        if not (0 <= k < len(cand)):
            break
        sel.append(index.ids[cand[k]])
    return sel, {"n_expensive": n_calls}


def make_enega(use_es=True, use_qd=True, use_syn=True, use_cost=True):
    def run(index, q, cfg, rng, utility, budget):
        mu, archive, rec = enaga_search(index, q, utility, cfg, rng, budget=budget,
                                        use_es=use_es, use_syn=use_syn, use_cost=use_cost,
                                        use_qd=use_qd)
        return archive, rec
    return run


METHODS = {
    "random": method_random,
    "bm25": method_bm25,
    "dense": method_dense,
    "nga_fixed": method_nga_fixed,
    "nga_true": method_nga_true,
    "llm_greedy": method_llm_greedy,
    "enega": make_enega(),
    "enega_noES": make_enega(use_es=False),
    "enega_noQD": make_enega(use_qd=False),
    "enega_nosyn": make_enega(use_syn=False),
    "enega_nocost": make_enega(use_cost=False),
}

# Methods whose primary output is a portfolio (archive) rather than a single set.
PORTFOLIO_METHODS = {"enega", "enega_noES", "enega_noQD", "enega_nosyn", "enega_nocost"}

DEFAULT_RUN_ORDER = [
    "random", "bm25", "dense", "nga_fixed",
    "nga_true", "llm_greedy",
    "enega", "enega_noES", "enega_noQD", "enega_nosyn", "enega_nocost",
]
