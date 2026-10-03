"""Composition utility U(q, S) and the expensive-evaluation budget (Method 3.4).

Two interchangeable modes:
  oracle : U from the gold annotation of ToolRet/ToolBench (F1 / Recall / NDCG of S
           vs gold tools). Free and reproducible; used to run the full pipeline and
           as the oracle inside true-NGA (Preliminary Alg. 1).
  llm    : the real U_LLM -- an OpenAI-compatible model is prompted with the query
           and the tool docs of S and must identify which tools it would call;
           the answer is scored against gold. Every call increments the budget
           counter, enabling budget-fair comparisons (RQ3).
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

import numpy as np

from .config import Config
from .data import Query, ToolDoc


class EvalBudget:
    """B_LLM counter (Eq. 18): number of expensive evaluations spent so far."""

    def __init__(self, limit: int = 0):
        self.limit = int(limit)
        self.count = 0

    def exhausted(self) -> bool:
        return self.limit > 0 and self.count >= self.limit

    def spend(self, k: int = 1) -> None:
        self.count += k


def _with_retries(fn, tries: int = 3, backoff: float = 5.0):
    """Retry transient API errors (429/5xx/timeouts) with linear backoff."""
    import time
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "401" in msg or "authentication" in msg.lower():
                raise
            last = e
            time.sleep(backoff * (i + 1))
    raise last


# ------------------------------------------------------------------ set metrics
def set_f1(pred: list[str], gold: list[str]) -> float:
    if not pred or not gold:
        return 1.0 if not pred and not gold else 0.0
    inter = len(set(pred) & set(gold))
    p = inter / len(set(pred))
    r = inter / len(set(gold))
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def set_recall(pred: list[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    return len(set(pred) & set(gold)) / len(set(gold))


def set_ndcg(pred: list[str], gold: list[str]) -> float:
    if not gold or not pred:
        return 0.0
    gains = np.array([1.0 if t in set(gold) else 0.0 for t in pred])
    disc = 1.0 / np.log2(np.arange(2, len(pred) + 2))
    dcg = float((gains * disc).sum())
    ideal = float((np.ones(min(len(gold), len(pred))) * disc[:min(len(gold), len(pred))]).sum())
    return dcg / ideal if ideal > 0 else 0.0


_METRICS = {"f1": set_f1, "recall": set_recall, "ndcg": set_ndcg}


class OracleUtility:
    """U(q, S) from gold annotations. `count_calls=True` makes it behave like the
    expensive oracle of Preliminary Alg. 1 (each call spends budget)."""

    def __init__(self, cfg: Config, count_calls: bool = False, budget: EvalBudget | None = None):
        self.fn = _METRICS[cfg.utility]
        self.count_calls = count_calls
        self.budget = budget

    def __call__(self, q: Query, S: list[str]) -> float:
        if self.count_calls:
            if self.budget is not None:
                if self.budget.exhausted():
                    return 0.0
                self.budget.spend()
        return self.fn(S, q.gold_ids)


# ------------------------------------------------------------------- LLM oracle
_PROMPT = """You are a tool-using assistant. Given a user query and a shortlist of candidate API tools, decide which tools are needed to fully answer the query.

## Query
{query}

## Candidate tools
{tools}

## Task
Output ONLY a JSON array with the ids of the tools that should be called, e.g. ["id1","id2"]. If none apply, output [].
"""


class LLMUtility:
    """U_LLM(q, S): one chat completion per composition; every call counts against B_LLM."""

    def __init__(self, cfg: Config, tools: dict[str, ToolDoc], budget: EvalBudget):
        self.cfg = cfg
        self.tools = tools
        self.budget = budget
        self.fn = _METRICS[cfg.utility]
        self.cache: dict[str, float] = {}
        self.cache_path = Path(cfg.llm.cache_path)
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        k, v = json.loads(line)
                        self.cache[k] = v
        self._client = None
        self._lock = None  # created lazily; guards cache writes for threaded callers
        self._tok_usage = {"prompt": 0, "completion": 0}

    def _client_(self):
        if self._client is None:
            if not self.cfg.llm.base_url:
                raise RuntimeError("LLM eval mode requires --llm-base-url "
                                   "(any OpenAI-compatible endpoint, e.g. local vLLM).")
            from openai import OpenAI
            self._client = OpenAI(base_url=self.cfg.llm.base_url,
                                  api_key=self.cfg.llm.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
        return self._client

    @staticmethod
    def _strip_think(text: str) -> str:
        """Reasoning models (e.g. MiniMax-M2) emit <think>...</think> first."""
        if "</think>" in text:
            return text.rsplit("</think>", 1)[1]
        return text

    def _key(self, q: Query, S: list[str]) -> str:
        return f"{self.cfg.llm.model}::{q.qid}::{','.join(sorted(S))}"

    def _render(self, q: Query, S: list[str]) -> str:
        lines = []
        for i, tid in enumerate(S):
            d = self.tools.get(tid)
            if d is None:
                continue
            req = "; ".join(f"{p.name}({p.type})" for p in d.required_params)
            lines.append(f"[{i}] id={d.id} | name={d.name} | category={d.category} | "
                         f"method={d.method} | required: {req} | desc: {d.description[:300]}")
        return _PROMPT.format(query=q.text, tools="\n".join(lines))

    def __call__(self, q: Query, S: list[str]) -> float:
        if self.budget.exhausted():
            return 0.0
        key = self._key(q, S)
        if key in self.cache:
            return self.cache[key]
        prompt = self._render(q, S)
        resp = _with_retries(lambda: self._client_().chat.completions.create(
            model=self.cfg.llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.llm.temperature,
            max_tokens=self.cfg.llm.max_tokens,
            timeout=self.cfg.llm.timeout,
        ))
        self.budget.spend()
        u = getattr(resp, "usage", None)
        if u is not None:
            self._tok_usage["prompt"] += getattr(u, "prompt_tokens", 0)
            self._tok_usage["completion"] += getattr(u, "completion_tokens", 0)
        text = self._strip_think(resp.choices[0].message.content or "[]")
        m = re.search(r"\[.*?\]", text, re.S)
        try:
            chosen = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            chosen = []
        # accept both "id=" values and array indices
        sid = set(S)
        picked = [c for c in chosen if isinstance(c, str)]
        if picked and not any(c in sid for c in picked):
            picked = [S[int(c)] for c in picked if isinstance(c, (int, str)) and str(c).lstrip("-").isdigit()
                      and 0 <= int(c) < len(S)]
        r = self.fn(picked, q.gold_ids)
        if not chosen and getattr(resp.choices[0], "finish_reason", None) == "length":
            return r  # truncated before any answer: don't poison the cache with 0.0
        if self._lock is None:
            import threading
            self._lock = threading.Lock()
        with self._lock:
            self.cache[key] = r
            self._append_cache(key, r)
        return r

    def _append_cache(self, key: str, r: float) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps([key, r]) + "\n")


# ------------------------------------------------------------------- Zero-Shot LLM Judge (No Gold Leakage)
_JUDGE_PROMPT = """You are an expert AI agent evaluator.
Given a user query and a candidate tool set, evaluate how well this tool set enables an AI agent to solve the query.

## User Query:
{query}

## Candidate Tool Set:
{tools}

## Evaluation Criteria:
1. Completeness: Does this toolset cover the required capabilities to solve the query? (0.0 to 1.0)
2. Relevance: Are the tools relevant without distracting or useless clutter? (0.0 to 1.0)
3. Redundancy: Are there redundant tools duplicating each other? (0.0 to 1.0)

Output ONLY a valid JSON object formatted as:
{{
  "completeness": <float between 0.0 and 1.0>,
  "relevance": <float between 0.0 and 1.0>,
  "redundancy_penalty": <float between 0.0 and 1.0>,
  "utility": <float between 0.0 and 1.0>,
  "thought": "<brief 1-sentence reasoning>"
}}
"""


class LLMJudgeUtility:
    """U_judge(q, S): Zero-shot LLM-as-a-Judge.
    Evaluates the sufficiency, relevance, and non-redundancy of tool set S for query q
    WITHOUT ANY ACCESS TO q.gold_ids (completely zero-shot unsupervised during search)."""

    def __init__(self, cfg: Config, tools: dict[str, ToolDoc], budget: EvalBudget):
        self.cfg = cfg
        self.tools = tools
        self.budget = budget
        self.cache: dict[str, float] = {}
        self.cache_path = Path(cfg.llm.cache_path)
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            k, v = json.loads(line)
                            self.cache[k] = float(v)
                        except Exception:
                            pass
        self._client = None
        self._init_lock = threading.Lock()
        self._lock = threading.Lock()
        self._tok_usage = {"prompt": 0, "completion": 0}

    def _client_(self):
        if self._client is None:
            with self._init_lock:
                if self._client is None:
                    if not self.cfg.llm.base_url:
                        raise RuntimeError("LLM eval mode requires --llm-base-url (OpenAI-compatible endpoint).")
                    from openai import OpenAI
                    self._client = OpenAI(
                        base_url=self.cfg.llm.base_url,
                        api_key=self.cfg.llm.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                    )
        return self._client

    @staticmethod
    def _strip_think(text: str) -> str:
        if "</think>" in text:
            return text.rsplit("</think>", 1)[1]
        return text

    def _key(self, q: Query, S: list[str]) -> str:
        # "judge2" namespace = order-neutral rendering (sorted tool list); keeps the
        # old judge:: scores from being mixed with the new prompt semantics.
        return f"judge2::{self.cfg.llm.model}::{q.qid}::{','.join(sorted(S))}"

    def _render(self, q: Query, S: list[str]) -> str:
        lines = []
        # Sorted (order-neutral) rendering: kills LLM position bias and matches the
        # sorted cache key, so the same set always sees the same prompt.
        for i, tid in enumerate(sorted(S)):
            d = self.tools.get(tid)
            if d is None:
                continue
            req = "; ".join(f"{p.name}({p.type})" for p in d.required_params)
            lines.append(f"[{i}] id={d.id} | name={d.name} | category={d.category} | "
                         f"required: {req} | desc: {d.description[:300]}")
        return _JUDGE_PROMPT.format(query=q.text, tools="\n".join(lines))

    def __call__(self, q: Query, S: list[str]) -> float:
        if not S:
            return 0.0
        if self.budget.exhausted():
            return 0.0
        key = self._key(q, S)
        if key in self.cache:
            return self.cache[key]

        prompt = self._render(q, S)
        resp = _with_retries(lambda: self._client_().chat.completions.create(
            model=self.cfg.llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.llm.temperature,
            max_tokens=self.cfg.llm.max_tokens,
            timeout=self.cfg.llm.timeout,
        ))
        self.budget.spend()
        u = getattr(resp, "usage", None)
        if u is not None:
            self._tok_usage["prompt"] += getattr(u, "prompt_tokens", 0)
            self._tok_usage["completion"] += getattr(u, "completion_tokens", 0)

        raw_text = resp.choices[0].message.content or "{}"
        clean_text = self._strip_think(raw_text)

        utility = 0.0
        m = re.search(r"\{.*?\}", clean_text, re.S)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if "utility" in parsed:
                    utility = float(parsed["utility"])
                else:
                    comp = float(parsed.get("completeness", 0.5))
                    rel = float(parsed.get("relevance", 0.5))
                    red = float(parsed.get("redundancy_penalty", 0.0))
                    utility = max(0.0, min(1.0, 0.7 * comp + 0.3 * rel - 0.2 * red))
            except Exception:
                utility = 0.1
        else:
            m_num = re.search(r'"utility"\s*:\s*([0-9.]+)', clean_text)
            if m_num:
                utility = float(m_num.group(1))
            else:
                utility = 0.1

        utility = max(0.0, min(1.0, float(utility)))

        with self._lock:
            self.cache[key] = utility
            self._append_cache(key, utility)
        return utility

    def _append_cache(self, key: str, r: float) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps([key, r]) + "\n")


# ------------------------------------------------------------------- Fast Unsupervised Proxy (No Gold Leakage)
class UnsupervisedUtility:
    """U_unsupervised(q, S): Deterministic zero-shot proxy for fast offline testing.
    Computes intrinsic semantic relevance, category/schema synergy, and redundancy penalty
    WITHOUT ANY ACCESS TO q.gold_ids."""

    def __init__(self, cfg: Config, index, count_calls: bool = False,
                 budget: EvalBudget | None = None):
        self.cfg = cfg
        self.index = index
        self.count_calls = count_calls
        self.budget = budget

    def __call__(self, q: Query, S: list[str]) -> float:
        if not S:
            return 0.0
        if self.count_calls and self.budget is not None:
            if self.budget.exhausted():
                return 0.0
            self.budget.spend()

        valid_idxs = [self.index.pos[t] for t in S if t in self.index.pos]
        if not valid_idxs:
            return 0.0

        # 1. Relevance to query (hybrid lexical + dense when enabled)
        q_rels = self.index.rel_norm(q)[valid_idxs]
        mean_rel = float(np.mean(q_rels))
        max_rel = float(np.max(q_rels))

        # 2. Pairwise redundancy among selected tools
        if len(valid_idxs) > 1:
            embs = self.index.emb[valid_idxs]
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            normed = embs / np.maximum(norms, 1e-8)
            sim_mat = np.dot(normed, normed.T)
            np.fill_diagonal(sim_mat, 0.0)
            max_dup = float(np.max(sim_mat))
            dup_penalty = max(0.0, (max_dup - 0.90) / 0.10) if max_dup > 0.90 else 0.0
        else:
            dup_penalty = 0.0

        # 3. Synergy across selected tools
        syn_scores = []
        for i in range(1, len(valid_idxs)):
            sub_sel = valid_idxs[:i]
            tgt = valid_idxs[i]
            step_syn = self.index.cfg.lambda_s * self.index.sem_syn(sub_sel)[tgt] + \
                       (1.0 - self.index.cfg.lambda_s) * self.index.sch_syn(sub_sel)[tgt]
            syn_scores.append(float(step_syn))
        mean_syn = float(np.mean(syn_scores)) if syn_scores else 0.0

        # 4. Execution cost penalty
        mean_cost = float(np.mean(self.index.cost[valid_idxs]))

        # Composite score
        u = 0.4 * max_rel + 0.3 * mean_rel + 0.3 * mean_syn - 0.15 * dup_penalty - 0.15 * mean_cost
        return max(0.0, min(1.0, u))

