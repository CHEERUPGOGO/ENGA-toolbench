"""Experiment configuration (dataclass + YAML/CLI overrides)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class LLMConfig:
    """OpenAI-compatible endpoint for the real U_LLM evaluator (Method 3.4)."""
    base_url: str = "https://api.deepseek.com"           # OpenAI-compatible endpoint (DeepSeek)
    api_key: str = ""                                     # set via OPENAI_API_KEY env var, do NOT hardcode
    model: str = "deepseek-v4-flash"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 120
    mode: str = "select"        # "select": pick needed tool ids from S; judged vs gold
    cache_path: str = "results/llm_cache.jsonl"


@dataclass
class Config:
    # ---------- data ----------
    data_dir: str = "data/toolret"
    source: str = "toolbench"     # ToolRet subset for queries ("raw" -> raw ToolBench jsons)
    library: str = "web"          # tool library (web | code | customized)
    raw_split: str = "G1_test"    # used when source == "raw"
    n_queries: int = 50           # queries sampled for the experiment
    n_tools: int = 5000           # sub-library size N (gold tools always included)
    seed: int = 0

    # ---------- NGA decoder (Method 3.2) ----------
    b: int = 5                    # tool budget b
    lambda_s: float = 0.5         # semantic/schema synergy mix (Eq. 6)
    cost_mode: str = "synthetic"  # "schema": token footprint only; "synthetic": + hashed latency/failure
    lambda_t: float = 0.34        # token footprint weight  (Eq. 9)
    lambda_tau: float = 0.33      # latency weight
    lambda_f: float = 0.33        # failure-rate weight

    # ---------- ES (Method 3.3) ----------
    P: int = 8                    # population size (policies per generation)
    G: int = 5                    # generations
    eta: float = 0.7              # ES learning rate (Eq. 15)
    sigma0: float = 0.4
    sigma_decay: float = 0.85
    init_alpha: tuple = (1.0, 0.2, 0.05)
    antithetic: bool = True
    eval_mu: bool = True          # elitist anchor: also evaluate D(q; mu) each generation (+1 eval/gen)

    # ---------- QD archive (Method 3.5) ----------
    K: int = 5                    # portfolio size
    lambda_d: float = 0.3         # diversity weight in archive score (Eq. 20)
    tau_q: float = 0.5            # quality filter: percentile of seen rewards (Eq. 19)
    tau_q_floor: float = 0.05

    # ---------- budget ----------
    budget: int = 0               # 0 = unlimited expensive evals; else B_llm cap (RQ3)

    # ---------- evaluation ----------
    eval_mode: str = "oracle"     # "oracle": U from gold annotation | "llm": real LLM
    utility: str = "f1"           # f1 | recall | ndcg
    embed: str = "sbert"          # dense embedding: sbert (all-MiniLM-L6-v2)
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_mirror: bool = False       # set HF_ENDPOINT=https://hf-mirror.com before model load
    embed_cache: str = "data/embed_cache"

    # ---------- misc ----------
    out_dir: str = "results"
    candidate_cap: int = 200      # per-step candidate cap for oracle-NGA / LLM-greedy
    verbose: bool = True

    llm: LLMConfig = field(default_factory=LLMConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        import yaml
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        llm = raw.pop("llm", {})
        cfg = Config(**{k: v for k, v in raw.items() if k in Config.__dataclass_fields__})
        if llm:
            cfg.llm = LLMConfig(**{k: v for k, v in llm.items() if k in LLMConfig.__dataclass_fields__})
        return cfg

    def apply_args(self, args: argparse.Namespace) -> "Config":
        """CLI overrides (only keys explicitly provided)."""
        mapping = {
            "n_queries": "n_queries", "n_tools": "n_tools", "b": "b", "seed": "seed",
            "source": "source", "library": "library", "budget": "budget",
            "P": "P", "G": "G", "K": "K", "eval_mode": "eval_mode", "embed": "embed",
            "embed_model": "embed_model", "utility": "utility", "cost_mode": "cost_mode",
            "candidate_cap": "candidate_cap", "data_dir": "data_dir", "out_dir": "out_dir",
            "lambda_d": "lambda_d",
        }
        for arg, attr in mapping.items():
            val = getattr(args, attr, None)
            if val is not None:
                setattr(self, attr, val)
        for attr in ("llm_base_url", "llm_api_key", "llm_model"):
            val = getattr(args, attr, None)
            if val:
                setattr(self.llm, attr.removeprefix("llm_"), val)
        if getattr(args, "hf_mirror", False):
            self.hf_mirror = True
        if getattr(args, "quiet", False):
            self.verbose = False
        return self
