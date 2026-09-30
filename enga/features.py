"""Lightweight composition signals (Method 3.2, Eqs. 5-9).

All signals are deterministic and LLM-free:
  I_rel  (Eq. 5)  cosine similarity between fixed tool/query embeddings
  S_syn  (Eq. 6-8) semantic diversity + schema-level out->in compatibility
  C_run  (Eq. 9)  token footprint (+ optional hashed latency/failure: ToolRet
                  provides no runtime statistics, so lambda_tau/lambda_f terms
                  are deterministic hashes of the tool id -- disclosed in README)

Representation space per index:
  SentenceTransformer dense embeddings (default: all-MiniLM-L6-v2, cached on disk)
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

from .config import Config
from .data import ToolDoc, Query
from .utils import stable_hash

_WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _name_key(name: str) -> str:
    ws = [_stem(w) for w in tokens(name)]
    return " ".join(ws) if ws else ""


def _type_ok(out_t: str, in_t: str) -> bool:
    if out_t == in_t:
        return True
    num = {"number", "integer"}
    txt = {"string", "object"}
    return (out_t in num and in_t in num) or (out_t in txt and in_t in txt)


# ---------------------------------------------------------------------- index
class ToolIndex:
    """Precomputed numeric view of a tool library + queries (the V and q of the paper)."""

    def __init__(self, tools: dict[str, ToolDoc], ids: list[str], cfg: Config,
                 query_texts: list[str] | None = None):
        self.cfg = cfg
        self.ids = list(ids)
        self.tools = tools
        self.pos = {t: i for i, t in enumerate(self.ids)}
        self.n = len(self.ids)
        self.query_texts = query_texts or []

        docs = [tools[t] for t in self.ids]
        self.doc_texts = [d.doc_text() for d in docs]

        # Initialize dense embeddings via SentenceTransformer (cached on disk)
        self._init_sbert(self.doc_texts, self.query_texts)

        # schema info for Compat (Eq. 8)
        self.req_names = [[_name_key(p.name) for p in d.required_params] for d in docs]
        self.req_types = [{_name_key(p.name): p.type for p in d.required_params} for d in docs]
        self.out_fields = [[(_name_key(p.name), p.type) for p in d.out_fields] for d in docs]

        # schema signature vectors for Gamma(S) (Eq. 16)
        self.sch_vec = self._schema_vectors(docs)

        # runtime cost C_run (Eq. 9)
        self.cost = self._cost_vector(docs)

        # Domain/API family & category for positive co-occurrence synergy
        self.cats = [d.category for d in docs]
        self.fams = []
        for d in docs:
            # Extract real API service name from ToolBench name (e.g. api_video from api_video_GET_players)
            parts = re.split(r'_(?:GET|POST|PUT|DELETE|PATCH)_', d.name, flags=re.IGNORECASE)
            self.fams.append(parts[0] if len(parts) > 1 else (d.name.split('_')[0] if '_' in d.name else d.name))

        self._rel_cache: dict[str, np.ndarray] = {}
        self._reln_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ embedding init
    def _get_st_model(self):
        if self._st_model is None:
            import os
            if self.cfg.hf_mirror:
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from sentence_transformers import SentenceTransformer
            try:
                self._st_model = SentenceTransformer(self.cfg.embed_model, local_files_only=True)
            except Exception:
                self._st_model = SentenceTransformer(self.cfg.embed_model)
        return self._st_model

    def _init_sbert(self, doc_texts: list[str], query_texts: list[str]) -> None:
        self._st_model = None
        key = f"{self.cfg.embed_model.replace('/', '__')}_{stable_hash('|'.join(self.ids)) & 0xffffffffffffffff}"
        cpath = Path(self.cfg.embed_cache) / f"{key}.npy"
        if cpath.exists():
            self.emb = np.load(cpath)
        else:
            model = self._get_st_model()
            self.emb = np.asarray(
                model.encode(doc_texts, normalize_embeddings=True,
                             batch_size=64, show_progress_bar=False),
                dtype=np.float32)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.save(cpath, self.emb)

    # ------------------------------------------------------------------- I_rel
    def rel(self, q: Query) -> np.ndarray:
        """I_rel(v, q) for all tools (Eq. 5). Cached per query id."""
        if q.qid in self._rel_cache:
            return self._rel_cache[q.qid]
        model = self._get_st_model()
        qe = np.asarray(model.encode([q.text], normalize_embeddings=True,
                                     show_progress_bar=False)[0], dtype=np.float32)
        rel = self.emb @ qe
        self._rel_cache[q.qid] = rel
        return rel

    def rel_norm(self, q: Query) -> np.ndarray:
        """I_rel rescaled to [0,1] via per-query min-max.

        Order-preserving (identical ranking to raw cosine), so the three
        channels of Eq. 7 live on comparable scales: TF-IDF cosines occupy
        ~[0, 0.35] while S_sem occupies [0, 1]; without rescaling alpha would
        weigh the ranges, not the trade-off.
        """
        r = self.rel(q)
        if q.qid in self._reln_cache:
            return self._reln_cache[q.qid]
        lo, hi = float(r.min()), float(r.max())
        rn = ((r - lo) / max(1e-6, hi - lo)).astype(np.float32)
        self._reln_cache[q.qid] = rn
        return rn

    # ------------------------------------------------------------------- S_syn
    def sem_syn(self, sel_idx: list[int]) -> np.ndarray:
        """Domain & API-Family Co-occurrence Synergy S_syn(v, X) (Method 3.2, Eq. 7).
        
        Rewards candidate tools that share the same API family (0.7) or functional
        category (0.3) with already selected tools, capturing genuine multi-tool
        cooperation, while penalizing near-identical duplicates (cos > 0.95).
        """
        if not sel_idx:
            return np.zeros(self.n, dtype=np.float32)
        
        sel_fams = {self.fams[s] for s in sel_idx if self.fams[s]}
        sel_cats = {self.cats[s] for s in sel_idx if self.cats[s]}
        
        syn = np.zeros(self.n, dtype=np.float32)
        for v in range(self.n):
            if self.fams[v] in sel_fams:
                syn[v] += 0.7
            elif self.cats[v] in sel_cats:
                syn[v] += 0.3
                
        # Duplicate penalty: only penalize near-identical tools (cos > 0.95)
        cos_matrix = self.emb @ self.emb[sel_idx].T
        max_cos = np.max(cos_matrix, axis=1)
        syn -= np.where(max_cos > 0.95, 0.3, 0.0)
        
        return np.clip(syn, 0.0, 1.0).astype(np.float32)

    def sch_syn(self, sel_idx: list[int]) -> np.ndarray:
        """S_sch(v, X) = max_{x in X} Compat(out(x), in(v))  (Eq. 8)."""
        if not sel_idx:
            return np.zeros(self.n, dtype=np.float32)
        out = np.zeros(self.n, dtype=np.float32)
        for x in sel_idx:
            fields = self.out_fields[x]
            if not fields:
                continue
            for v in range(self.n):
                req_names = self.req_names[v]
                if not req_names:
                    continue
                hits = 0
                for fname, ftype in fields:
                    vt = self.req_types[v].get(fname)
                    if vt is not None and _type_ok(ftype, vt):
                        hits += 1
                if hits:
                    score = hits / len(req_names)
                    if score > out[v]:
                        out[v] = score
        return out

    # ------------------------------------------------------------------- Gamma
    def _schema_vectors(self, docs: list[ToolDoc]) -> np.ndarray:
        """Feature-hashed (name,type) bag for Gamma(S) (Eq. 16), dim 64."""
        D = 64
        V = np.zeros((len(docs), D), dtype=np.float32)
        for i, d in enumerate(docs):
            for p in d.required_params + d.optional_params:
                V[i, stable_hash(f"{_name_key(p.name)}:{p.type}") % D] += 1.0
            if d.category:
                V[i, stable_hash(f"cat:{d.category}") % D] += 0.5
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return V / norms

    # ------------------------------------------------------------------- C_run
    def _cost_vector(self, docs: list[ToolDoc]) -> np.ndarray:
        """C_run(v) (Eq. 9), min-max normalized to [0, 1]."""
        cfg = self.cfg
        L = np.array([len(t) for t in (tokens(x) for x in self.doc_texts)], dtype=np.float32)
        L = L / max(1e-6, float(L.max()))
        if cfg.cost_mode == "synthetic":
            T = np.array([(stable_hash(t.id + "lat") % 1000) / 1000.0 for t in docs], dtype=np.float32)
            R = np.array([(stable_hash(t.id + "fail") % 1000) / 1000.0 for t in docs], dtype=np.float32)
        else:
            T = np.zeros_like(L)
            R = np.zeros_like(L)
        C = cfg.lambda_t * L + cfg.lambda_tau * T + cfg.lambda_f * R
        lo, hi = float(C.min()), float(C.max())
        return (C - lo) / max(1e-6, hi - lo)

    # --------------------------------------------------------------------- BM25
    def bm25_scores(self, qtext: str) -> np.ndarray:
        """BM25 baseline (k1 = 1.2, b = 0.75) over doc texts."""
        if not hasattr(self, "_bm25"):
            self._build_bm25()
        tf, df, N, avgdl, dl, vocab = self._bm25
        scores = np.zeros(self.n, dtype=np.float32)
        for w, cnt in Counter(tokens(qtext)).items():
            j = vocab.get(w)
            if j is None:
                continue
            idf = np.log(1.0 + (N - df[w] + 0.5) / (df[w] + 0.5))
            tf_col = tf[:, j]
            scores += cnt * idf * tf_col * 2.2 / (tf_col + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))
        return scores

    def _build_bm25(self) -> None:
        toks = [tokens(t) for t in self.doc_texts]
        vocab: dict[str, int] = {}
        for ts in toks:
            for w in ts:
                vocab.setdefault(w, len(vocab))
        tf = np.zeros((self.n, len(vocab)), dtype=np.float32)
        df: Counter = Counter()
        for i, ts in enumerate(toks):
            for w, c in Counter(ts).items():
                tf[i, vocab[w]] = c
                df[w] += 1
        dl = np.array([len(ts) for ts in toks], dtype=np.float32)
        self._bm25 = (tf, dict(df), float(self.n), float(dl.mean()) or 1.0, dl, vocab)
