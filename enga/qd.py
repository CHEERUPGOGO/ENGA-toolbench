"""Quality-Diversity composition archive (Method 3.5, Eqs. 16-21)."""
from __future__ import annotations

import numpy as np

from .features import ToolIndex


def gamma(index: ToolIndex, comp: list[str]) -> np.ndarray:
    """Gamma(S) = [Gamma_cap(S); Gamma_sch(S)] (Eq. 16), L2-normalized."""
    idx = [index.pos[t] for t in comp if t in index.pos]
    if not idx:
        return np.zeros(index.emb.shape[1] + index.sch_vec.shape[1], dtype=np.float32)
    cap = index.emb[idx].mean(axis=0)
    sch = index.sch_vec[idx].mean(axis=0)
    g = np.concatenate([cap, sch]).astype(np.float32)
    n = float(np.linalg.norm(g))
    return g / n if n > 0 else g


def comp_distance(g1: np.ndarray, g2: np.ndarray) -> float:
    """D_comp = 1 - cos (Eq. 17)."""
    denom = float(np.linalg.norm(g1) * np.linalg.norm(g2))
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(g1, g2) / denom)


class Archive:
    """Keeps at most K compositions maximizing A(S) = r(S) + lambda_d * N(S, A)
    among candidates passing the quality filter r(S) >= tau_q (Eqs. 19-21).

    tau_q adapts as the percentile of rewards seen so far for this query.
    """

    def __init__(self, index: ToolIndex, K: int, lambda_d: float,
                 tau_q: float = 0.5, tau_q_floor: float = 0.05):
        self.index = index
        self.K = K
        self.lambda_d = lambda_d
        self.tau_q = tau_q
        self.tau_q_floor = tau_q_floor
        self.entries: list[dict] = []   # {comp, r, gamma, score}
        self.seen_rewards: list[float] = []

    def _threshold(self) -> float:
        if not self.seen_rewards:
            return self.tau_q_floor
        return max(self.tau_q_floor, float(np.quantile(self.seen_rewards, self.tau_q)))

    def novelty(self, g: np.ndarray) -> float:
        """N(S, A) = min_{S' in A} D_comp(S, S') (Eq. 18); 1 for empty archive."""
        if not self.entries:
            return 1.0
        return min(comp_distance(g, e["gamma"]) for e in self.entries)

    def try_insert(self, comp: list[str], r: float) -> bool:
        self.seen_rewards.append(float(r))
        if r < self._threshold():
            return False
        g = gamma(self.index, comp)
        nov = self.novelty(g)
        score = float(r) + self.lambda_d * nov
        cand = {"comp": list(comp), "r": float(r), "gamma": g, "score": score, "novelty": nov}
        if len(self.entries) < self.K:
            self.entries.append(cand)
            self.entries.sort(key=lambda e: -e["score"])
            return True
        if score > self.entries[-1]["score"]:
            self.entries[-1] = cand
            self.entries.sort(key=lambda e: -e["score"])
            # recompute novelty of survivors after replacement
            self._refresh_novelty()
            return True
        return False

    def _refresh_novelty(self) -> None:
        for i, e in enumerate(self.entries):
            others = [o["gamma"] for j, o in enumerate(self.entries) if j != i]
            e["novelty"] = min((comp_distance(e["gamma"], g) for g in others), default=1.0)

    def portfolio(self) -> list[list[str]]:
        return [e["comp"] for e in self.entries]

    def best_reward(self) -> float:
        return max((e["r"] for e in self.entries), default=0.0)

    def diversity(self) -> float:
        if len(self.entries) < 2:
            return 0.0
        ds = [comp_distance(self.entries[i]["gamma"], self.entries[j]["gamma"])
              for i in range(len(self.entries)) for j in range(i + 1, len(self.entries))]
        return float(np.mean(ds))
