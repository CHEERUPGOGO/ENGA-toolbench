"""Evolution Strategy over NGA policies (Method 3.3, Eqs. 11-15).

OpenAI-ES on alpha = [alpha1, alpha2, alpha3]:
  sample alpha_i = mu + sigma * eps_i          (Eq. 12)
  decode S_i = D(q; alpha_i) via lightweight NGA (Eq. 13)
  reward r_i = U(q, S_i)  -- ONE expensive evaluation per policy (Eq. 14)
  mu <- mu + eta/(P*sigma) * sum_i r_hat_i * eps_i   (Eq. 15, centered ranks)
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .data import Query
from .features import ToolIndex
from .nga import decode
from .qd import Archive, gamma


from scipy.stats import rankdata


def centered_ranks(rewards: np.ndarray) -> np.ndarray:
    """Rank-normalize rewards to [-0.5, 0.5] (standard OpenAI-ES trick with average ties)."""
    n = len(rewards)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    # If all rewards are identical, gradient should be exactly 0
    if np.all(rewards == rewards[0]):
        return np.zeros(n, dtype=np.float64)
    ranks = rankdata(rewards, method="average")
    return (ranks - (n + 1) / 2.0) / n


class ESMonitor:
    """Logs per-generation ES trajectory (for RQ5 alpha-pattern analysis)."""

    def __init__(self):
        self.history: list[dict] = []

    def log(self, gen: int, mu, sigma: float, rewards, best_comp, best_r: float):
        self.history.append({
            "gen": gen, "mu": [float(x) for x in mu], "sigma": float(sigma),
            "mean_r": float(np.mean(rewards)), "best_r": float(best_r),
            "best_comp": list(best_comp),
        })


def enaga_search(index: ToolIndex, q: Query, utility, cfg: Config, rng: np.random.Generator,
                 budget=None, use_es: bool = True, use_syn: bool = True, use_cost: bool = True,
                 use_qd: bool = True):
    """Full ENGA for one query. Returns (mu, archive, records).

    use_es=False  -> random alpha sampling without the ES update (ablation).
    use_qd=False  -> portfolio = top-K by reward instead of the QD archive (ablation).
    """
    dim = 3
    mu = np.array(cfg.init_alpha, dtype=np.float64)
    sigma = float(cfg.sigma0)
    archive = Archive(index, cfg.K, cfg.lambda_d, cfg.tau_q, cfg.tau_q_floor)
    no_qd: list[tuple[float, list[str]]] = []
    monitor = ESMonitor()
    n_expensive = 0
    n_lightweight = 0

    for g in range(cfg.G):
        if cfg.antithetic and cfg.P >= 2:
            half = cfg.P // 2
            eps_half = rng.standard_normal((half, dim))
            eps = np.vstack([eps_half, -eps_half])
        else:
            eps = rng.standard_normal((cfg.P, dim))
        P = len(eps)

        rewards, comps, eps_list = [], [], []
        # elitist anchor: evaluate the current mean policy once per generation
        if cfg.eval_mu:
            if budget is not None and budget.exhausted():
                break
            S_mu = decode(index, q, np.clip(mu, 0.0, None), cfg.b,
                          use_syn=use_syn, use_cost=use_cost)
            r_mu = utility(q, S_mu)
            n_expensive += 1
            rewards.append(r_mu)
            comps.append(S_mu)
            eps_list.append(np.zeros(dim))
            if use_qd:
                archive.try_insert(S_mu, r_mu)
            else:
                no_qd.append((r_mu, S_mu))

        for i in range(P):
            if budget is not None and budget.exhausted():
                break
            alpha = np.clip(mu + sigma * eps[i], 0.0, None)
            ops: list = []
            S = decode(index, q, alpha, cfg.b, use_syn=use_syn, use_cost=use_cost, n_lightweight=ops)
            n_lightweight += int(sum(ops))
            r = utility(q, S)
            n_expensive += 1
            rewards.append(r)
            comps.append(S)
            eps_list.append(eps[i])
            if use_qd:
                archive.try_insert(S, r)
            else:
                no_qd.append((r, S))
        if len(rewards) < 2:
            break
        rewards = np.asarray(rewards, dtype=np.float64)
        eps_arr = np.vstack(eps_list)
        best_i = int(np.argmax(rewards))
        monitor.log(g, mu, sigma, rewards, comps[best_i], float(rewards[best_i]))

        if use_es and len(rewards) >= 3:
            # Eq. 15: mu <- mu + eta/(P*sigma) * sum_i r_hat_i * eps_i
            # (the mu-evaluation enters with eps = 0, i.e. contributes no gradient)
            r_hat = centered_ranks(rewards)
            grad = (r_hat[:, None] * eps_arr).sum(axis=0)
            mu = np.clip(mu + cfg.eta / (P * sigma) * grad, 0.0, None)
        sigma *= cfg.sigma_decay

    if not use_qd:
        no_qd.sort(key=lambda x: -x[0])
        seen: set = set()
        for r, S in no_qd:
            key = tuple(sorted(S))
            if key in seen:
                continue
            seen.add(key)
            archive.entries.append({"comp": list(S), "r": float(r),
                                    "gamma": gamma(index, S),
                                    "score": float(r), "novelty": 0.0})
            if len(archive.entries) >= cfg.K:
                break

    records = {
        "n_expensive": n_expensive, "n_lightweight": n_lightweight,
        "es_history": monitor.history, "final_mu": [float(x) for x in mu],
    }
    return mu, archive, records
