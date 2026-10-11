"""Aggregate P0 multi-seed results (10-catalog expansion, 方案A).

Consumes one comparison_results JSON per training seed (seed 42 = full arm set;
seeds 43/44 = reduced evo arms on the SAME pool via --pool-seed 42), and emits:

  1. per-arm per-seed means + across-seed mean/std (training-randomness only:
     pool, queries, and retrieval arms are shared, so retrieval std is 0 by
     construction and evo/learned-α std isolates ES + init randomness),
  2. pooled paired tests over all (seed, query) pairs,
  3. routing accuracy and judge-gold correlation per seed.

Usage:
    python analyze_p0_multiseed.py            # the three P0 seeds
    python analyze_p0_multiseed.py --results results/comparison_results_10cat.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
METRICS = ("f1", "recall", "ndcg", "reward")

ARM_ORDER = [
    ("decode_bm25", "BM25 top-5 (lexical baseline)", "0 LLM"),
    ("decode_dense", "Dense cosine top-5", "0 LLM"),
    ("decode_hybrid", "Hybrid BM25+Dense top-5", "0 LLM"),
    ("decode_cold", "Cold a0 (random init)", "0 LLM"),
    ("decode_default", "Global Default [0.80,0.16,0.04]", "0 LLM"),
    ("decode_tail", "Tail a30 (non-averaged prior)", "0 LLM"),
    ("decode_warm", "Amortized a_mean (ours, Polyak)", "0 LLM"),
    ("evo_cold", "Cold a0 -> ES re-evolution", "45 LLM"),
    ("evo_default", "Global Default -> ES re-evolution", "45 LLM"),
    ("evo_warm", "Amortized a_mean -> ES re-evolution", "45 LLM"),
]

PAIRED = [
    ("evo_warm", "evo_cold", "再演化: amortized vs cold"),
    ("evo_warm", "evo_default", "再演化: amortized vs default"),
    ("evo_warm", "decode_warm", "再演化 vs 直接解码 (amortized)"),
    ("decode_warm", "decode_default", "解码: amortized vs global default"),
    ("decode_warm", "decode_tail", "解码: amortized (mean) vs tail"),
    ("decode_warm", "decode_bm25", "解码: amortized vs BM25"),
    ("decode_warm", "decode_hybrid", "解码: amortized vs hybrid"),
    ("evo_warm", "decode_hybrid", "再演化 (amortized) vs hybrid 解码"),
]


def load_seed(path: Path) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    per_q = {r["qid"]: r for r in d["per_query_results"]}
    return {"tag": d["config"].get("seed"), "path": str(path), "per_q": per_q,
            "routing": d.get("catalog_routing_accuracy"), "corr": d.get("judge_gold_corr"),
            "aggregate": d.get("aggregate", {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default=",".join([
        "results/comparison_results_10cat.json",
        "results/comparison_results_10cat_s43.json",
        "results/comparison_results_10cat_s44.json",
    ]))
    ap.add_argument("--out", type=str, default="results/P0_multiseed_summary.md")
    args = ap.parse_args()

    seeds = []
    for p in args.results.split(","):
        p = (ROOT / p.strip())
        if p.exists():
            seeds.append(load_seed(p))
        else:
            print(f"[skip] missing {p}")
    if not seeds:
        sys.exit("no result files found")
    n = len(seeds)
    qids = sorted(set.intersection(*[set(s["per_q"]) for s in seeds]))
    print(f"{n} seeds, {len(qids)} shared test queries\n")

    lines = [f"# P0 多种子汇总 (10 Catalog, 100 test, pool_seed=42 固定, n={n} seeds)",
             "", "种子只改变训练随机性 (a0 采样 + ES 扰动); 池与检索基线跨种子共享, "
             "其 std=0 属预期。evo/学习到的 alpha 的 std 即训练随机性。", ""]

    # ---- per-seed header
    lines += ["## 各种子概况", "",
              "| seed | routing acc | judge-gold corr |",
              "| :--- | :---: | :---: |"]
    for s in seeds:
        lines.append(f"| {s['tag']} | {s['routing']:.2f} | {s['corr']:.3f} |")
    lines.append("")

    # ---- main table
    lines += ["## 各方案指标 (mean ± std over seeds; per-seed mean in brackets)", "",
              "| 方案 | F1@5 | Recall@5 | NDCG@5 | Judge Reward | 开销 |",
              "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for arm, label, cost in ARM_ORDER:
        present = [s for s in seeds if any(arm in s["per_q"][q] for q in qids)]
        if not present:
            continue
        cells = []
        for m in METRICS:
            seed_means = np.array([np.mean([s["per_q"][q][arm][m] for q in qids])
                                   for s in present])
            if len(seed_means) == 1:
                cells.append(f"{seed_means[0]:.4f}")
            else:
                cells.append(f"{seed_means.mean():.4f}±{seed_means.std(ddof=1):.4f} "
                             f"({', '.join(f'{x:.4f}' for x in seed_means)})")
        n_seeds_arm = len(present)
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {cost} |")
    lines.append("")

    # ---- pooled paired tests
    lines += ["## 配对检验 (pooled over seeds x queries, F1@5)", "",
              "| 对比 | seeds | n | mean diff | W/T/L | t |",
              "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for a, b, label in PAIRED:
        diffs = []
        for s in seeds:
            for q in qids:
                if a in s["per_q"][q] and b in s["per_q"][q]:
                    diffs.append(s["per_q"][q][a]["f1"] - s["per_q"][q][b]["f1"])
        if not diffs:
            continue
        d = np.array(diffs)
        w = int((d > 1e-9).sum()); t_ = int((np.abs(d) <= 1e-9).sum()); l = int((d < -1e-9).sum())
        tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 1e-12 else float("inf")
        lines.append(f"| {label} | {n if len(diffs)==n*len(qids) else 'part'} | {len(d)} | "
                     f"{d.mean():+.4f} | {w}/{t_}/{l} | {tstat:+.2f} |")
    lines.append("")

    # ---- per-seed paired diffs for the headline comparisons
    lines += ["## 分种子 headline 差值 (F1@5)", "",
              "| 对比 | " + " | ".join(str(s["tag"]) for s in seeds) + " |",
              "| :--- | " + " | ".join(":---:" for _ in seeds) + " |"]
    for a, b, label in PAIRED:
        row = []
        for s in seeds:
            dd = [s["per_q"][q][a]["f1"] - s["per_q"][q][b]["f1"]
                  for q in qids if a in s["per_q"][q] and b in s["per_q"][q]]
            row.append(f"{np.mean(dd):+.4f}" if dd else "—")
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    lines.append("")

    out = "\n".join(lines)
    print(out)
    (ROOT / args.out).write_text(out, encoding="utf-8")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
