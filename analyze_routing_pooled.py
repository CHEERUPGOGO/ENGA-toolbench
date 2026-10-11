"""Routing-correctness breakdown, pooled across seeds (10-catalog expansion).

For each decode/evo arm available in the given runs, compares test queries the
centroid router assigned to the RIGHT catalog vs a WRONG one. On 10 catalogs the
router is only 72% accurate, but neighboring domain priors are near-interchangeable;
this analysis quantifies that (mis-routed queries score on par or higher).

Usage:
    python analyze_routing_pooled.py                # all three P0 seeds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARMS = ["decode_warm", "decode_cold", "decode_default", "decode_hybrid", "evo_warm", "evo_cold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default=",".join([
        "results/comparison_results_10cat.json",
        "results/comparison_results_10cat_s43.json",
        "results/comparison_results_10cat_s44.json",
    ]))
    ap.add_argument("--out", type=str, default="results/routing_analysis_pooled.md")
    args = ap.parse_args()

    rows = []          # (arm, correct?, metric) samples pooled across seeds
    confusion: dict[tuple[str, str], int] = {}
    n_correct = n_total = 0
    used_seeds = []
    for p in args.results.split(","):
        p = ROOT / p.strip()
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        d = json.load(open(p, encoding="utf-8"))
        used_seeds.append(d["config"].get("seed"))
        for r in d["per_query_results"]:
            ok = bool(r.get("match_success"))
            n_total += 1
            n_correct += int(ok)
            if not ok:
                confusion[(str(r["true_catalog"]), str(r["matched_catalog"]))] = \
                    confusion.get((str(r["true_catalog"]), str(r["matched_catalog"])), 0) + 1
            for arm in ARMS:
                if arm in r:
                    for m in ("f1", "reward"):
                        rows.append((arm, ok, m, r[arm][m]))

    lines = [f"# 路由正确性分解 (pooled, seeds={used_seeds}, {n_total} (seed,query) 对)",
             "",
             f"路由准确率: {n_correct/n_total:.3f} ({n_correct}/{n_total})",
             "",
             "| 方案 | 指标 | 路由正确 | 路由错误 | 差值 |",
             "| :--- | :---: | :---: | :---: | :---: |"]
    for arm in ARMS:
        sub = [(ok, m, v) for a, ok, m, v in rows if a == arm]
        if not sub:
            continue
        for m in ("f1", "reward"):
            c = np.array([v for ok, mm, v in sub if ok and mm == m])
            w = np.array([v for ok, mm, v in sub if not ok and mm == m])
            lines.append(f"| {arm} | {m} | {c.mean():.4f} (n={len(c)}) | "
                         f"{w.mean():.4f} (n={len(w)}) | {w.mean()-c.mean():+.4f} |")
    lines += ["", "## Top-10 路由混淆 (真目录 -> 误路由目录: 次数)", ""]
    for (t, m), c in sorted(confusion.items(), key=lambda kv: -kv[1])[:10]:
        lines.append(f"- {t} -> {m}: {c}")
    lines += ["", "结论: 若误路由查询得分不低于正确路由查询, 说明相邻领域先验可互换, "
                  "路由精度不是 amortized 先验的性能瓶颈。"]

    out = "\n".join(lines)
    print(out.encode("gbk", errors="replace").decode("gbk") if False else out)
    (ROOT / args.out).write_text(out, encoding="utf-8")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
