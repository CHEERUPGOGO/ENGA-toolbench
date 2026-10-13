"""Publication-ready plotting script for Motivation Pre-Experiments (Exp 1, 2, 3).
Strict requirement: 100% REAL EXPERIMENT DATA. No internal plot titles (captions handled by LaTeX).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", default=str(ROOT / "results" / "pre_experiments_v2.json"),
                   help="Path to pre_experiments_v2.json")
    p.add_argument("--out-dir", default=str(ROOT / "figures"),
                   help="Directory to save figures")
    return p.parse_args()


def main():
    args = parse_args()
    json_path = Path(args.json)
    if not json_path.exists():
        sys.exit(f"Error: JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#CBD5E1'
    plt.rcParams['axes.linewidth'] = 0.8

    # =========================================================================
    # Fig 1: Exp 1
    # =========================================================================
    exp1 = data.get("exp1", {})
    summary1 = exp1.get("summary", {})
    if summary1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)
        methods = ['Inner-Loop\n(Greedy)', 'Lightweight\n(Fixed Policy)', 'Outer Policy\n(ENGA)']

        for m_key in ['inner_greedy', 'lightweight_final', 'outer_policy_es']:
            if m_key not in summary1:
                raise KeyError(f"Missing required Exp 1 method: {m_key}")

        f1s = [summary1['inner_greedy']['mean_f1'],
               summary1['lightweight_final']['mean_f1'],
               summary1['outer_policy_es']['mean_f1']]
        us = [summary1['inner_greedy']['mean_u'],
              summary1['lightweight_final']['mean_u'],
              summary1['outer_policy_es']['mean_u']]
        calls = [summary1['inner_greedy']['mean_calls'],
                 summary1['lightweight_final']['mean_calls'],
                 summary1['outer_policy_es']['mean_calls']]

        x = np.arange(len(methods))
        width = 0.35

        b1 = ax1.bar(x - width/2, f1s, width, label='F1 vs Gold', color='#38BDF8', edgecolor='none', alpha=0.9)
        b2 = ax1.bar(x + width/2, us, width, label='Utility U', color='#0284C7', edgecolor='none', alpha=0.9)
        ax1.set_ylabel('Quality Score', fontsize=11, fontweight='bold', color='#1E293B')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, fontsize=9.5, fontweight='bold', color='#1E293B')
        ax1.set_ylim(0, 1.08)
        ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

        for bar in list(b1) + list(b2):
            yval = bar.get_height()
            ax1.annotate(f'{yval:.2f}', xy=(bar.get_x() + bar.get_width()/2, yval),
                         xytext=(0, 2), textcoords="offset points", ha='center', va='bottom',
                         fontsize=8.5, fontweight='bold')

        colors = ['#EF4444', '#94A3B8', '#10B981']
        bars2 = ax2.bar(methods, calls, color=colors, alpha=0.85, width=0.5, edgecolor='none')
        ax2.set_ylabel('Expensive Evaluations (Calls / Query)', fontsize=11, fontweight='bold', color='#1E293B')
        ax2.set_yscale('log')
        ax2.set_ylim(0.5, max(calls) * 5)
        for bar in bars2:
            yval = bar.get_height()
            ax2.annotate(f'{yval:.1f}x', xy=(bar.get_x() + bar.get_width()/2, yval),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                         fontsize=9.5, fontweight='bold')

        plt.tight_layout()
        fig.savefig(out_dir / "exp1_inner_vs_outer.png", bbox_inches='tight', dpi=300)
        plt.close(fig)
        print("  [OK] Saved exp1_inner_vs_outer.png")

    # =========================================================================
    # Fig 2: Exp 2
    # =========================================================================
    exp2 = data.get("exp2", {})
    if exp2:
        jaccards = exp2.get("all_jaccards")
        if not jaccards:
            jaccards = [float(x) for pq in exp2.get("per_query", []) for x in pq.get("jaccard_samples", [])]
        if not jaccards:
            raise ValueError(f"Exp 2 error: No real pairwise Jaccard samples found in {json_path}!")

        alphas = exp2.get("alphas")
        if not alphas:
            raise ValueError(f"Exp 2 error: No alphas found in {json_path}!")

        mean_u = exp2.get("mean_u_per_alpha")
        if not mean_u:
            per_q = exp2.get("per_query", [])
            if per_q and "u" in per_q[0]:
                mean_u = [float(np.mean([pq["u"][k] for pq in per_q])) for k in range(len(alphas))]
            else:
                raise ValueError(f"Exp 2 error: No real utility evaluations per alpha found in {json_path}!")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), dpi=300)

        ax1.hist(jaccards, bins=20, color='#0284C7', edgecolor='none', alpha=0.85, density=True)
        mean_val = float(np.mean(jaccards))
        ax1.axvline(mean_val, color='#E11D48', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.3f}')
        ax1.set_xlabel(r'Pairwise Jaccard Distance $d_J(S_\alpha, S_{\alpha^\prime})$', fontsize=10.5, fontweight='bold', color='#1E293B')
        ax1.set_ylabel('Density', fontsize=10.5, fontweight='bold', color='#1E293B')
        ax1.legend(fontsize=9.5, frameon=True, facecolor='white')

        a1 = [a[0] for a in alphas]
        a2 = [a[1] for a in alphas]

        if max(a1) <= 1.01 and max(a2) <= 1.01:
            xs_b = np.linspace(0.02, 0.96, 100)
            ys_b = 0.98 - xs_b
            ax2.plot(xs_b, ys_b, color='#64748B', linestyle='--', linewidth=1.3,
                     alpha=0.75, label=r'Simplex Boundary ($\alpha_1+\alpha_2=0.98$)')
            ax2.fill_between(xs_b, 0.02, ys_b, color='#F8FAFC', alpha=0.5, zorder=0)
            ax2.set_xlim(0.0, 1.0)
            ax2.set_ylim(0.0, 1.0)
            ax2.legend(fontsize=8.5, loc='upper right', frameon=True, facecolor='white')

        scatter = ax2.scatter(a1, a2, c=mean_u, cmap='viridis', s=65, alpha=0.9,
                              edgecolors='#1E293B', linewidths=0.4, zorder=3)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Mean Utility U', fontsize=10, fontweight='bold', color='#1E293B')
        ax2.set_xlabel(r'Relevance Weight $\alpha_1$', fontsize=10.5, fontweight='bold', color='#1E293B')
        ax2.set_ylabel(r'Synergy Weight $\alpha_2$', fontsize=10.5, fontweight='bold', color='#1E293B')

        plt.tight_layout()
        fig.savefig(out_dir / "exp2_policy_regularity_landscape.png", bbox_inches='tight', dpi=300)
        plt.close(fig)
        print("  [OK] Saved exp2_policy_regularity_landscape.png")

    # =========================================================================
    # Fig 3: Exp 3
    # =========================================================================
    exp3 = data.get("exp3", {})
    table = exp3.get("table", {})
    if table:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=300)

        budgets = [int(b) for b in table.keys()]
        budgets.sort()
        ga_u = [table[str(b)]["ga_u"] for b in budgets]
        es_u = [table[str(b)]["es_u"] for b in budgets]
        ga_f1 = [table[str(b)]["ga_f1"] for b in budgets]
        es_f1 = [table[str(b)]["es_f1"] for b in budgets]

        ax1.plot(budgets, es_u, marker='o', linewidth=2.4, color='#0284C7', label='Policy Search (ENGA-ES)', markersize=7)
        ax1.plot(budgets, ga_u, marker='s', linewidth=2.0, linestyle='--', color='#EF4444', label='Direct Search (Discrete GA)', markersize=6)
        ax1.set_xlabel('Evaluation Budget $B_{LLM}$', fontsize=10.5, fontweight='bold', color='#1E293B')
        ax1.set_ylabel('Utility $U$', fontsize=10.5, fontweight='bold', color='#1E293B')
        ax1.set_xticks(budgets)
        ax1.set_xticklabels([str(b) for b in budgets], fontsize=9.5, fontweight='bold')
        ax1.legend(fontsize=9, frameon=True, facecolor='white')
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2.plot(budgets, es_f1, marker='o', linewidth=2.4, color='#0284C7', label='Policy Search (ENGA-ES)', markersize=7)
        ax2.plot(budgets, ga_f1, marker='s', linewidth=2.0, linestyle='--', color='#EF4444', label='Direct Search (Discrete GA)', markersize=6)
        ax2.set_xlabel('Evaluation Budget $B_{LLM}$', fontsize=10.5, fontweight='bold', color='#1E293B')
        ax2.set_ylabel('F1 Score vs Gold', fontsize=10.5, fontweight='bold', color='#1E293B')
        ax2.set_xticks(budgets)
        ax2.set_xticklabels([str(b) for b in budgets], fontsize=9.5, fontweight='bold')
        ax2.legend(fontsize=9, frameon=True, facecolor='white')
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        fig.savefig(out_dir / "exp3_policy_vs_direct_search.png", bbox_inches='tight', dpi=300)
        plt.close(fig)
        print("  [OK] Saved exp3_policy_vs_direct_search.png")


if __name__ == "__main__":
    main()
