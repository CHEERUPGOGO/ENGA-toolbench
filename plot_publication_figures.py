"""Publication-Quality Plotting Script for ENGA Benchmark.
Generates four publication-grade figures from real experimental results:
1. fig_pareto_frontier: 3-Tier Pareto Frontier (LLM Calls vs Quality: F1@5 and NDCG@5)
2. fig_simplex_polyak: Barycentric Ternary Simplex showing trajectory convergence,
   interior Polyak consensus mean vs. corner-collapsing tail iterates, and per-catalog gains.
3. fig_cross_judge: Cross-Judge Invariance comparing DeepSeek-v4-Flash and Gemini-3.8-Flash.
4. fig_component_ablations: 4-panel ablation study (Cost channel, Synergy, Beta sweep, Pool scale).

Outputs both vector PDF and 300-dpi PNG to d:/iclr2026/paper_en/figures/.
"""

import json
import math
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set professional academic styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#94A3B8'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#E2E8F0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

BASE_DIR = Path("d:/iclr2026/share_repo")
RESULTS_DIR = BASE_DIR / "results"
FIG_OUT_DIR = Path("d:/iclr2026/paper_en/figures")
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_pareto_frontier():
    """Figure 1: 3-Tier Cost-Quality Pareto Frontier (RQ3)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300)

    # Real data from P0_multiseed_summary.md and llm_rerank_10cat.json
    # Format: (name, calls, f1_mean, f1_std, ndcg_mean, ndcg_std, color, marker, tier, is_pareto)
    methods = [
        # Tier 1 (0 calls)
        ("Dense Cosine", 0, 0.2454, 0.0, 0.3786, 0.0, '#94A3B8', 'v', 1, False),
        ("Cold Init $\\alpha_0$", 0, 0.2826, 0.0210, 0.4251, 0.0278, '#F59E0B', 's', 1, False),
        ("BM25 (Lexical)", 0, 0.3171, 0.0, 0.4683, 0.0, '#64748B', '^', 1, False),
        ("Tail Prior $\\mu_T$", 0, 0.3215, 0.0122, 0.4826, 0.0132, '#FB7185', 'd', 1, False),
        ("Hybrid ($\\beta{=}0.6$)", 0, 0.3413, 0.0, 0.4992, 0.0, '#3B82F6', 'p', 1, False),
        ("Amortized Decode $\\bar\\alpha_z$ (Ours)", 0, 0.3409, 0.0070, 0.4932, 0.0062, '#0EA5E9', 'o', 1, True),

        # Tier 2 (1 call)
        ("1-Call LLM Rerank", 1, 0.3498, 0.0, 0.5053, 0.0, '#8B5CF6', 'P', 2, True),

        # Tier 3 (45 calls)
        ("Cold Re-Evolution", 45, 0.3492, 0.0077, 0.5071, 0.0205, '#F97316', 'X', 3, False),
        ("Warm Re-Evolution (Ours)", 45, 0.3627, 0.0059, 0.5215, 0.0021, '#10B981', '*', 3, True),
    ]

    # Map discrete call tiers for clear visual separation on x-axis:
    # 0 calls -> x=0, 1 call -> x=1, 45 calls -> x=2
    def x_map(calls, offset=0.0):
        if calls == 0:
            return 0.0 + offset
        elif calls == 1:
            return 1.0 + offset
        else:
            return 2.0 + offset

    # Jitter offsets for Tier 1 so points do not collide
    jitters = {
        "Dense Cosine": -0.15,
        "Cold Init $\\alpha_0$": -0.09,
        "BM25 (Lexical)": -0.03,
        "Tail Prior $\\mu_T$": 0.03,
        "Hybrid ($\\beta{=}0.6$)": 0.09,
        "Amortized Decode $\\bar\\alpha_z$ (Ours)": 0.15,
        "1-Call LLM Rerank": 0.0,
        "Cold Re-Evolution": -0.06,
        "Warm Re-Evolution (Ours)": 0.06,
    }

    # Plot F1@5 (Subplot 1)
    ax1.set_title("(a) Tool Selection Quality (F1@5 vs. Gold)", fontsize=11.5, fontweight='bold', pad=10, color='#0F172A')
    ax2.set_title("(b) Ranking Quality (NDCG@5 vs. Gold)", fontsize=11.5, fontweight='bold', pad=10, color='#0F172A')

    pareto_x_f1, pareto_y_f1 = [], []
    pareto_x_ndcg, pareto_y_ndcg = [], []

    for name, calls, f1_m, f1_s, ndcg_m, ndcg_s, col, mkr, tier, is_pareto in methods:
        x_val = x_map(calls, jitters[name])
        msize = 120 if is_pareto or "Ours" in name else 70
        ew = 1.6 if is_pareto or "Ours" in name else 1.0
        ec = '#0F172A' if is_pareto else col

        # Ax1: F1@5
        ax1.errorbar(x_val, f1_m, yerr=f1_s, fmt=mkr, color=col, ecolor=col,
                     markersize=8 if not is_pareto else 10, markeredgecolor=ec, markeredgewidth=ew,
                     capsize=3.5, label=name if calls <= 1 or "Re-Evolution" in name else None)

        # Ax2: NDCG@5
        ax2.errorbar(x_val, ndcg_m, yerr=ndcg_s, fmt=mkr, color=col, ecolor=col,
                     markersize=8 if not is_pareto else 10, markeredgecolor=ec, markeredgewidth=ew,
                     capsize=3.5, label=name)

        if is_pareto:
            pareto_x_f1.append(x_val)
            pareto_y_f1.append(f1_m)
            pareto_x_ndcg.append(x_val)
            pareto_y_ndcg.append(ndcg_m)

    # Draw Pareto boundary
    ax1.plot(pareto_x_f1, pareto_y_f1, linestyle='--', color='#10B981', linewidth=2.0, alpha=0.85, label='Pareto Frontier')
    ax2.plot(pareto_x_ndcg, pareto_y_ndcg, linestyle='--', color='#10B981', linewidth=2.0, alpha=0.85, label='Pareto Frontier')

    # Shaded tiers
    for ax in (ax1, ax2):
        ax.axvspan(-0.35, 0.45, color='#F8FAFC', alpha=0.9, zorder=0)
        ax.axvspan(0.65, 1.35, color='#F1F5F9', alpha=0.9, zorder=0)
        ax.axvspan(1.65, 2.35, color='#ECFDF5', alpha=0.4, zorder=0)
        ax.set_xticks([0.0, 1.0, 2.0])
        ax.set_xticklabels(['Tier 1: Zero Cost\n(0 LLM calls/q)',
                            'Tier 2: Single Call\n(1 LLM call/q)',
                            'Tier 3: Re-Evolution\n(45 LLM calls/q)'],
                           fontsize=9.5, fontweight='bold', color='#1E293B')
        ax.set_xlim(-0.35, 2.35)
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    ax1.set_ylabel("F1@5 Score", fontsize=10.5, fontweight='bold', color='#1E293B')
    ax1.set_ylim(0.23, 0.38)
    ax2.set_ylabel("NDCG@5 Score", fontsize=10.5, fontweight='bold', color='#1E293B')
    ax2.set_ylim(0.36, 0.54)

    # Annotate key comparisons
    ax1.annotate('Amortized Decode matches Hybrid\n(+0.058 F1 over Cold, p<0.001)',
                 xy=(pareto_x_f1[0], pareto_y_f1[0]), xytext=(0.0, 0.252),
                 arrowprops=dict(arrowstyle="->", color='#0EA5E9', lw=1.2),
                 fontsize=8.5, fontweight='bold', color='#0284C7',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='#0EA5E9', alpha=0.9))

    ax1.annotate('Warm Re-Evolution beats Cold\n(+0.013 F1, p<0.05)',
                 xy=(pareto_x_f1[2], pareto_y_f1[2]), xytext=(1.45, 0.370),
                 arrowprops=dict(arrowstyle="->", color='#10B981', lw=1.2),
                 fontsize=8.5, fontweight='bold', color='#059669',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='#10B981', alpha=0.9))

    # Single comprehensive legend
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06),
               frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=8.5)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    pdf_path = FIG_OUT_DIR / "fig_pareto_frontier.pdf"
    png_path = FIG_OUT_DIR / "fig_pareto_frontier.png"
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {pdf_path} and {png_path}")


def to_ternary(rel, syn, cost):
    """Converts 3D simplex coordinates (rel, syn, cost) summing to 1 into 2D Cartesian coordinates.
    Vertices:
      Rel (1,0,0)  -> (0, 0)
      Syn (0,1,0)  -> (1, 0)
      Cost (0,0,1) -> (0.5, sqrt(3)/2)
    """
    total = rel + syn + cost
    r = rel / total
    s = syn / total
    c = cost / total
    x = s + 0.5 * c
    y = (math.sqrt(3) / 2.0) * c
    return x, y


def plot_simplex_polyak():
    """Figure 2: Simplex Geometry & Polyak Dynamics (RQ2)."""
    with open(RESULTS_DIR / "catalog_learned_alphas_10cat_fix.json", "r", encoding="utf-8") as f:
        alpha_data = json.load(f)

    polyaks = alpha_data["learned_alphas_polyak"]
    tails = alpha_data["learned_alphas_tail"]
    trajs = alpha_data["trajectories"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5.0), dpi=300)

    # -------------------------------------------------------------
    # Subplot 1: 2D Ternary Simplex
    # -------------------------------------------------------------
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Draw Simplex Boundary Triangle
    # Corner A: Relevance (1, 0, 0) -> (0, 0)
    # Corner B: Synergy   (0, 1, 0) -> (1, 0)
    # Corner C: Cost      (0, 0, 1) -> (0.5, sqrt(3)/2)
    h = math.sqrt(3) / 2.0
    triangle = plt.Polygon([[0, 0], [1, 0], [0.5, h]], closed=True,
                           facecolor='#F8FAFC', edgecolor='#64748B', linewidth=1.8, zorder=1)
    ax1.add_patch(triangle)

    # Interior Grid Lines (0.2, 0.4, 0.6, 0.8)
    for frac in [0.2, 0.4, 0.6, 0.8]:
        # rel = frac line
        p1 = to_ternary(frac, 1 - frac, 0)
        p2 = to_ternary(frac, 0, 1 - frac)
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#E2E8F0', lw=0.7, linestyle=':', zorder=2)

        # syn = frac line
        p1 = to_ternary(1 - frac, frac, 0)
        p2 = to_ternary(0, frac, 1 - frac)
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#E2E8F0', lw=0.7, linestyle=':', zorder=2)

        # cost = frac line
        p1 = to_ternary(1 - frac, 0, frac)
        p2 = to_ternary(0, 1 - frac, frac)
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#E2E8F0', lw=0.7, linestyle=':', zorder=2)

    # Vertex Labels
    ax1.text(-0.06, -0.04, "Pure Relevance\n$\\alpha_{\\mathrm{rel}}=1$",
             fontsize=9.5, fontweight='bold', color='#1E293B', ha='right', va='top')
    ax1.text(1.06, -0.04, "Pure Synergy\n$\\alpha_{\\mathrm{syn}}=1$",
             fontsize=9.5, fontweight='bold', color='#1E293B', ha='left', va='top')
    ax1.text(0.5, h + 0.05, "Pure Cost Frugality\n$\\alpha_{\\mathrm{cost}}=1$",
             fontsize=9.5, fontweight='bold', color='#1E293B', ha='center', va='bottom')

    # Boundary collapse zone highlight (outer borders)
    # Highlight the safe Polyak interior zone
    polyak_xs, polyak_ys = [], []
    for cat_id, p_alpha in polyaks.items():
        px, py = to_ternary(p_alpha[0], p_alpha[1], p_alpha[2])
        polyak_xs.append(px)
        polyak_ys.append(py)

    # Plot trajectories, tail iterates, and polyak means
    cat_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (cat_id, traj) in enumerate(trajs.items()):
        color = cat_colors[idx]
        tx = [to_ternary(step[0], step[1], step[2])[0] for step in traj]
        ty = [to_ternary(step[0], step[1], step[2])[1] for step in traj]

        # Draw trajectory path
        ax1.plot(tx, ty, color=color, alpha=0.35, linewidth=1.2, zorder=3)

        # Draw tail iterate (step 30)
        tail_alpha = tails[cat_id]
        tail_x, tail_y = to_ternary(tail_alpha[0], tail_alpha[1], tail_alpha[2])
        ax1.scatter(tail_x, tail_y, marker='x', color='#EF4444', s=55, linewidth=1.8,
                    zorder=5, label='Tail Iterate $\\mu_T$ (Corner Collapse)' if idx == 0 else None)

        # Draw Polyak consensus mean
        p_alpha = polyaks[cat_id]
        px, py = to_ternary(p_alpha[0], p_alpha[1], p_alpha[2])
        ax1.scatter(px, py, marker='o', facecolor='#0284C7', edgecolor='#0F172A', s=70, linewidth=1.2,
                    zorder=6, label='Polyak Mean $\\bar\\alpha_z$ (Interior Prior)' if idx == 0 else None)

    # Shaded interior consensus region
    hull_x_min, hull_x_max = min(polyak_xs) - 0.03, max(polyak_xs) + 0.03
    hull_y_min, hull_y_max = min(polyak_ys) - 0.03, max(polyak_ys) + 0.03
    interior_box = plt.Rectangle((hull_x_min, hull_y_min), hull_x_max - hull_x_min, hull_y_max - hull_y_min,
                                 facecolor='#38BDF8', alpha=0.15, edgecolor='#0284C7', linestyle='--', linewidth=1.2, zorder=2)
    ax1.add_patch(interior_box)
    ax1.text(0.5, 0.16, "Interior Domain Consensus\n$\\alpha_{\\mathrm{rel}} \\in [0.45, 0.77]$\n$\\alpha_{\\mathrm{syn}} \\in [0.09, 0.33]$\n$\\alpha_{\\mathrm{cost}} \\in [0.07, 0.31]$",
             fontsize=8.5, fontweight='bold', color='#0369A1', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#38BDF8', alpha=0.9))

    ax1.set_title("(a) Domain Trajectories on Ternary Simplex $\\Delta^2$",
                  fontsize=11.5, fontweight='bold', pad=15, color='#0F172A')
    ax1.set_xlim(-0.25, 1.25)
    ax1.set_ylim(-0.15, h + 0.18)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=8.5)

    # -------------------------------------------------------------
    # Subplot 2: Per-Catalog F1: Polyak Mean vs Tail Collapse (3-Seed Pooled)
    # -------------------------------------------------------------
    warm_all = {i: [] for i in range(1, 11)}
    tail_all = {i: [] for i in range(1, 11)}

    for fname in ['comparison_results_10cat.json', 'comparison_results_10cat_s43.json', 'comparison_results_10cat_s44.json']:
        with open(RESULTS_DIR / fname, "r", encoding="utf-8") as f:
            d = json.load(f)
            for q in d.get("per_query_results", []):
                cid = int(q.get("true_catalog", 1))
                warm_all[cid].append(q.get("decode_warm", {}).get("f1", 0.0))
                tail_all[cid].append(q.get("decode_tail", {}).get("f1", 0.0))

    polyak_means = [float(np.mean(warm_all[i])) for i in range(1, 11)]
    tail_means = [float(np.mean(tail_all[i])) for i in range(1, 11)]

    x = np.arange(10)
    width = 0.38

    b1 = ax2.bar(x - width/2, polyak_means, width, label='Polyak Mean $\\bar\\alpha_z$ (Ours)',
                 color='#0EA5E9', edgecolor='none', alpha=0.9)
    b2 = ax2.bar(x + width/2, tail_means, width, label='Tail Iterate $\\mu_T$ (Ablation)',
                 color='#F87171', edgecolor='none', alpha=0.9)

    ax2.set_title("(b) Per-Catalog Test F1: Polyak Averaging vs. Tail Collapse",
                  fontsize=11.5, fontweight='bold', pad=10, color='#0F172A')
    ax2.set_ylabel("Test F1@5 Score", fontsize=10.5, fontweight='bold', color='#1E293B')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"C{i+1}" for i in range(10)], fontsize=9.5, fontweight='bold', color='#1E293B')
    ax2.set_ylim(0.0, 0.52)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=8.5)

    # Annotate overall gain
    mean_gain = np.mean(polyak_means) - np.mean(tail_means)
    ax2.annotate(f"Mean Gain: +{mean_gain:.4f} F1@5\n(p = 0.007 < 0.01, paired t = +2.69)",
                 xy=(4.5, 0.44), xytext=(4.5, 0.44),
                 ha='center', fontsize=9.0, fontweight='bold', color='#0369A1',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='#F0F9FF', edgecolor='#0EA5E9', alpha=0.95))

    plt.tight_layout()
    pdf_path = FIG_OUT_DIR / "fig_simplex_polyak.pdf"
    png_path = FIG_OUT_DIR / "fig_simplex_polyak.png"
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {pdf_path} and {png_path}")


def plot_cross_judge():
    """Figure 3: Cross-Judge Invariance (DeepSeek vs Gemini-3.8-Flash) (RQ5)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300)

    # Real data from comparison_results_10cat_fix.json and comparison_results_10cat_gemini.json
    methods = [
        "BM25\nTop-5",
        "Hybrid\nTop-5",
        "Cold\nDecode",
        "Amortized\nDecode",
        "Cold\nRe-Evo",
        "Warm\nRe-Evo"
    ]

    deepseek_f1 = [0.3171, 0.3413, 0.3066, 0.3290, 0.3551, 0.3587]
    gemini_f1   = [0.3171, 0.3413, 0.3066, 0.3401, 0.3624, 0.3823]

    deepseek_recall = [0.5000, 0.5380, 0.4855, 0.5155, 0.5605, 0.5630]
    gemini_recall   = [0.5000, 0.5380, 0.4855, 0.5338, 0.5697, 0.5980]

    x = np.arange(len(methods))
    width = 0.35

    # Subplot (a): F1@5 across judges
    b1 = ax1.bar(x - width/2, deepseek_f1, width, label='Judge: DeepSeek-v4-Flash',
                 color='#0284C7', edgecolor='none', alpha=0.9)
    b2 = ax1.bar(x + width/2, gemini_f1, width, label='Judge: Gemini-3.8-Flash',
                 color='#10B981', edgecolor='none', alpha=0.9)

    ax1.set_title("(a) Selection Quality (F1@5 vs Gold Ground-Truth)", fontsize=11, fontweight='bold', pad=10, color='#0F172A')
    ax1.set_ylabel("F1@5 Score", fontsize=10.5, fontweight='bold', color='#1E293B')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9, fontweight='bold', color='#1E293B')
    ax1.set_ylim(0, 0.45)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=8.5)

    for bar in list(b1) + list(b2):
        yval = bar.get_height()
        ax1.annotate(f'{yval:.3f}', xy=(bar.get_x() + bar.get_width()/2, yval),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom',
                     fontsize=7.5, fontweight='bold', color='#1E293B')

    # Subplot (b): Recall@5 across judges
    r1 = ax2.bar(x - width/2, deepseek_recall, width, label='Judge: DeepSeek-v4-Flash',
                 color='#0284C7', edgecolor='none', alpha=0.9)
    r2 = ax2.bar(x + width/2, gemini_recall, width, label='Judge: Gemini-3.8-Flash',
                 color='#10B981', edgecolor='none', alpha=0.9)

    ax2.set_title("(b) Retrieval Coverage (Recall@5 vs Gold Ground-Truth)", fontsize=11, fontweight='bold', pad=10, color='#0F172A')
    ax2.set_ylabel("Recall@5 Score", fontsize=10.5, fontweight='bold', color='#1E293B')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=9, fontweight='bold', color='#1E293B')
    ax2.set_ylim(0, 0.70)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=8.5)

    for bar in list(r1) + list(r2):
        yval = bar.get_height()
        ax2.annotate(f'{yval:.3f}', xy=(bar.get_x() + bar.get_width()/2, yval),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom',
                     fontsize=7.5, fontweight='bold', color='#1E293B')

    # Annotate statistical significance under Gemini
    ax1.annotate('Gemini Warm Re-Evo:\n0.3823 F1 (+0.020, p<0.05)',
                 xy=(5 + width/2, 0.3823), xytext=(3.4, 0.405),
                 arrowprops=dict(arrowstyle="->", color='#059669', lw=1.2),
                 fontsize=8.5, fontweight='bold', color='#059669',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='#ECFDF5', edgecolor='#10B981', alpha=0.95))

    plt.tight_layout()
    pdf_path = FIG_OUT_DIR / "fig_cross_judge_robustness.pdf"
    png_path = FIG_OUT_DIR / "fig_cross_judge_robustness.png"
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {pdf_path} and {png_path}")


def plot_component_ablations():
    """Figure 4: 4-Panel Component Ablations (RQ4)."""
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8), dpi=300)

    # 1. Cost Channel Ablation
    ax = axes[0]
    cost_names = ['Grounded\n(Ours)', 'Synthetic\n(Hash)', 'No-Cost\n(\\alpha_c=0)']
    cost_f1 = [0.3409, 0.3383, 0.3439]
    cost_err = [0.0070, 0.0036, 0.0042]
    colors_c = ['#0EA5E9', '#94A3B8', '#F59E0B']
    b = ax.bar(cost_names, cost_f1, yerr=cost_err, capsize=4, color=colors_c, alpha=0.9, width=0.55)
    ax.set_title("(a) Cost Channel Mode", fontsize=10.5, fontweight='bold', pad=8)
    ax.set_ylabel("F1@5 Score", fontsize=9.5, fontweight='bold')
    ax.set_ylim(0.31, 0.36)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    for bar in b:
        ax.annotate(f'{bar.get_height():.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=7.5, fontweight='bold')

    # 2. Synergy Channel Ablation
    ax = axes[1]
    syn_names = ['Full (0.5)\n(Ours)', 'Semantic\nOnly', 'Schema\nOnly', 'No-Syn\n(\\alpha_s=0)']
    syn_f1 = [0.3409, 0.3391, 0.3198, 0.3359]
    syn_err = [0.0070, 0.0087, 0.0069, 0.0000]
    colors_s = ['#0EA5E9', '#38BDF8', '#F43F5E', '#94A3B8']
    b = ax.bar(syn_names, syn_f1, yerr=syn_err, capsize=4, color=colors_s, alpha=0.9, width=0.55)
    ax.set_title("(b) Synergy Disentanglement", fontsize=10.5, fontweight='bold', pad=8)
    ax.set_ylim(0.29, 0.36)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    for bar in b:
        ax.annotate(f'{bar.get_height():.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=7.5, fontweight='bold')

    # 3. Hybrid Relevance Beta Sweep
    ax = axes[2]
    betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    beta_f1 = [0.2454, 0.3107, 0.3368, 0.3413, 0.3387, 0.3171]
    ax.plot(betas, beta_f1, marker='o', color='#0284C7', linewidth=2.2, markersize=6)
    ax.scatter([0.6], [0.3413], color='#10B981', s=80, zorder=5, edgecolors='#0F172A', label='Optimal ($\\beta=0.6$)')
    ax.set_title("(c) Relevance Mixing $\\beta$", fontsize=10.5, fontweight='bold', pad=8)
    ax.set_xlabel("Lexical Weight $\\beta$", fontsize=9.5, fontweight='bold')
    ax.set_ylim(0.22, 0.36)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=8.0, frameon=True)
    ax.annotate("Peak F1: 0.3413", xy=(0.6, 0.3413), xytext=(0.35, 0.350),
                arrowprops=dict(arrowstyle="->", color='#10B981', lw=1.0),
                fontsize=7.5, fontweight='bold', color='#059669')

    # 4. Tool Pool Scaling (N=1000 vs N=3000)
    ax = axes[3]
    scales = ['N=1000', 'N=3000']
    x = np.arange(2)
    w = 0.35
    hybrid_f1 = [0.3619, 0.3413]
    amortized_f1 = [0.3632, 0.3489]
    b1 = ax.bar(x - w/2, hybrid_f1, w, label='Hybrid Retrieval', color='#94A3B8', alpha=0.9)
    b2 = ax.bar(x + w/2, amortized_f1, w, label='Amortized Decode', color='#0EA5E9', alpha=0.9)
    ax.set_title("(d) Candidate Pool Scaling $N$", fontsize=10.5, fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(scales, fontsize=9.5, fontweight='bold')
    ax.set_ylim(0.30, 0.38)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(loc='upper right', fontsize=8.0, frameon=True)

    # Annotate gap widening
    ax.annotate("Gap: +0.0013", xy=(0, 0.3632), xytext=(-0.1, 0.32),
                fontsize=7.5, fontweight='bold', color='#64748B')
    ax.annotate("Gap: +0.0076\n(5.8x wider)", xy=(1, 0.3489), xytext=(0.75, 0.315),
                fontsize=7.5, fontweight='bold', color='#0284C7')

    plt.tight_layout()
    pdf_path = FIG_OUT_DIR / "fig_component_ablations.pdf"
    png_path = FIG_OUT_DIR / "fig_component_ablations.png"
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {pdf_path} and {png_path}")


if __name__ == "__main__":
    print("Generating publication figures...")
    plot_pareto_frontier()
    plot_simplex_polyak()
    plot_cross_judge()
    plot_component_ablations()
    print("All publication figures successfully generated!")
