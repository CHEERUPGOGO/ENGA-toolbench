# ENGA: Domain-Amortized Evolutionary Tool Selection for Large Language Model Agents

Official code, benchmark data, experimental artifacts, and LaTeX manuscript for **Domain-Amortized Evolutionary Tool Selection** (`ENGA`).

In large-scale candidate tool pools (`ToolRet` / `ToolBench-web`, $N = 3{,}000$ tools), `ENGA` decouples inner-loop subset construction from expensive black-box LLM evaluation via a three-channel parameterized greedy decoder $D(q; \boldsymbol{\alpha})$ constrained to the clipped probability simplex $\Delta^2_\epsilon$, offline domain-amortized Evolution Strategies (ES) with Polyak-Ruppert trajectory averaging, and a three-tier test-time deployment frontier.

---

## Unified Repository Structure

```text
share_repo/
├── enga/                             # Core ENGA v2 Python package
│   ├── config.py                     # Global hyperparameters & OpenAI-compatible LLM client config
│   ├── data.py                       # ToolRet dataset loader & deterministic sub-library sampler
│   ├── features.py                   # Hybrid relevance (BM25 + SBERT), API-family/schema synergy & grounded cost
│   ├── nga.py                        # Parameterized Nested Greedy Decoder D(q; alpha)
│   ├── es.py                         # Antithetic rank-based ES & Euclidean simplex projection (eps_floor=0.02)
│   ├── evaluator.py                  # Blind LLM-as-a-Judge evaluator, cache persistence & validation
│   └── baselines.py                  # BM25, Dense cosine, and Hybrid retrieval baselines
├── data/                             # Benchmark datasets, domain splits & precomputed embeddings
│   ├── toolret/                      # ToolRet / ToolBench-web API corpus & query annotations (parquet)
│   ├── catalogs/top_catalogs.json    # Clustered domain catalogs
│   ├── catalog_splits10.json         # Primary 10-catalog benchmark split (30 train / 10 test queries per catalog)
│   ├── catalog_splits.json           # 5-catalog diagnostic split
│   └── embed_cache/                  # Precomputed Sentence-BERT (all-MiniLM-L6-v2) embedding cache
├── results/                          # Complete experimental logs, multi-seed checkpoints & LLM judge cache
│   ├── P0_multiseed_summary.md       # Pooled 3-seed (42, 43, 44; 300 test evals) statistical summary
│   ├── offline_ablations_10cat.md    # 4-way component ablations (Cost, Synergy, Hybrid beta, Pool scale N)
│   ├── comparison_results_10cat*.json# Per-seed & cross-judge (DeepSeek-v4-Flash / Gemini-3.8-Flash) results
│   ├── catalog_learned_alphas*.json  # Learned Polyak-Ruppert simplex priors & tail iterates per catalog
│   ├── llm_rerank_10cat.json         # 1-Call LLM Rerank (Hybrid Top-50 -> 5) baseline results
│   ├── pre_experiments_v2.json       # Motivation pre-experiments (Exp 1, 2, 3) on the v2 N=3,000 engine
│   └── llm_cache.jsonl               # Deterministic cache of 22,000+ real LLM-as-a-Judge evaluations
├── paper/                            # ICLR 2026 LaTeX manuscript, figures, tables & Overleaf bundle
│   ├── main.tex                      # Main LaTeX entry point
│   ├── main.pdf                      # Compiled 21-page camera-ready PDF
│   ├── ENGA_ICLR2026_Overleaf.zip    # Ready-to-upload Overleaf package
│   ├── sections/                     # Manuscript sections (introduction, Preliminary, methodnew, Evaluation, Appendix, PMP)
│   ├── tables/                       # LaTeX tables (table.tex, summarization.tex)
│   └── figures/                      # Vector PDF & 300-DPI PNG figures (title-free publication format)
├── run_catalog_experiment.py         # Main domain-amortized training & multi-tier evaluation driver
├── run_ablations_10cat.py            # Offline component ablations across 10 catalogs x 3 seeds (0 LLM calls)
├── run_llm_rerank.py                 # 1-Call LLM Rerank baseline evaluation script
├── run_pre_experiments_v2.py         # Motivation pre-experiments (Exp 1-3) aligned with v2 engine
├── analyze_p0_multiseed.py           # Pooled 3-seed aggregation & paired t-test calculator
├── analyze_routing_pooled.py         # Cross-seed centroid routing accuracy & confusion matrix analyzer
├── plot_publication_figures.py       # Generates the 4 main evaluation figures into paper/figures/
├── plot_pre_experiments.py           # Generates the 3 motivation figures (Exp 1-3) into paper/figures/
├── split_catalogs.py                 # Domain catalog clustering & train/test split generator
├── tune_hybrid_beta.py               # Training-set grid search for hybrid lexical-semantic weight beta
├── probe_judge.py                    # Connectivity probe for OpenAI-compatible LLM Judge endpoints
├── EXPERIMENT_PROTOCOL.md            # Formal mathematical derivations & experimental protocol
└── requirements.txt                  # Python dependencies
```

---

## Quick Start & Reproducibility

### 1. Environment Setup
Requires Python **3.10+**:
```bash
pip install -r requirements.txt
```

### 2. Offline Verification & Figure Reproduction (Zero API Calls Required)
All 22,000+ LLM Judge evaluations and learned domain priors are cached in `results/`. You can reproduce all statistical tables, ablations, and figures completely offline:

```bash
# 1. Reproduce pooled 3-seed (Seeds 42, 43, 44) main results & paired t-tests
python analyze_p0_multiseed.py

# 2. Reproduce cross-seed centroid routing analysis (93.0% accuracy)
python analyze_routing_pooled.py

# 3. Run the 4 component ablations (Cost channel, Synergy channel, Hybrid beta, Pool scale N)
python run_ablations_10cat.py

# 4. Run Motivation Pre-Experiments (Exp 1, 2, 3) and regenerate motivation figures
python run_pre_experiments_v2.py
python plot_pre_experiments.py

# 5. Regenerate the 4 main publication figures (PDF & PNG in paper/figures/)
python plot_publication_figures.py
```

### 3. Running Full Online Evolution with an LLM Judge Endpoint
To run fresh domain-amortized training and evaluation against a live OpenAI-compatible endpoint (`DeepSeek-v4-Flash` or `Gemini-3.8-Flash`):

```bash
# Set your API key
export OPENAI_API_KEY=sk-xxxx   # Linux/macOS
set OPENAI_API_KEY=sk-xxxx      # Windows

# Run 10-catalog main experiment (Seeds 42, 43, 44)
python run_catalog_experiment.py --splits data/catalog_splits10.json --seed 42 --tag 10cat
python run_catalog_experiment.py --splits data/catalog_splits10.json --seed 43 --pool-seed 42 --tag 10cat_s43
python run_catalog_experiment.py --splits data/catalog_splits10.json --seed 44 --pool-seed 42 --tag 10cat_s44

# Run Cross-Judge generalization check with Gemini-3.8-Flash
python run_catalog_experiment.py --splits data/catalog_splits10.json --llm-model gemini-3.8-flash --tag 10cat_gemini
```

### 4. Compiling the Paper
```bash
cd paper
xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

---

## Summary of Main Experimental Results (10 Catalogs $\times$ 3 Seeds = 300 Test Evaluations, $N=3{,}000$)

| Deployment Tier | Method | LLM Calls / Query | F1@5 ($\uparrow$) | Recall@5 ($\uparrow$) | NDCG@5 ($\uparrow$) | Judge Utility ($\uparrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Tier 1: Zero-Cost** | Dense Cosine Top-5 (MiniLM) | 0 | $0.2454 \pm 0.0000$ | $0.4008 \pm 0.0000$ | $0.3786 \pm 0.0000$ | $0.4910 \pm 0.0000$ |
| **Tier 1: Zero-Cost** | Cold Init Decode ($\boldsymbol{\alpha}_0$) | 0 | $0.2827 \pm 0.0210$ | $0.4534 \pm 0.0329$ | $0.4251 \pm 0.0278$ | $0.5629 \pm 0.0213$ |
| **Tier 1: Zero-Cost** | BM25 Lexical Top-5 | 0 | $0.3171 \pm 0.0000$ | $0.5000 \pm 0.0000$ | $0.4683 \pm 0.0000$ | $0.5947 \pm 0.0000$ |
| **Tier 1: Zero-Cost** | Tail Iterate Decode ($\boldsymbol{\mu}_{z,T}$, Ablation) | 0 | $0.3214 \pm 0.0120$ | $0.5063 \pm 0.0185$ | $0.4826 \pm 0.0132$ | $0.6095 \pm 0.0186$ |
| **Tier 1: Zero-Cost** | Hybrid Retrieval ($\beta = 0.6$) | 0 | $0.3413 \pm 0.0000$ | $0.5380 \pm 0.0000$ | $\mathbf{0.4993 \pm 0.0000}$ | $0.6305 \pm 0.0000$ |
| **Tier 1: Zero-Cost** | **Amortized Decode ($\bar{\boldsymbol{\alpha}}_z$, Ours)** | **0** | $\mathbf{0.3409 \pm 0.0070}$ | $\mathbf{0.5342 \pm 0.0108}$ | $0.4933 \pm 0.0061$ | $\mathbf{0.6350 \pm 0.0044}$ |
| **Tier 2: Single-Call** | 1-Call LLM Rerank (Hybrid Top-50 $\to$ 5) | 1 | $0.3497 \pm 0.0000$ | $0.5447 \pm 0.0000$ | $0.5051 \pm 0.0000$ | $0.6844 \pm 0.0000$ |
| **Tier 3: Re-Evolution** | Cold Re-Evolution ($\boldsymbol{\alpha}_0 \to \text{ES}$) | 45 | $0.3493 \pm 0.0078$ | $0.5529 \pm 0.0105$ | $0.5071 \pm 0.0206$ | $0.7136 \pm 0.0091$ |
| **Tier 3: Re-Evolution** | **Warm Re-Evolution ($\bar{\boldsymbol{\alpha}}_z \to \text{ES}$, Ours)** | **45** | $\mathbf{0.3622 \pm 0.0054}$ | $\mathbf{0.5736 \pm 0.0092}$ | $\mathbf{0.5218 \pm 0.0014}$ | $\mathbf{0.7289 \pm 0.0057}$ |

### Key Paired Hypothesis Tests ($N_{\text{eval}} = 300$)
- **Amortized Decode ($\bar{\boldsymbol{\alpha}}_z$) vs. Cold Init ($\boldsymbol{\alpha}_0$)**: $\Delta \text{F1@5} = +0.0582$ ($t = +4.92, p < 0.001$, Win/Tie/Loss = $112/148/40$)
- **Polyak Mean ($\bar{\boldsymbol{\alpha}}_z$) vs. Tail Iterate ($\boldsymbol{\mu}_{z,T}$)**: $\Delta \text{F1@5} = +0.0194$ ($t = +2.69, p = 0.007 < 0.01$, Win/Tie/Loss = $55/212/33$)
- **Warm Re-Evolution vs. Cold Re-Evolution**: $\Delta \text{F1@5} = +0.0128$ ($t = +2.14, p = 0.033 < 0.05$, Win/Tie/Loss = $55/211/34$); under independent `Gemini-3.8-Flash` evaluation, $\Delta \text{F1@5} = +0.0200$ ($t = +2.04, p < 0.05$, reaching $0.3823$ F1@5 and $0.5980$ Recall@5).
