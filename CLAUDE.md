# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Domain-amortized evolution for LLM tool selection: a greedy decoder picks b=5 tools from a ~3000-tool ToolRet (ToolBench-web) pool, parameterized by α=(α_rel, α_syn, α_cost) on the simplex (sum=1). The relevance channel is hybrid (β·BM25 + (1−β)·dense, per-query min-max; β=0.6 tuned on train queries only via `tune_hybrid_beta.py`) and the cost model is "grounded" (doc/param prompt size, param×HTTP-method payload, heuristic category-based risk prior; `category_name` coverage in the data is only ~38%). OpenAI-ES optimizes α per domain catalog; the Polyak mean of the training trajectory is the amortized prior ("ours"). Test time compares direct decode vs re-evolution against BM25/dense/hybrid retrieval baselines, scored by a blind LLM-as-a-Judge (DeepSeek).

## Commands

```bash
# smoke test, no API key needed (~1 min)
python run_catalog_experiment.py --eval-mode unsupervised --P 2 --G 1 --train-limit 2 --test-limit 1 --tag smoke

# verify the judge endpoint with one real call (needs OPENAI_API_KEY)
python probe_judge.py

# full run (judge mode, needs OPENAI_API_KEY)
python run_catalog_experiment.py

# resweep the hybrid beta on train queries (gold-labeled, zero LLM cost)
python tune_hybrid_beta.py
```

`--tag` suffixes all output files so runs stay isolated. Outputs land in `results/`: `EXPERIMENT_REPORT.md`, `comparison_results.json` (per-query detail), `catalog_learned_alphas.json`, `checkpoint_<mode><tag>.json`.

`split_catalogs.py` regenerates `data/catalog_splits.json` from `data/catalogs/top_catalogs.json` (30 train / 10 test per catalog, top-5 by centroid cohesion) — only needed if the catalog selection changes.

The API key is read from the `OPENAI_API_KEY` env var via `enga/config.py` (`LLMConfig`). Never hardcode it.

## Architecture

`run_catalog_experiment.py` orchestrates everything and depends on the `enga/` package:

- `enga/data.py` — `load_toolret()` builds the tool pool + query set; `build_experiment_set()` samples the sub-library with gold tools force-included.
- `enga/features.py` — `ToolIndex`: dense embeddings (sentence-transformers, cached under `data/embed_cache/`) and BM25 scores over tool docs.
- `enga/nga.py` — `decode(index, q, alpha, b)`: greedy decoder, gain = α_rel·rel + α_syn·syn − α_cost·cost.
- `enga/es.py` — `enaga_search`: OpenAI-ES with antithetic sampling and centered-rank gradients.
- `enga/evaluator.py` — `LLMJudgeUtility` (blind judge, JSON prompt), oracle utilities, `EvalBudget`, and the `results/llm_cache.jsonl` cache (key = `judge2::{model}::{qid}::{sorted tool ids}`).
- `enga/baselines.py` — `method_bm25` / `method_dense` / `method_hybrid` retrieval baselines (zero LLM cost).
- `enga/config.py` — all hyperparameters (P=8, G=5, b=5, eta=0.7, sigma0=0.4, sigma_decay=0.85, cost_mode="grounded", use_hybrid=True, hybrid_beta=0.6).

Scripts are dual-layout: `enga/` next to the script (this repo) or `../experiments/` (original workspace) — resolved at import time.

## Invariants (do not break)

- **Simplex constraint**: every α used by decode/ES is projected to sum=1 (`project_simplex`, Euclidean projection). Note Euclidean projection is NOT ratio-preserving — the default arm uses uniform normalization of (1.0, 0.2, 0.05) → /1.25 instead.
- **Amortized prior = Polyak mean** of the training trajectory (`polyak_mean`, mean of traj[1:]), not the tail point (tail is kept only as an ablation arm).
- **Relative sigma**: sigma = sigma0 · ||mu|| with decay, so warm (large ||μ||) and cold (small) starts get proportionally fair perturbations.
- **Deterministic per-(query, arm) RNG**: `np.random.SeedSequence(seed, spawn_key=(stable_hash(key) % 2**32,))` — results are order- and interruption-independent.
- **Checkpoint signature guard**: the checkpoint stores a config signature and refuses to resume under a different config; per-query partial results make resume crash-safe.
- **Judge blindness**: the judge sees only the query and the SORTED tool list (no gold, no method labels, temp=0). Any prompt change must bump the `judge2::` cache namespace.
- **Hybrid beta is selected on TRAIN queries only** (`tune_hybrid_beta.py`, β=0.6 on a 0.4–0.6 plateau). Never retune it on test queries, and note the unsupervised proxy utility shares the decoder's hybrid signal — unsupervised numbers cannot evidence hybrid's benefit, only judge-mode runs can.
- **Thread safety**: catalogs run in parallel (only the α chain inside one catalog is sequential); the OpenAI client init, cache writes, and checkpoint writes all take locks. Use `ckpt_mutate` for checkpoint updates.
- Cost accounting: each evolution arm costs (P+1)×G = 45 judge calls per query; each decode arm costs 1 (reward scoring only).

## Data

`data/toolret/` is the ToolRet ToolBench-web subset; `data/catalogs/top_catalogs.json` holds the 5 hand-curated domain catalogs; `data/catalog_splits.json` the train/test split. `data/embed_cache/` is regenerated on first run (gitignored).
