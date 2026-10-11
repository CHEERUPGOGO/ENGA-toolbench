# Offline Ablation Study Report (10-Catalog Test Set)

## 1. 成本通道消融 (Cost Channel Ablation, Mean ± Std over 3 seeds)
| Cost Mode | Description | F1@5 (Mean ± Std) | Per-Seed F1 (42, 43, 44) |
| :--- | :--- | :---: | :---: |
| grounded | Grounded Cost (Physical Token/Method/Risk metadata) | 0.3409 ± 0.0070 | (0.3489, 0.3381, 0.3357) |
| synthetic | Synthetic Cost (Hash-based pseudo-random) | 0.3383 ± 0.0036 | (0.3424, 0.3356, 0.3369) |
| no_cost | No Cost Channel (alpha_cost = 0) | 0.3439 ± 0.0042 | (0.3485, 0.3429, 0.3403) |

## 2. 协同通道消融 (Synergy Disentanglement, Mean ± Std over 3 seeds)
| Synergy Setting | Description | F1@5 (Mean ± Std) | Per-Seed F1 (42, 43, 44) |
| :--- | :--- | :---: | :---: |
| full_0.5 | Full Synergy (0.5 Semantic + 0.5 Schema) | 0.3409 ± 0.0070 | (0.3489, 0.3381, 0.3357) |
| semantic_only | Semantic Synergy Only (lambda_s = 1.0) | 0.3391 ± 0.0087 | (0.3410, 0.3296, 0.3466) |
| schema_only | Schema Compatibility Only (lambda_s = 0.0) | 0.3198 ± 0.0069 | (0.3249, 0.3226, 0.3119) |
| no_syn | No Synergy Channel (alpha_syn = 0) | 0.3359 ± 0.0000 | (0.3359, 0.3359, 0.3359) |

## 3. 混合通道 Beta 敏感性 (Hybrid Beta Sweep on 100 Test Queries)
| Beta (BM25 weight) | 1 - Beta (Dense weight) | Test Retrieval F1@5 |
| :---: | :---: | :---: |
| 0.0 | 1.0 | 0.2454 |
| 0.2 | 0.8 | 0.3107 |
| 0.4 | 0.6 | 0.3368 |
| 0.6 | 0.4 | 0.3413 |
| 0.8 | 0.2 | 0.3387 |
| 1.0 | 0.0 | 0.3171 |

## 4. 候选库规模扩展性 (Scale Sensitivity: N=1000 vs N=3000)
| Tool Pool Scale N | BM25 | Dense | Hybrid (beta=0.6) | Amortized Decode (Ours) |
| :---: | :---: | :---: | :---: | :---: |
| N=1000 | 0.3470 | 0.2879 | 0.3619 | 0.3632 |
| N=3000 | 0.3171 | 0.2454 | 0.3413 | 0.3489 |