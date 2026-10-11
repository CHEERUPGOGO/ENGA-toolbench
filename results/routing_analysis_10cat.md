# Routing-correctness breakdown (seed 42, 10 catalogs, 100 test queries)

routing correct n=72, wrong n=28

| arm | F1 (correct) | F1 (wrong) | delta |
|---|---|---|---|
| decode_hybrid | 0.3321 | 0.3649 | -0.0328 |
| decode_default | 0.3356 | 0.3649 | -0.0293 |
| decode_cold | 0.3055 | 0.3095 | -0.0040 |
| decode_tail | 0.2915 | 0.3662 | -0.0746 |
| decode_warm | 0.3387 | 0.3751 | -0.0364 |
| evo_cold | 0.3536 | 0.3694 | -0.0159 |
| evo_warm | 0.3632 | 0.3840 | -0.0208 |

Top confusions (true -> matched): 3->6 x5, 6->3 x5, 2->3 x4, 5->6 x3, 2->6 x1, 4->1 x1

Finding: mis-routed queries score HIGHER across all arms; confusion concentrates on
near-neighbor domains (TV streaming <-> movie cast/crew, film festival -> TV streaming)
whose priors are near-interchangeable. Routing imprecision does not hurt the amortized prior.