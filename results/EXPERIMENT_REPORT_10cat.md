# Continuous Domain-Amortized Evolution 实验对比报告 (v2)

## 实验设定
- **评估模式**: `judge` (`deepseek-v4-flash`, 训练/测试零 gold 泄露)
- **演化超参数**: P=8, G=5, alpha 约束于单纯形 (sum=1), sigma=sigma0*||alpha||
- **摊销先验**: 训练轨迹 Polyak 均值 a_mean (v1 使用尾部 a30, 仅作消融)
- **数据**: 10 Catalog, 每类训练 30 条 / 测试 10 条; 路由准确率 **72.0%** (72/100)
- **judge-gold 相关性**: 0.608

## 核心指标对比表

| 类别 | 方法 | F1@5 | Recall@5 | NDCG@5 | Judge Reward | 测试开销 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 直接解码 | BM25 top-5 (lexical baseline) | 0.3171 | 0.5000 | 0.4683 | 0.4941 | 0 LLM |
| 直接解码 | Dense cosine top-5 (dense baseline) | 0.2454 | 0.4008 | 0.3786 | 0.4190 | 0 LLM |
| 直接解码 | Hybrid BM25+Dense top-5 (retrieval) | 0.3413 | 0.5380 | 0.4992 | 0.5076 | 0 LLM |
| 直接解码 | Global Default (norm. [0.80, 0.16, 0.04]) | 0.3438 | 0.5413 | 0.5014 | 0.5072 | 0 LLM |
| 直接解码 | Cold a0 (random init) | 0.3066 | 0.4855 | 0.4570 | 0.4616 | 0 LLM |
| 直接解码 | Tail a30 (v1 prior, ablation) | 0.3124 | 0.4955 | 0.4685 | 0.4743 | 0 LLM |
| 直接解码 | Amortized a_mean (Ours, Polyak) | 0.3489 | 0.5488 | 0.4996 | 0.5035 | 0 LLM |
| 再演化 | Global Default (norm. [0.80, 0.16, 0.04]) -> ES | 0.3716 | 0.5813 | 0.5305 | 0.6574 | 45 LLM |
| 再演化 | Cold a0 (random init) -> ES | 0.3580 | 0.5655 | 0.5289 | 0.6457 | 45 LLM |
| 再演化 | Amortized a_mean (Ours, Polyak) -> ES | 0.3690 | 0.5772 | 0.5204 | 0.6471 | 45 LLM |

## 配对检验 (F1@5)
- 再演化 amortized vs cold: diff=+0.0110 (W/T/L = 9/85/6, t=+0.95)
- 再演化 amortized vs default: diff=-0.0026
- 解码 amortized vs global default: diff=+0.0051
- 解码 amortized (mean) vs tail: diff=+0.0364
- 解码 amortized vs BM25: diff=+0.0318
- 解码 amortized vs Hybrid: diff=+0.0076
- 再演化 vs 直接解码 (amortized): diff=+0.0202