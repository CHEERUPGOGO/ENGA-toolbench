# Continuous Domain-Amortized Evolution 实验对比报告 (v2)

## 实验设定
- **评估模式**: `judge` (`deepseek-v4-flash`, 训练/测试零 gold 泄露)
- **演化超参数**: P=8, G=5, alpha 约束于单纯形 (sum=1), sigma=sigma0*||alpha||
- **摊销先验**: 训练轨迹 Polyak 均值 a_mean (v1 使用尾部 a30, 仅作消融)
- **数据**: 10 Catalog, 每类训练 30 条 / 测试 10 条; 路由准确率 **72.0%** (72/100)
- **judge-gold 相关性**: 0.620

## 核心指标对比表

| 类别 | 方法 | F1@5 | Recall@5 | NDCG@5 | Judge Reward | 测试开销 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 直接解码 | BM25 top-5 (lexical baseline) | 0.3171 | 0.5000 | 0.4683 | 0.4941 | 0 LLM |
| 直接解码 | Dense cosine top-5 (dense baseline) | 0.2454 | 0.4008 | 0.3786 | 0.4190 | 0 LLM |
| 直接解码 | Hybrid BM25+Dense top-5 (retrieval) | 0.3413 | 0.5380 | 0.4992 | 0.5076 | 0 LLM |
| 直接解码 | Global Default (norm. [0.80, 0.16, 0.04]) | 0.3438 | 0.5413 | 0.5014 | 0.5072 | 0 LLM |
| 直接解码 | Cold a0 (random init) | 0.2673 | 0.4303 | 0.4058 | 0.4145 | 0 LLM |
| 直接解码 | Tail a30 (v1 prior, ablation) | 0.3166 | 0.4980 | 0.4846 | 0.4765 | 0 LLM |
| 直接解码 | Amortized a_mean (Ours, Polyak) | 0.3357 | 0.5288 | 0.4871 | 0.4943 | 0 LLM |
| 再演化 | Cold a0 (random init) -> ES | 0.3439 | 0.5413 | 0.4883 | 0.6386 | 45 LLM |
| 再演化 | Amortized a_mean (Ours, Polyak) -> ES | 0.3615 | 0.5672 | 0.5203 | 0.6579 | 45 LLM |

## 配对检验 (F1@5)
- 再演化 amortized vs cold: diff=+0.0176 (W/T/L = 10/86/4, t=+1.72)
- 解码 amortized vs global default: diff=-0.0081
- 解码 amortized (mean) vs tail: diff=+0.0190
- 解码 amortized vs BM25: diff=+0.0186
- 解码 amortized vs Hybrid: diff=-0.0056
- 再演化 vs 直接解码 (amortized): diff=+0.0259