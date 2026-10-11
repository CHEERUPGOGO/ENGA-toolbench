# Continuous Domain-Amortized Evolution 实验对比报告 (v2)

## 实验设定
- **评估模式**: `judge` (`deepseek-v4-flash`, 训练/测试零 gold 泄露)
- **演化超参数**: P=8, G=5, alpha 约束于单纯形 (sum=1), sigma=sigma0*||alpha||
- **摊销先验**: 训练轨迹 Polyak 均值 a_mean (v1 使用尾部 a30, 仅作消融)
- **数据**: 10 Catalog, 每类训练 30 条 / 测试 10 条; 路由准确率 **72.0%** (72/100)
- **judge-gold 相关性**: 0.583

## 核心指标对比表

| 类别 | 方法 | F1@5 | Recall@5 | NDCG@5 | Judge Reward | 测试开销 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 直接解码 | BM25 top-5 (lexical baseline) | 0.3171 | 0.5000 | 0.4683 | 0.4941 | 0 LLM |
| 直接解码 | Dense cosine top-5 (dense baseline) | 0.2454 | 0.4008 | 0.3786 | 0.4190 | 0 LLM |
| 直接解码 | Hybrid BM25+Dense top-5 (retrieval) | 0.3413 | 0.5380 | 0.4992 | 0.5076 | 0 LLM |
| 直接解码 | Cold a0 (random init) | 0.3066 | 0.4855 | 0.4570 | 0.4616 | 0 LLM |
| 直接解码 | Tail a30 (v1 prior, ablation) | 0.3121 | 0.4938 | 0.4708 | 0.4591 | 0 LLM |
| 直接解码 | Amortized a_mean (Ours, Polyak) | 0.3290 | 0.5155 | 0.4812 | 0.5100 | 0 LLM |
| 再演化 | Cold a0 (random init) -> ES | 0.3551 | 0.5605 | 0.5264 | 0.6581 | 45 LLM |
| 再演化 | Amortized a_mean (Ours, Polyak) -> ES | 0.3587 | 0.5630 | 0.5313 | 0.6641 | 45 LLM |

## 配对检验 (F1@5)
- 再演化 amortized vs cold: diff=+0.0036 (W/T/L = 6/89/5, t=+0.27)
- 解码 amortized (mean) vs tail: diff=+0.0169
- 解码 amortized vs BM25: diff=+0.0120
- 解码 amortized vs Hybrid: diff=-0.0123
- 再演化 vs 直接解码 (amortized): diff=+0.0297