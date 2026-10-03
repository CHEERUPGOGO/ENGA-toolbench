# Continuous Domain-Amortized Evolution 实验对比报告 (v2)

## 实验设定
- **评估模式**: `judge` (`deepseek-v4-flash`, 训练/测试零 gold 泄露)
- **演化超参数**: P=8, G=5, alpha 约束于单纯形 (sum=1), sigma=sigma0*||alpha||
- **摊销先验**: 训练轨迹 Polyak 均值 a_mean (v1 使用尾部 a30, 仅作消融)
- **数据**: 5 Catalog, 每类训练 30 条 / 测试 10 条; 路由准确率 **86.0%** (43/50)
- **judge-gold 相关性**: 0.698

## 核心指标对比表

| 类别 | 方法 | F1@5 | Recall@5 | NDCG@5 | Judge Reward | 测试开销 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 直接解码 | BM25 top-5 (lexical baseline) | 0.2723 | 0.4150 | 0.3683 | 0.4493 | 0 LLM |
| 直接解码 | Dense cosine top-5 (dense baseline) | 0.2121 | 0.3400 | 0.3179 | 0.4080 | 0 LLM |
| 直接解码 | Hybrid BM25+Dense top-5 (retrieval) | 0.2956 | 0.4560 | 0.4176 | 0.4794 | 0 LLM |
| 直接解码 | Global Default (norm. [0.80, 0.16, 0.04]) | 0.2986 | 0.4543 | 0.4229 | 0.4916 | 0 LLM |
| 直接解码 | Cold a0 (random init) | 0.2844 | 0.4383 | 0.4086 | 0.4810 | 0 LLM |
| 直接解码 | Tail a30 (v1 prior, ablation) | 0.2470 | 0.3793 | 0.3479 | 0.4198 | 0 LLM |
| 直接解码 | Amortized a_mean (Ours, Polyak) | 0.2976 | 0.4517 | 0.4247 | 0.5018 | 0 LLM |
| 再演化 | Default -> ES | 0.3233 | 0.4883 | 0.4441 | 0.6364 | 45 LLM |
| 再演化 | Cold a0 -> ES (baseline) | 0.3520 | 0.5390 | 0.4769 | 0.6648 | 45 LLM |
| 再演化 | Amortized a_mean -> ES (Ours) | 0.3645 | 0.5490 | 0.4779 | 0.6612 | 45 LLM |

## 配对检验 (F1@5)
- 再演化 amortized vs cold: diff=+0.0125 (W/T/L = 3/45/2, t=+0.76)
- 解码 amortized vs global default: diff=-0.0010
- 解码 amortized (mean) vs tail: diff=+0.0506
- 再演化 vs 直接解码 (amortized): diff=+0.0669