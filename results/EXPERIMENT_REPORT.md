# Continuous Domain-Amortized Evolution 实验对比报告 (v2)

## 实验设定
- **评估模式**: `judge` (`deepseek-v4-flash`, 训练/测试零 gold 泄露)
- **演化超参数**: P=8, G=5, alpha 约束于单纯形 (sum=1), sigma=sigma0*||alpha||
- **摊销先验**: 训练轨迹 Polyak 均值 a_mean (v1 使用尾部 a30, 仅作消融)
- **数据**: 5 Catalog, 每类训练 30 条 / 测试 10 条; 路由准确率 **86.0%** (43/50)
- **judge-gold 相关性**: 0.636

## 核心指标对比表

| 类别 | 方法 | F1@5 | Recall@5 | NDCG@5 | Judge Reward | 测试开销 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 直接解码 | BM25 top-5 (retrieval baseline) | 0.2723 | 0.4150 | 0.3683 | 0.4493 | 0 LLM |
| 直接解码 | Dense cosine top-5 (retrieval baseline) | 0.2121 | 0.3400 | 0.3179 | 0.4080 | 0 LLM |
| 直接解码 | Global Default (norm. [0.80, 0.16, 0.04]) | 0.2483 | 0.3940 | 0.3606 | 0.4334 | 0 LLM |
| 直接解码 | Cold a0 (random init) | 0.2073 | 0.3183 | 0.2905 | 0.3512 | 0 LLM |
| 直接解码 | Tail a30 (v1 prior, ablation) | 0.1586 | 0.2567 | 0.2233 | 0.3530 | 0 LLM |
| 直接解码 | Amortized a_mean (Ours, Polyak) | 0.2221 | 0.3533 | 0.3356 | 0.4156 | 0 LLM |
| 再演化 | Default -> ES | 0.2536 | 0.4000 | 0.3627 | 0.5994 | 45 LLM |
| 再演化 | Cold a0 -> ES (baseline) | 0.2352 | 0.3650 | 0.3317 | 0.5898 | 45 LLM |
| 再演化 | Amortized a_mean -> ES (Ours) | 0.2593 | 0.4100 | 0.3833 | 0.6026 | 45 LLM |

## 配对检验 (F1@5)
- 再演化 amortized vs cold: diff=+0.0241 (W/T/L = 6/42/2, t=+1.61)
- 解码 amortized vs global default: diff=-0.0261
- 解码 amortized (mean) vs tail: diff=+0.0636
- 再演化 vs 直接解码 (amortized): diff=+0.0371