# 域摊销进化的 LLM 工具选择实验

在大规模工具池（ToolRet / ToolBench-web，约 3000 个工具）上，研究**如何为一个领域学出好的贪心解码器权重**：

- 解码器按 `gain = α_rel·相关度 + α_syn·协同度 − α_cost·成本` 逐步选出 b=5 个工具；
- 权重 α 被约束在单纯形上（三维和为 1），用 OpenAI-ES（进化策略）按域优化；
- 训练轨迹的 **Polyak 均值** 作为摊销先验（ours），测试时对比 **直接解码** vs **再演化**；
- 与 **BM25 / Dense 检索基线** 做横向对比，全部由盲评 LLM-as-a-Judge（DeepSeek）打分。

## 目录结构

```
├── run_catalog_experiment.py   # 主实验脚本（训练 + 测试 + 报告）
├── split_catalogs.py           # 从 top_catalogs.json 生成训练/测试划分（数据已附带，可不跑）
├── probe_judge.py              # 用一次真实调用检查 judge 端点是否可用
├── enga/                       # 核心包：数据、解码器、ES、评估器、基线
├── data/
│   ├── toolret/                # ToolRet 工具库与查询（ToolBench web 子集）
│   ├── catalogs/top_catalogs.json  # 5 个领域 catalog（含人工校验的查询集）
│   └── catalog_splits.json     # 每个 catalog 30 条训练 / 10 条测试
└── results/                    # 一次完整正式运行的全部产物（含 LLM 缓存，见下）
```

## 环境配置

Python **3.10+**（代码使用了 `X | Y` 类型语法）。

```bash
pip install -r requirements.txt
```

评估模式二选一：

| 模式 | 需要 API key? | 说明 |
| :--- | :--- | :--- |
| `--eval-mode unsupervised` | 否 | 本地零样本代理打分，跑通流程 / 冒烟测试用 |
| `--eval-mode judge`（默认） | 是 | 真实 LLM-as-a-Judge，`OPENAI_API_KEY` 环境变量提供 |

judge 模式默认走 DeepSeek 端点（`https://api.deepseek.com`，模型 `deepseek-v4-flash`），只需：

```bash
set OPENAI_API_KEY=sk-xxxx        # Windows；Linux/macOS 用 export
```

端点 / 模型如需修改，改 `enga/config.py` 里的 `LLMConfig` 即可（key 一律走环境变量，不要写进代码）。

首次运行会自动下载 `sentence-transformers/all-MiniLM-L6-v2`（约 90MB）并缓存全部嵌入到 `data/embed_cache/`（该目录不入库）。国内网络可先 `set HF_ENDPOINT=https://hf-mirror.com`。

## 快速开始

```bash
# 1) 冒烟测试：无 key、约 1 分钟，确认环境没问题
python run_catalog_experiment.py --eval-mode unsupervised --P 2 --G 1 --train-limit 2 --test-limit 1 --tag smoke

# 2) 检查 judge 端点（需要 OPENAI_API_KEY，一次真实调用）
python probe_judge.py

# 3) 正式实验（judge 模式，5 catalog × 30 训练 + 50 测试查询）
python run_catalog_experiment.py
```

主参数：`--P` 种群规模（默认 8）、`--G` 代数（默认 5）、`--train-limit` / `--test-limit` 每 catalog 查询数、`--workers` LLM 并发（默认 8）、`--tag` 给输出文件加后缀以隔离不同运行。

实验结束后在 `results/` 生成 `EXPERIMENT_REPORT.md`（对比表 + 配对检验）、`comparison_results.json`（逐查询明细）、`catalog_learned_alphas.json`（每域学到的 α）。

## 结果摘要（附带的正式运行，50 条测试查询）

| 类别 | 方法 | F1@5 | Recall@5 | NDCG@5 | Judge Reward | 测试开销 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 直接解码 | BM25 top-5 | 0.2723 | 0.4150 | 0.3683 | 0.4493 | 0 LLM |
| 直接解码 | Dense cosine top-5 | 0.2121 | 0.3400 | 0.3179 | 0.4080 | 0 LLM |
| 直接解码 | Global Default | 0.2483 | 0.3940 | 0.3606 | 0.4334 | 0 LLM |
| 直接解码 | Cold a0（随机初始化） | 0.2073 | 0.3183 | 0.2905 | 0.3512 | 0 LLM |
| 直接解码 | Tail a30（v1 先验，消融） | 0.1586 | 0.2567 | 0.2233 | 0.3530 | 0 LLM |
| 直接解码 | **Amortized a_mean（ours）** | 0.2221 | 0.3533 | 0.3356 | 0.4156 | 0 LLM |
| 再演化 | Default → ES | 0.2536 | 0.4000 | 0.3627 | 0.5994 | 45 LLM |
| 再演化 | Cold a0 → ES | 0.2352 | 0.3650 | 0.3317 | 0.5898 | 45 LLM |
| 再演化 | **Amortized a_mean → ES（ours）** | **0.2593** | **0.4100** | **0.3833** | **0.6026** | 45 LLM |

主要发现：

1. **摊销先验有效**：以 Polyak 均值初始化再演化，比随机初始化（cold）高 +0.024 F1，比 v1 的尾部点（tail）高 +0.064（配对 t=2.52，显著）；
2. **再演化优于直接解码**：摊销方法下再演化比直接解码高 +0.037（t=2.01，显著）；
3. **当前最强检索基线仍是 BM25**（F1 0.2723，零 LLM 开销）——解码器的相关度通道用的是稠密余弦，弱于词面匹配，是下一步改进方向（换 BM25/混合通道）；
4. 域路由准确率 86%（43/50），judge 与 gold 指标相关性 0.636。

开销口径：训练 5×30×(P+1)×G = 6750 次调用；测试阶段每个查询，直接解码方案各 1 次（仅打分）、再演化方案各 (P+1)×G = 45 次，9 个方案共 141 次。

## 复现说明

- **LLM 缓存**：`results/llm_cache.jsonl` 记录了正式运行的全部 judge 调用（key = 模型 + 查询 + 排序后的工具集）。保留它，同配置重跑几乎零成本；删掉它则全量重新计费。
- **断点续跑**：`results/checkpoint_judge.json` 按查询粒度保存进度，中断后重跑同命令会自动跳过已完成部分。checkpoint 带配置签名校验，混用不同配置会拒绝续跑。
- **随机性**：每个（查询, 方案）的 RNG 由稳定哈希派生，与执行顺序、是否中断无关。
- judge 是盲评：只看查询与**排序后**的工具列表，看不到 gold、方法名，temperature=0，避免顺序与标签偏好。

## 数据来源

工具库与查询来自公开的 [ToolRet](https://github.com/zjunlp/ToolRet)（ToolBench web 子集）；5 个 catalog 由其查询集聚类并人工校验得到，划分脚本见 `split_catalogs.py`。
