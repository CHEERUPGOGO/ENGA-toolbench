# 实验设置全量说明（v3：hybrid + grounded）

本文档描述当前默认配置下的完整实验设置，所有数字以代码为准（`run_catalog_experiment.py` + `enga/`）。
上一版（v2）为纯 Dense 通道 + synthetic 成本，结果对照见 README 的折叠表。

## 1. 任务定义

从 ~3000 工具的池中为每个查询选 **b=5** 个工具。贪心解码器由 3 维权重
**α = (α_rel, α_syn, α_cost)** 参数化，约束在单纯形上（三分量非负、和恒为 1），
用 OpenAI-ES 按"领域 catalog"分别进化 α；训练轨迹的 **Polyak 均值**作为摊销先验。
测试时对比"直接解码"与"再演化"，并与 BM25/Dense/Hybrid 检索基线横向对比，
全部由盲评 LLM-as-a-Judge（DeepSeek）打分。

## 2. 数据

| 项 | 值 |
|---|---|
| 工具库 | ToolRet 的 ToolBench-web 子集，共 **37,292** 个工具（文档 100% 可解析为 JSON；3 处 RapidAPI 遗留示例 token 已替换为占位符 `REDACTED_GITHUB_EXAMPLE_TOKEN`） |
| 实验子库 | **N = 3000**：先收集全部 200 条查询的 gold 工具强制入池，再从其余工具随机补齐（`rng.permutation`，seed=42），按 id 排序。见 `enga/data.py :: build_experiment_set` |
| Catalog | `data/catalogs/top_catalogs.json` 按 centroid_cohesion 取 top5：二维码/条码生成、影视流媒体推荐、电影节发现、新闻聚合、派对策划贺卡 |
| 划分 | 每 catalog **30 条训练 / 10 条测试** → 150 train + 50 test（`data/catalog_splits.json`，由 `split_catalogs.py` 生成；质心 = 30 条训练查询 MiniLM 嵌入均值后归一化） |
| 路由 | 测试查询 MiniLM 归一化嵌入与 5 个质心点积取 argmax；路由准确率 86%（43/50）；**错路由照常评测**，使用被路由到的 catalog 的先验 |

## 3. 特征与检索通道（`enga/features.py :: ToolIndex`）

- **稠密通道**：`sentence-transformers/all-MiniLM-L6-v2`，嵌入文本 =
  `name, category, method, description, required: …, optional: …`，L2 归一化后余弦；
  嵌入缓存于 `data/embed_cache/`（不入库，首跑自动生成）
- **BM25**：k1 = 1.2（公式中 2.2 = k1+1）、b = 0.75，对池内全部文档建词表
- **混合相关度通道 `rel_norm`**（解码器实际使用）：

  ```
  rel = β · minmax(BM25) + (1 − β) · minmax(dense)，β = 0.6
  ```

  两通道各自按查询 min-max。β 的选择依据：`tune_hybrid_beta.py` 在 **150 条训练查询**
  上按 gold F1@5 扫描 β ∈ {0, 0.1, …, 1.0} 及 RRF 融合——β=0.6 最优（0.3011），
  且 0.4–0.6 为平台区；测试集参考 0.2956 > BM25 0.2723 > dense 0.2121。
  `--no-hybrid` 可切回纯 Dense 通道做消融。

## 4. 贪心解码器 `decode`（`enga/nga.py`，零 LLM 调用）

每步对池内全部工具算边际增益，argmax 选 1 个，已选置 −inf，共 b=5 步：

```
gain(v) = α_rel · rel(v) + α_syn · [ λ_s · sem_syn(v) + (1 − λ_s) · sch_syn(v) ] − α_cost · cost(v)
```

其中 λ_s = 0.5。

- **sem_syn**（语义协同）：候选与已选工具同 API family +0.7、同 category +0.3、
  嵌入余弦 > 0.95 罚 0.3，截断到 [0, 1]
- **sch_syn**（结构协同）：候选 required 参数名/类型与已选工具输出字段的匹配比例，对已选取 max

## 5. 成本模型（`cost_mode = "grounded"`，v3 默认）

三部分线性组合后整体 min-max 到 [0, 1]，权重 λ = (λ_t, λ_τ, λ_f) = (0.34, 0.33, 0.33)：

| 分量 | 定义 |
|---|---|
| L（Prompt 体积） | `tokens(doc_text) + 8·|required| + 4·|optional|`，除以池内最大值 |
| T（调用负载） | `(|required| + 0.5·|optional|) × 方法系数`；POST/PUT/DELETE = 1.5，其余 1.0；归一 |
| R（风险先验，**启发式**） | 类别基础分：{Social, Finance, News_Media, Media, SMS, Communication} = 0.35；{Entertainment, Sports, Movies} = 0.20；其余 = 0.08。描述 < 20 字符 +0.20；required > 3 个 +0.15；上限 1.0 |

数据事实：本数据集中 `category_name` 覆盖率约 38%、`method` 约 38.7%，
未标注工具落入最低风险档。`--cost-mode synthetic` 可切回旧 hash 版本（延迟/失败率
由工具 id 哈希生成，与真实属性无关，仅作对照）。

## 6. 进化策略（`run_catalog_experiment.py :: parallel_es_step / run_query_evolution`）

| 参数 | 值 | 说明 |
|---|---|---|
| P | 8 | 种群规模，**对称（antithetic）采样**：4 对 ±ε |
| G | 5 | 代数 |
| η | 0.7 | 学习率 |
| σ₀ | 0.4 | σ = σ₀ · ‖μ‖（**相对步长**；单纯形上 ‖μ‖ ∈ [1/√3, 1]，暖/冷起点扰动幅度公平） |
| σ 衰减 | ×0.85 / 代 | |
| 候选生成 | `project_simplex(μ + σε)` | Duchi 欧氏投影到单纯形 |
| 梯度 | `Σᵢ centered_ranks(rᵢ) · εᵢ / (P·σ)` | 秩归一到 [−0.5, 0.5]，并列取平均，全同则梯度为 0 |
| 更新 | `μ ← project_simplex(μ + 0.7 · grad)` | |
| 精英锚点 | 每代额外评估 μ 自身（+1 次调用） | 参与"最优组合保留"，不进梯度 |
| 奖励 | judge 对该候选解码出的工具集打分 | 1 次 LLM 调用 / 候选 |

每查询一次完整 ES = (P+1) × G = **45 次** LLM 调用。

> 注：`enga/es.py :: enaga_search`（含 QD archive）是早期单查询版本的库代码，
> catalog 实验并未走它——实际 ES 是脚本内联实现，数学一致但带单纯形投影与并行化。

## 7. 训练协议（Phase 1，5 catalog 并行）

- 每 catalog 初始 **α₀ ~ U(0.3, 1.2)³ 后归一化**（RNG key：`a0:{cid}`）
- **顺序链**：30 条训练查询依次跑 ES，query t 从 query t−1 结束的 μ 出发；
  轨迹 = [α₀, μ₁, …, μ₃₀]
- 每查询独立 RNG（key：`train:{cid}:{step}`），与执行顺序/中断无关
- **摊销先验 = Polyak 均值** = mean(轨迹[1:])（去掉初始 α₀）；末点 a₃₀ 仅作消融（tail）
- 并发：catalog × 5（ThreadPool）、LLM × 8 workers

## 8. 测试协议（Phase 2，50 查询并行 × 5）

每查询评 **9 个方案**，所有起点 α 均在单纯形上：

| 组 | 方案 | 起点α | judge 调用 |
|---|---|---|---|
| 直接解码 | Global Default | [1, 0.2, 0.05] / 1.25 = [0.80, 0.16, 0.04]（**保比例**均匀归一；欧氏投影会畸变为 [0.9, 0.1, 0] 杀掉成本权重，故不用） | 1 |
| | Cold a0 | project(α₀) | 1 |
| | Tail a30 | project(轨迹末点) | 1 |
| | **Amortized a_mean（ours）** | project(Polyak 均值) | 1 |
| 检索基线 | BM25 / Dense / Hybrid | 无 α，各自 top-5 | 各 1 |
| 再演化 | Default / Cold / **Warm（ours）** → ES | 同名起点各跑一次完整 ES | 各 45 |

（直接解码/检索基线的 1 次调用仅为打 reward 列，不影响选择结果。）

**指标**：F1@5（集合 F1）、Recall@5、NDCG@5（二值增益 + log₂ 折扣）；
配对检验给出均值差 + 胜/平/负 + t 统计量；judge-gold 相关性 =
各再演化方案 (reward, F1) 的相关系数（v3 运行 = 0.698）。

## 9. LLM-as-a-Judge（`enga/evaluator.py :: LLMJudgeUtility`）

- **端点**：DeepSeek `deepseek-v4-flash`，temperature = 0，max_tokens = 2048，
  timeout = 120s，带重试；API key 只从 `OPENAI_API_KEY` 环境变量读取
- **盲评**：只看 查询 + **排序后**的工具列表（每条含 id / name / category /
  必填参数 / 描述前 300 字符）；**看不到 gold、看不到方法名**。
  排序渲染同时消除 LLM 位置偏置、并保证同一集合永远同一提示词
- 输出 JSON {completeness, relevance, redundancy_penalty, utility, thought}；
  直接取 utility；缺 utility 时 fallback `0.7·comp + 0.3·rel − 0.2·red`；
  解析失败计 0.1；**截断（finish_reason = length）不写缓存**
- **缓存**：key = `judge2::{model}::{qid}::{排序后工具 id}`，
  `results/llm_cache.jsonl` 追加式持久化，内存 + 文件双写，锁保护

## 10. 工程与复现保障

- **断点续跑**：checkpoint（`results/checkpoint_judge.json`）带**配置签名**
  （评估模式/模型/P/G/限额/seed/单纯形/相对 σ/η/σ₀/衰减/提示词版本/cost_mode/β/use_hybrid），
  签名不匹配直接拒绝续跑；训练阶段每查询存档（`partial_catalogs`），测试按 qid 幂等
- **确定性**：所有随机性走 `np.random.SeedSequence(seed, spawn_key=(stable_hash(key),))`，
  结果与执行顺序、是否中断、运行目录无关（已跨目录/跨布局两次验证逐位一致）
- **线程安全**：OpenAI client 双检锁初始化、缓存写锁、checkpoint 写锁

## 11. 调用开销口径

| 阶段 | 计算 | 次数 |
|---|---|---|
| 训练 | 5 catalog × 30 查询 × (P+1)×G | 6,750 |
| 测试 | 每查询 6×1（解码/基线打分）+ 3×45（再演化）= 141 | 7,050 |
| 合计 | | 13,800（缓存命中不计费） |

v3 正式运行因复用 v2 缓存实际新增 2,933 次调用。

## 12. CLI 默认值总表

| 参数 | 默认 | 含义 |
|---|---|---|
| `--eval-mode` | judge | judge（DeepSeek 盲评）/ unsupervised（本地代理，冒烟用） |
| `--P` / `--G` | 8 / 5 | 种群规模 / 代数 |
| `--train-limit` / `--test-limit` | 30 / 10 | 每 catalog 查询数 |
| `--seed` | 42 | 全局种子 |
| `--workers` | 8 | LLM 并发 |
| `--catalog-parallel` / `--query-parallel` | 5 / 5 | catalog / 测试查询并发 |
| `--hybrid-beta` | 0.6 | 混合通道 BM25 权重（训练集上调出） |
| `--cost-mode` | grounded | grounded / synthetic / schema |
| `--no-hybrid` | 关 | 切回纯 Dense 相关度通道 |
| `--tag` | 空 | 输出文件后缀，隔离不同运行 |

## 附录：v3 正式运行各 catalog 学到的 α（四舍五入；完整精度见 `results/catalog_learned_alphas.json`）

| catalog | tail a₃₀ | Polyak 均值（ours） |
|---|---|---|
| 1 | [0.770, 0.230, 0.000] | [0.624, 0.219, 0.158] |
| 2 | [1.000, 0.000, 0.000] | [0.662, 0.222, 0.116] |
| 3 | [0.842, 0.000, 0.158] | [0.705, 0.236, 0.059] |
| 4 | [0.000, 0.214, 0.786] | [0.596, 0.182, 0.222] |
| 5 | [0.555, 0.000, 0.445] | [0.657, 0.250, 0.092] |

可见末点常冲向极端（cat 4 甚至全成本），而 Polyak 均值稳定保持"相关度主导"——
这正是摊销先验取均值而非末点的原因。
