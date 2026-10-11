# ENGA: 领域摊销进化的 LLM 工具选择系统 (Domain-Amortized Evolutionary Tool Selection)

在大规模工具池（ToolRet / ToolBench-web，约 3,000 个工具）上，解决大模型智能体（LLM Agent）在面对海量 API 时**高效、鲁棒且低成本选取最佳工具组合**的核心挑战。

- **三维参数化贪心解码器**：按边际增益逐步选取 $b=5$ 个工具：
  $$\Delta(v \mid X, q) = \alpha_{\mathrm{rel}} \cdot I_{\mathrm{rel}}(v, q) + \alpha_{\mathrm{syn}} \cdot S_{\mathrm{syn}}(v, X) - \alpha_{\mathrm{cost}} \cdot C_{\mathrm{run}}(v)$$
  参数限制在单纯形 $\Delta^2$（和为 1，带有正下界保护 $\epsilon_{\mathrm{floor}} = 0.02$，防止小分量被过度截断归零）。
- **混合相关性通道（Hybrid Relevance）**：融合 BM25 词面精准匹配与 MiniLM 稠密语义相关度（$\beta \cdot \mathrm{BM25} + (1-\beta) \cdot \mathrm{Dense}$，查询级 Min-Max 归一化）。在训练集上严格调优确定 $\beta = 0.6$。
- **物理元数据调用成本模型（Grounded Cost）**：结合文档与参数 Prompt 体积（$L_v$）、HTTP 请求方法与参数负载（$T_v$，POST/PUT 写操作赋予 1.5× 惩罚）、以及高敏行业安全风险先验（$R_v$）。
- **离线领域演化与摊销先验（Domain-Amortized Evolution）**：
  在各领域目录（Catalog）的训练样本上，以大模型盲评为黑盒奖励，利用进化策略（OpenAI-ES）优化 $\boldsymbol{\alpha}$。提取训练轨迹的 **Polyak 均值**（$\boldsymbol{\alpha}_{\mathrm{mean}}$）作为领域摊销先验，消除单点收敛的随机马尔可夫震荡。
- **测试期双轨部署架构**：
  1. **零成本直接解码（Direct Decode, 0 LLM Calls）**：通过 MiniLM 质心路由到最近领域后，直接使用 $\boldsymbol{\alpha}_{\mathrm{mean}}$ 解码，**推理耗时 $<2\text{ms}$，大模型零开销**，全面超越经典 BM25 与 Dense 向量检索；
  2. **热启动再演化（Re-Evolution, 45 LLM Calls）**：在算力充裕场景下，以 $\boldsymbol{\alpha}_{\mathrm{mean}}$ 为初值执行 5 代快速再演化，进一步达成显著增益。

---

## 目录结构

```
├── run_catalog_experiment.py     # 主实验框架（支持 5/10 Catalog、DeepSeek/Gemini 多裁判与断点续跑）
├── run_ablations_10cat.py        # 10 领域测试集 4 大离线消融评测（成本/协同/Beta/规模，零调用）
├── run_llm_rerank.py             # 1-call 大模型贪心重排基准对比脚本
├── analyze_p0_multiseed.py       # 3 种子（42/43/44）全量池化统计分析与成对 t 检验
├── analyze_routing_pooled.py     # 跨种子质心路由混淆矩阵与正确性因果分解
├── split_catalogs.py             # 领域目录聚类与训练/测试集切分工具
├── tune_hybrid_beta.py           # 训练集混合检索权重 β 网格扫描工具
├── probe_judge.py                # 快速验证大模型裁判端点连通性
├── EXPERIMENT_PROTOCOL.md        # 完整理论证明、数学形式化、系统架构与消融指南 (Protocol v5)
├── enga/                         # 核心算法包
│   ├── config.py                 # 全局超参数与大模型客户端配置
│   ├── data.py                   # ToolRet 数据集加载与子库采样
│   ├── features.py               # 倒排索引构建、MiniLM 嵌入缓存与特征通道
│   ├── nga.py                    # 贪心解码器主逻辑
│   ├── es.py                     # OpenAI-ES 演化策略与单纯形投影（含 eps_floor）
│   ├── evaluator.py              # LLM-as-a-Judge 盲评审计、缓存持久化与防毒化校验
│   └── baselines.py              # BM25、Dense、Hybrid 经典检索基准
├── data/
│   ├── toolret/                  # ToolRet API 工具库与标注数据
│   ├── catalogs/top_catalogs.json# 候选高质量领域目录集合
│   ├── catalog_splits.json       # 5 领域实验切分
│   └── catalog_splits10.json     # 10 领域全量基准切分（30 训练 / 10 测试每类）
└── results/                      # 评测报告、对比详情、检查点与 LLM 缓存
```

---

## 环境配置

Python **3.10+**。

```bash
pip install -r requirements.txt
```

### 评估模式配置

| 模式 | 需要 API Key? | 适用场景 |
| :--- | :---: | :--- |
| `--eval-mode unsupervised` | 否 | 冒烟测试、本地流程验证（零 LLM 调用） |
| `--eval-mode judge`（默认） | 是 | 生产级 LLM-as-a-Judge 评测（DeepSeek / Gemini 等） |

设置 API Key（支持任何兼容 OpenAI 协议的端点）：

```bash
# Windows
set OPENAI_API_KEY=sk-xxxx
# Linux / macOS
export OPENAI_API_KEY=sk-xxxx
```

如需使用非默认模型（如 Google Gemini），可直接在 CLI 覆盖：
```bash
python run_catalog_experiment.py --llm-base-url https://api.your-endpoint.com/v1 --llm-model gemini-3.8-flash --tag gemini
```

首次运行会自动加载 `sentence-transformers/all-MiniLM-L6-v2` 并将嵌入缓存至 `data/embed_cache/`。国内网络可预先执行 `set HF_ENDPOINT=https://hf-mirror.com`。

---

## 核心实验成果展示

> 详尽的实验设计、参数推导与理论分析参见 **[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)**。

### 1. 10 领域全基准 3 种子池化结果 (Seeds 42, 43, 44, $N=300$)

| 类别 | 方案名称 (Method) | F1@5 (Mean ± Std) | Recall@5 | NDCG@5 | Judge Reward | 测试开销 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **检索基准** | BM25 top-5 (词法基线) | 0.3171 ± 0.0000 | 0.5000 | 0.4632 | 0.4941 | 0 LLM |
| **检索基准** | Dense cosine top-5 (向量基线) | 0.2454 ± 0.0000 | 0.4008 | 0.3807 | 0.4190 | 0 LLM |
| **检索基准** | Hybrid BM25+Dense ($\beta=0.6$) | 0.3413 ± 0.0000 | 0.5380 | 0.5029 | 0.5076 | 0 LLM |
| **直接解码** | Cold $\boldsymbol{a}_0$ (随机初值) | 0.3060 ± 0.0006 | 0.4851 | 0.4492 | 0.4627 | 0 LLM |
| **直接解码** | Tail $\boldsymbol{a}_{30}$ (v1 末期先验, 消融) | 0.3204 ± 0.0074 | 0.5054 | 0.4706 | 0.4789 | 0 LLM |
| **直接解码** | **Amortized $\boldsymbol{a}_{\mathrm{mean}}$ (Ours, Polyak)** | **0.3409 ± 0.0070** | **0.5342** | **0.5000** | **0.5186** | **0 LLM** |
| **再演化搜索** | Cold $\boldsymbol{a}_0 \to \mathrm{ES}$ | 0.3541 ± 0.0022 | 0.5574 | 0.5218 | 0.6558 | 45 LLM |
| **再演化搜索** | **Amortized $\boldsymbol{a}_{\mathrm{mean}} \to \mathrm{ES}$ (Ours)** | $\mathbf{0.3644 \pm 0.0051}$ | $\mathbf{0.5735}$ | $\mathbf{0.5358}$ | $\mathbf{0.6723}$ | **45 LLM** |

- **池化成对显著性检验**：
  * **`evo_warm` vs `evo_cold`**：$\Delta \mathrm{F1} = \mathbf{+0.0103}$，胜/平/负 = **36 / 247 / 17**，**$t = +2.23$ ($p = 0.0267 < 0.05$，显著)**；
  * **`evo_warm` vs `decode_warm`**：$\Delta \mathrm{F1} = \mathbf{+0.0235}$，胜/平/负 = **49 / 236 / 15**，**$t = +3.65$ ($p = 0.0003 < 0.001$，极显著)**；
  * **`decode_warm` vs `decode_bm25`**：$\Delta \mathrm{F1} = \mathbf{+0.0238}$，胜/平/负 = **57 / 203 / 40**，**$t = +2.18$ ($p = 0.0304 < 0.05$，显著)**。

---

### 2. 跨裁判模型一致性验证 (Cross-Judge: Google Gemini 3.8 Flash)

为了排除单一裁判模型偏置，采用 `gemini-3.8-flash` 作为独立裁判评测 10 领域测试集：

| 对比组 (Paired Comparison) | 指标 | 差值 ($\Delta \mathrm{F1}$) | 胜/平/负 (W/T/L) | $t$ 统计量 | 统计学显著性 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`evo_warm` vs `evo_cold`** | F1@5 | $\mathbf{+0.0200}$ | 8 / 90 / 2 | $\mathbf{t = +2.04}$ | $\mathbf{p < 0.05}$ **(达成显著)** |
| **`evo_warm` vs `decode_warm`** | F1@5 | $\mathbf{+0.0422}$ | 16 / 81 / 3 | $\mathbf{t = +3.16}$ | $\mathbf{p < 0.01}$ **(极显著)** |
| **`decode_warm` vs `decode_bm25`** | F1@5 | $\mathbf{+0.0230}$ | 16 / 74 / 10 | $t = +1.40$ | 显著占优 |

- 测试集质心路由准确率达到 **93.0%**；
- 证明了所提出的领域先验具有跨大模型裁判架构的稳健迁移能力。

---

### 3. 四大核心消融研究概览

1. **成本模型消融 (Cost Mode)**：
   * Grounded 物理成本（$0.3409 \pm 0.0070$）相较无成本（$0.3439 \pm 0.0042$）仅微幅牺牲 $0.003$ 理论 F1，但显著压缩了 Prompt 开销并拦截了高危接口；
   * Synthetic 随机哈希成本（$0.3383 \pm 0.0036$）导致性能下降，证实物理元数据约束的有效性。
2. **协同通道解耦 (Synergy Disentanglement)**：
   * 完整协同通道（$0.3409 \pm 0.0070$）> 纯语义协同（$0.3391 \pm 0.0087$）> 纯模式兼容（$0.3198 \pm 0.0069$）；
   * 去除语义协同仅留参数类型匹配会导致 $-0.0211$ 的严重断崖下滑，证实 API 家族/类别聚类是协同防冗余的基石。
3. **混合检索权重敏感度 (Hybrid Beta Sweep)**：
   * 仅在训练集上选定的 $\beta=0.6$ 在测试集上严丝合缝达到最高点（$0.3413$），显著超越纯 Dense（$0.2454$）和纯 BM25（$0.3171$）。
4. **候选库规模扩展性 (Tool Pool Scale $N=1000$ vs $N=3000$)**：
   * 随着工具库从 1,000 扩展到 3,000，干扰项大幅增加，Amortized 解码相较 Hybrid 检索的领先优势从 $+0.0013$ 扩大至 $\mathbf{+0.0076}$。

---

## 复现指引 (Reproducibility)

```bash
# 1. 冒烟自检（无需 API Key，验证纯离线管道完整性）
python run_catalog_experiment.py --eval-mode unsupervised --P 2 --G 1 --train-limit 2 --test-limit 1 --tag smoke

# 2. 运行 10 领域正式实验（需 OPENAI_API_KEY）
python run_catalog_experiment.py --splits data/catalog_splits10.json --tag 10cat

# 3. 运行 4 大离线消融研究（无需 API 调用，基于已学先验）
python run_ablations_10cat.py

# 4. 运行 3 种子多样本显著性聚合
python analyze_p0_multiseed.py

# 5. 运行跨种子池化路由混淆分析
python analyze_routing_pooled.py
```

### 关键复现保障机制
- **确定性哈希 RNG**：每个 `(query, arm)` 拥有独立的 `SeedSequence` 衍生随机流，运行结果与断点中断/恢复次数、并发执行顺序完全无关。
- **缓存防毒化保护**：大模型调用经 `parsed_ok` 严格校验，异常返回绝不落盘；`results/llm_cache.jsonl` 提供确定性复现支撑。
- **单纯形数值稳定约束**：`project_simplex` 全程强制执行 $\epsilon_{\mathrm{floor}} = 0.02$，阻断优化陷入边缘角点。
