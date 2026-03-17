# EEG Ablation Study: ARC-only vs ARC+Lab

**课题**: EEG 情感识别 — 动态图神经网络 vs 静态图结构

**核心假设**: 接入科研社区 (SciEvalBook Lab) 的沉淀知识（高分 idea + 多模型讨论），能产出比纯自主生成更高质量的研究假设和实验设计。

## 实验设计

### 消融对比

|  | A 组 Baseline (无社区) | B 组 Treatment (有社区) |
|--|---|---|
| **无文献** | ✅ 已完成 | ✅ 已完成 |

### 当前进度

- [x] 无文献 + 无社区 (A): 假设已生成（3 个假设）
- [x] 无文献 + 有社区 (B): 假设已生成（4 个假设，含社区提炼）
- [x] LLM Judge 对比: B 组胜出（综合 9 vs 6）
- [ ] A 组实验执行 (Stage 9-22)
- [ ] B 组实验执行 (Stage 9-22)

### Judge 评分结果 (Claude Sonnet)

| 维度 | A组 (无社区) | B组 (有社区) |
|------|:-:|:-:|
| 科学创新性 | 6 | **8** |
| 方法具体性 | 7 | **9** |
| 神经科学合理性 | 7 | **9** |
| 可行性(6GB) | **8** | 7 |
| 实验设计清晰度 | 8 | **9** |
| 整体学术价值 | 6 | **9** |

---

## 项目结构

```
AutoResearchClaw-EEG-Ablation/
├── researchclaw/                    # ARC 核心包
│   ├── community/                   # 🆕 社区知识集成模块
│   │   ├── seb_client.py            #   SciEvalBook HTTP 客户端 (stdlib only)
│   │   ├── bridge.py                #   社区桥接: 获取数据 → 提取 idea → 知识卡片
│   │   └── refiner.py               #   Idea 提炼循环 (anchor → proposal → review → revise)
│   ├── pipeline/
│   │   ├── executor.py              #   🔧 Stage 7/8 注入社区知识
│   │   ├── runner.py                #   Pipeline 编排
│   │   └── stages.py                #   22-stage 定义
│   ├── config.py                    #   🔧 新增 CommunityConfig
│   ├── llm/                         #   LLM 客户端
│   ├── experiment/                  #   Sandbox 实验执行
│   └── ...
├── config_eeg_baseline.yaml         # A 组配置 (community.enabled: false)
├── config_eeg_treatment.yaml        # B 组配置 (community.enabled: true)
├── run_ablation.py                  # 消融实验编排脚本
└── artifacts/eeg-ablation/          # 实验产物
    └── no-lit/                      # 无文献变体
        ├── baseline/                # A 组
        │   ├── stage-07/synthesis.md
        │   ├── stage-08/hypotheses.md        # ← A 组假设
        │   └── stage-08/perspectives/        # 三视角辩论记录
        └── treatment/               # B 组
            ├── community/                     # 社区数据快照
            │   ├── community_snapshot.json    # 原始 API 数据
            │   ├── community_cards.md         # 知识卡片
            │   └── community_ideas.md         # 提取的 idea 索引
            ├── community-refine/              # 提炼过程全记录
            │   ├── problem-anchor.md          # 问题锚点（冻结）
            │   ├── round-0-proposal.md        # 初始提案
            │   ├── round-1-review.md          # LLM Review (5维评分)
            │   ├── FINAL_PROPOSAL.md          # 最终提案
            │   ├── REFINEMENT_LOG.md          # 提炼日志
            │   └── score-history.md           # 分数演变
            ├── stage-07/synthesis.md
            └── stage-08/hypotheses.md         # ← B 组假设
```

---

## 前提要求

### 硬件

| 资源 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | GTX 1660 Ti 6GB | RTX 3060+ |
| 显存 | 6 GB | 8 GB+ |
| 内存 | 16 GB | 32 GB |
| 存储 | 40 GB（数据 + 模型 + 产物） | 60 GB |
| 训练时间 | ~2-4h per group | ~1-2h (更好 GPU) |

### 软件依赖

```bash
# Python 3.10+
pip install torch torchvision torch_geometric
pip install mne scikit-learn pandas numpy scipy matplotlib
pip install openai pyyaml requests h5py
pip install datasets huggingface_hub   # DREAMER 数据集自动下载
pip install -e .  # 安装 AutoResearchClaw 包本身
```

### 数据集

实验使用 HuggingFace 上的 DREAMER 数据集（**无需申请，自动下载**）：

| 数据集 | 被试 | 通道 | 情感标签 | 样本数 | 获取方式 |
|--------|------|------|---------|--------|---------|
| **DREAMERA** | 23 | 14 (Emotiv EPOC) | Binary Arousal | 170,246 | `hf_hub_download("monster-monash/DREAMERA")` |
| **DREAMERV** | 23 | 14 (Emotiv EPOC) | Binary Valence | 170,246 | `hf_hub_download("monster-monash/DREAMERV")` |

每个样本为 256×14 的时间序列（2秒窗口，128Hz采样率）。14通道对应 Emotiv EPOC 电极：AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4。

```python
# 自动下载（首次约 500MB，之后使用缓存）
from huggingface_hub import hf_hub_download
import numpy as np

X_path = hf_hub_download(repo_id="monster-monash/DREAMERA", filename="DREAMERA_X.npy", repo_type="dataset")
y_path = hf_hub_download(repo_id="monster-monash/DREAMERA", filename="DREAMERA_Y.npy", repo_type="dataset")
X = np.load(X_path)  # (170246, 256, 14)
y = np.load(y_path)  # (170246,) binary
```

> **集群离线？** 先在可联网机器下载：`huggingface-cli download monster-monash/DREAMERA --repo-type dataset --local-dir ./data/dreamer`，然后 scp 到集群。

### LLM API

Pipeline 的 Stage 9-22 仍需 LLM 调用（实验设计、代码生成、结果分析、论文撰写）：

```yaml
# config 中的 LLM 配置
llm:
  provider: "openai-compatible"
  base_url: "https://openai-api.shenmishajing.workers.dev/v1"  # 需集群能访问
  api_key: "..."
  primary_model: "gpt-4o"
```

**如果集群无法访问外部 API**，需要：
1. 设置代理: `export HTTPS_PROXY=http://your-proxy:port`
2. 或换成集群内部的 LLM endpoint，修改 config 中 `base_url`
3. 或先在可联网机器跑完 Stage 9（实验设计），只把代码执行部分放集群

### SciEvalBook 社区 API（仅 Treatment 组的 Stage 7 需要）

```
GET http://localhost:8000/api/community-knowledge/{challenge_id}
```

> **集群不需要访问此 API** — 社区数据已在 `artifacts/.../community/` 中快照保存。Stage 9+ 不再调用社区。

---

## 集群操作步骤

### 1. Clone

```bash
git clone -b eeg-ablation https://github.com/Zhouzone/AutoResearchClaw-EEG-Ablation.git
cd AutoResearchClaw-EEG-Ablation
pip install -e .
```

### 2. 验证环境

```bash
# GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 依赖
python3 -c "import torch_geometric, mne, sklearn; print('deps OK')"

# LLM API 可达性
curl -s https://openai-api.shenmishajing.workers.dev/v1/models | head -5
```

### 3. 配置调整

```bash
# 如果集群 python 路径不同
sed -i 's|python_path: "python3"|python_path: "/your/path/python"|' config_eeg_*.yaml

# 如果需要代理
export HTTPS_PROXY=http://proxy:port

# 如果 LLM API 地址不同
sed -i 's|base_url:.*|base_url: "https://your-api/v1"|' config_eeg_*.yaml
```

### 4. 跑 A 组（无社区, Baseline）

```bash
python3 -c "
from researchclaw.config import RCConfig
from researchclaw.adapters import AdapterBundle
from researchclaw.pipeline.runner import execute_pipeline
from researchclaw.pipeline.stages import Stage
from pathlib import Path

config = RCConfig.load('config_eeg_baseline.yaml')
execute_pipeline(
    run_dir=Path('artifacts/eeg-ablation/no-lit/baseline'),
    run_id='rc-baseline-exp',
    config=config,
    adapters=AdapterBundle(),
    from_stage=Stage.EXPERIMENT_DESIGN,
    auto_approve_gates=True,
    skip_noncritical=True,
)
print('A 组完成')
"
```

### 5. 跑 B 组（有社区, Treatment）

```bash
python3 -c "
from researchclaw.config import RCConfig
from researchclaw.adapters import AdapterBundle
from researchclaw.pipeline.runner import execute_pipeline
from researchclaw.pipeline.stages import Stage
from pathlib import Path

config = RCConfig.load('config_eeg_treatment.yaml')
execute_pipeline(
    run_dir=Path('artifacts/eeg-ablation/no-lit/treatment'),
    run_id='rc-treatment-exp',
    config=config,
    adapters=AdapterBundle(),
    from_stage=Stage.EXPERIMENT_DESIGN,
    auto_approve_gates=True,
    skip_noncritical=True,
)
print('B 组完成')
"
```

### 6. 如果 B 组跑不通

**常见问题及解决**:

| 问题 | 解决方案 |
|------|---------|
| LLM API 不通 | 设代理或换 endpoint |
| CUDA OOM | 减小 batch: 改 Stage 10 生成的代码中 `batch_size` (16→8) |
| 数据集未下载 | DREAMER 自动从 HuggingFace 下载；集群离线时先 `huggingface-cli download` 再 scp |
| torch_geometric 缺失 | `pip install torch_geometric` (需匹配 CUDA 版本) |
| 实验代码 bug | ARC 的 Stage 13 (Iterative Refine) 会自动修复，最多 5 轮 |

**回社区获取帮助**（在能访问 SciEvalBook 的机器上）:

```bash
# 获取更多 neuroscience 领域挑战的社区知识
curl http://localhost:8000/api/challenges?discipline=neuroscience

# 查看某个相关挑战的模型回答
curl http://localhost:8000/api/community-knowledge/<challenge_id>

# 把新的 challenge_id 加入 config_eeg_treatment.yaml → community.challenge_ids
```

---

## 公平性保证

| 控制项 | 措施 |
|--------|------|
| 同一 LLM | 两组配置相同 `primary_model: gpt-4o` |
| 同一假设格式 | 都经过 3 视角辩论 (innovator/pragmatist/contrarian) |
| 同一硬件 | 顺序执行，同一 GPU |
| 同一 topic | 相同 `research.topic` 字符串 |
| 同一实验约束 | 相同 sandbox 配置 (6GB, 1800s time budget) |
| 唯一变量 | B 组多了社区知识获取 + idea 提炼 |

---

## 预期产出

每组跑完后会产出:

```
stage-09/  exp_plan.yaml          # 实验计划
stage-10/  experiment/             # 生成的实验代码
stage-12/  runs/                   # 实验运行记录
stage-13/  experiment_final/       # 迭代修复后的最终代码
stage-14/  experiment_summary.json # 结果统计 (accuracy, F1)
stage-16/  paper_outline.md        # 论文大纲
stage-17/  paper_draft.tex         # 论文草稿
stage-20/  quality_gate.json       # 质量评分
```

最终对比: A 组 vs B 组的 `experiment_summary.json` 中的 accuracy、论文 quality gate score。
