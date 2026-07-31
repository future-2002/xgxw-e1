# E1 实验：真实求助语料上的咨询质量对比

> 论文主实验。回答核心问题：**三极议会多智能体框架在真实心理求助场景下，咨询质量是否显著优于单模型基线，并接近人类咨询水平？**

## 为什么是这个实验

对标 WiseMind / EFT-CoT / AutoCBT 等竞品论文的疗效证据范式：**真实求助语料 + 心理咨询专业维度评估 + 人类回答对照**。区别于此前只证明"系统安全、跑得快"的实验，本实验直接证明"系统帮到了人"。

## 数据

| 语料 | 规模 | 用途 | 来源 |
|---|---|---|---|
| PsyQA | 22341 条真实心理求助帖（含高赞人类回答） | 单轮质量评测主语料 | `lsy641/PsyQA` |
| CPsyCounD | 3134 场多轮咨询对话 | 多轮上下文评测 / E2 虚拟来访者 | `CAS-SIAT-XinHai/CPsyCoun` |

原始数据放 `../e1_corpus/raw/`（不入库）。下载（HF 镜像）：
```bash
cd ../e1_corpus/raw
curl -sL -o PsyQA_full.json  "https://hf-mirror.com/datasets/lsy641/PsyQA/resolve/main/PsyQA_full.json"
curl -sL -o CPsyCounD.json   "https://hf-mirror.com/datasets/CAS-SIAT-XinHai/CPsyCoun/resolve/main/CPsyCounD.json"
```

## 七个对照条件

| 条件 | 后端 | 说明 |
|---|---|---|
| `gvc_full` | 系统 API | 三极议会完整版 |
| `ablation_no_ca` | 系统 API | 消融：去 CA 验证器 |
| `ablation_no_skg` | 系统 API | 消融：去 SKG 知识图谱 |
| `ablation_no_convergence` | 系统 API | 消融：去 EVINCE 收敛判据（固定轮次） |
| `single_llm` | 云端直连 | 单模型 + 朴素咨询师提示词（弱基线） |
| `cbt_prompt` | 云端直连 | 单模型 + 结构化 CBT 提示词（强基线，对齐 AutoCBT 单智能体） |
| `human` | 语料自带 | PsyQA 高赞人类回答（上界参照） |

消融开关已接入 `config/experiment.py` + `agents/parliament/engine.py`，通过 `PARLIAMENT_EXPERIMENT` 环境变量切换，编排脚本自动重建容器。

## 评估

- **AutoCBT 六维**（0-7 分/维）：共情 / 识别 / 反思 / 策略 / 鼓励 / 相关性，对齐 AutoCBT 论文 Table 2
- **安全检查**（布尔）：诊断性断言 / 有害建议 / 评判指责
- **盲评**：评审模型只看 `求助内容 + 回复`，不知晓来源条件；默认评审模型 `qwen-max`（与生成模型 `qwen-plus` 不同档，降低自评偏置）
- **统计**：Welch t 检验 + Cohen's d + Holm 多重比较校正 + 95% CI

## 运行（Docker 环境，全流程宿主机一键）

```bash
# 前置：docker compose 基础设施已起；docker/.env 中 LLM_API_KEY 为有效 DashScope Key

# 冒烟测试（每条件 3 条，验证链路）
./scripts/run_e1_experiment.sh --smoke

# 完整实验（200 样本 × 7 条件）
./scripts/run_e1_experiment.sh

# 只跑部分条件
./scripts/run_e1_experiment.sh --only single_llm,cbt_prompt,human
```

脚本流程：切换消融配置并重建 `ai-service` → 容器内 `run_generation.py` 生成 → `run_judge.py` 盲评 → `analyze_results.py` 统计 → 恢复 baseline。

## 分步执行（容器内调试）

```bash
docker compose exec ai-service python /app/experiments/e1_counseling_quality/sample_corpus.py --psyqa-n 200
docker compose exec ai-service python /app/experiments/e1_counseling_quality/run_generation.py --condition gvc_full
docker compose exec ai-service python /app/experiments/e1_counseling_quality/run_judge.py
docker compose exec ai-service python /app/experiments/e1_counseling_quality/analyze_results.py
```

所有步骤支持**断点续跑**（按样本 id 去重），中断后重跑同一命令即可补齐。

## 产物

```
../e1_results/generations/<condition>.jsonl   # 各条件生成回复
../e1_results/judgments/<condition>.jsonl     # 各条件盲评分数
../e1_results/E1_analysis_<ts>.json           # 完整统计结果
../e1_results/E1_report_<ts>.md               # Markdown 实验报告（可直接进论文）
```

## 文件

| 文件 | 职责 |
|---|---|
| `sample_corpus.py` | 语料分层采样 |
| `conditions.py` | 七条件定义 + 三种后端调用（GVC/直连/人类） |
| `run_generation.py` | 批量生成回复（断点续跑） |
| `run_judge.py` | LLM-as-Judge 盲评（断点续跑） |
| `analyze_results.py` | 统计分析 + 报告生成 |
