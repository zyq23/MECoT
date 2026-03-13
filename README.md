# MECoT: Internalizing Multi-Expert Chain-of-Thought for Interpretable Mental Health Support


本项目提供 **MECoT** 框架及 **ChairChat** 模型的官方实现代码。针对心理健康干预中大语言模型存在的注意力漂移与干预策略单一等局限性，本研究引入多智能体协同机制，通过构建包含认知行为疗法（CBT）与人本主义疗法双视角的训练框架，完成内化思维链（CoT）的参数拟合。

## 核心架构特性

* **多流派策略融合**：有效整合人本主义的无条件接纳与 CBT 的认知引导，克服单一干预视角的局限。
* **内化思维链 (CoT)**：构建包含情绪诊断、多专家推演与策略融合的完整推演路径，指导模型执行严格的“共情接纳、苏格拉底式提问、行为引导”的连贯干预动作。
* **高效参数微调**：基于 ChatGLM2-6B 基座模型，采用 LoRA 技术覆盖全量线性层，在低显存消耗下最大化流派知识的拟合能力。

## 仓库目录结构

本仓库的代码逻辑与论文中提出的干预框架严格对齐，主要包含三大核心训练与决策模块：

* `cbt_finetune/`: 认知行为疗法（CBT）专家模型的独立训练与微调代码。
* `humanistic_finetune/`: 人本主义疗法专家模型的独立训练与微调代码。
* `chair_finetune/`: 主席智能体（Chair Agent）的多视角动态仲裁与综合决策代码，负责融合双流派回复并输出最终干预策略。
* `data_chair_enhance_cleaned/`: 包含构建三千余条多流派视角的心理健康对话数据集示例格式。

## 硬件与环境依赖

模型的训练过程在配备 NVIDIA RTX 3090 GPU 的硬件环境中进行验证。环境配置要求如下：

```bash
git clone https://github.com/zyq23/MECoT.git
cd MECoT
pip install -r requirements.txt

```

## 模型训练与推理流程

### 1. 专家模型独立训练

采用 `bfloat16` 半精度浮点格式并结合梯度检查点技术以规避梯度溢出风险。分别对两类专家模型进行参数高效微调：

```bash
# CBT专家模型训练
python cbt_finetune/train_cbt.py

# 人本主义专家模型训练
python humanistic_finetunel/train_humanistic.py

```

### 2. 主席智能体策略融合与训练

在专家模型训练完成后，启动综合决策模块，训练内化思维链逻辑，将诊断与推演机制内化至模型参数空间中：

```bash
python chair_finetunel/train_chair_final.py

```

### 3. 对话推理

在推理阶段，严格约束解码策略的超参数配置。设定极低的温度参数以执行确定性推理，并设定重复惩罚为 1.1，有效推进干预逻辑，防止模型陷入重复追问：

```bash
python chair_finetunel/inference_chair.py

```
