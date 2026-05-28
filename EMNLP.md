# OpenPRM: Process-Level Verification for Open-Domain Multimodal Reasoning

## 完整实施方案（修订版）

---

## 一、核心定位

### 1.1 一句话总结

**现有多模态 PRM 全部依赖 Monte Carlo sampling，只能在有标准答案的数学推理上工作。我们提出第一个面向开放域多模态任务的 process-level verification 方法，通过反向负样本构造策略摆脱对标准答案的依赖。**

### 1.2 要解决的核心问题

现有多模态 PRM（VisualPRM、VRPRM、GM-PRM 等）的数据构造全部依赖 Monte Carlo (MC) sampling：对每个推理步骤，采样多次 completion，根据最终答案是否正确来估计该步骤的 expected accuracy。

这个方法有一个致命限制：**必须有唯一正确答案来判断 completion 是否正确**。

因此，所有现有多模态 PRM 只在以下场景有效：数学推理（MathVista、MathVerse、MathVision）、逻辑推理（LogicVista）、学科知识问答（MMMU 选择题部分）。

而现实中大量多模态任务是**开放域**的，没有唯一正确答案：图片描述与详细分析（MM-Vet）、开放式视觉问答（LLaVA-Bench）、图表解读与总结（ChartQA 开放题）、常识推理与场景理解。

这些任务的 process supervision 完全空白。

### 1.3 三个创新点

**创新点 1：首个面向开放域多模态任务的 Process-Level Verification 方法**

- 将 process supervision 从数学推理扩展到开放域视觉问答
- 在 MM-Vet、LLaVA-Bench 等开放域 benchmark 上展示 process-level BoN 的效果

**创新点 2：不依赖 MC Sampling 的 Process Supervision 数据构造 Pipeline**

- 借鉴 MR. Judge 的反向负样本生成策略，在推理链的中间步骤注入视觉幻觉/推理错误
- 不需要标准答案，只需要 SFT 数据中的参考 response 作为正样本
- 相比 MC sampling（VisualPRM 需要对每步采样 16 次 completion），数据构造成本大幅降低

**创新点 3：Visual Grounding-Aware Step Verification**

- 在开放域场景中，推理步骤的正确性高度依赖是否正确引用了图像内容
- 我们在 step-level 验证中显式检查每步是否有视觉幻觉（编造不存在的物体/属性/关系）
- 这在数学推理场景中不突出（图像信息通常在第一步就被完整读取），但在开放域场景中至关重要

### 1.4 与现有工作的关系

|工作|我们借鉴了什么|我们与它的区别|
|---|---|---|
|VisualPRM|评估框架（VisualProcessBench）、PRM 建模方式（value-based）|它只做数学推理，依赖 MC sampling；我们做开放域，用反向负样本|
|VRPRM|SFT+RL 两阶段训练、VERL+GRPO 框架|它的训练数据仍来自 VisualPRM400K（数学）；我们构造开放域数据|
|MR. Judge|反向负样本生成策略、多选题+推理范式|它只做 outcome-level 选择；我们做 step-level verification|
|EVPV|Visual premise verification 的思想|它是纯文本的 post-hoc 校准；我们训练端到端的多模态 PRM|
|LLaVA-Critic|开源多模态 judge 作为 baseline|它只做 pointwise/pairwise scoring；不做 process-level|

---

## 二、技术方案

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      OpenPRM 技术架构                             │
│                                                                 │
│  阶段 1: 开放域 Process Supervision 数据构造                       │
│  ├── 种子数据: MM-Vet / LLaVA-Bench / 开放域 VQA 的 SFT 数据      │
│  ├── 正样本: 用 Qwen2.5-VL 生成高质量 reasoning traces            │
│  ├── 负样本: MR.Judge 式反向错误注入（step-level）                  │
│  │   ├── 视觉幻觉注入: 编造/误读图像内容                           │
│  │   ├── 推理跳跃注入: 跳过关键步骤直接得出结论                     │
│  │   └── 描述不一致注入: 前后描述矛盾                              │
│  └── Step-level 标注: GPT-4o 自动验证每步正确性                    │
│                                                                 │
│  阶段 2: OpenPRM 训练                                            │
│  ├── SFT Warm-up: step-level 验证数据微调                         │
│  └── RL (optional): VERL + GRPO                                 │
│                                                                 │
│  阶段 3: 评估                                                    │
│  ├── 开放域 BoN: MM-Vet, LLaVA-Bench                            │
│  ├── 数学推理 BoN: MathVista (跨域泛化)                           │
│  ├── Step verification: VisualProcessBench + 开放域扩展           │
│  └── Cross-domain 分析: 开放域训练是否有助于数学推理？               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 数据构造 Pipeline（核心创新）

#### 为什么 MC Sampling 在开放域不可用？

MC sampling 的流程是：给定前 i 步，采样后续 completion，判断最终答案是否正确。 在数学题中，"正确"有明确定义（答案 = ground truth）。 在开放域中，"这张图里有什么？"没有唯一正确答案，无法判断 completion 是否"正确"。

#### 我们的替代方案：Reverse Step-Level Error Injection

**Step 1: 生成正样本推理链**

从开放域 SFT 数据集出发（如 LLaVA-Instruct、ShareGPT4V）， 用 Qwen2.5-VL-32B 为每个 (image, question) 对生成详细的 step-by-step reasoning trace。

格式：

```
Image: [image]
Question: Describe the activities in this image in detail.

Step 1: The image shows an outdoor park setting with green grass and trees.
Step 2: In the foreground, there are two children playing with a red ball.
Step 3: Behind them, an adult is sitting on a bench reading a book.
Step 4: The weather appears sunny with clear blue skies.
Step 5: Overall, the image depicts a peaceful afternoon scene in a park.
```

**Step 2: 反向错误注入（借鉴 MR. Judge）**

针对开放域场景设计三类错误注入：

|错误类型|描述|示例|
|---|---|---|
|Visual Hallucination|编造图中不存在的物体/属性|"Step 2: 前景有两个小孩在玩**蓝色**球" (实际是红色)|
|Reasoning Gap|跳过关键观察直接下结论|跳过对人物动作的描述，直接说"这是一个运动场景"|
|Inconsistency|前后描述矛盾|Step 1 说"室内"，Step 3 说"阳光明媚的户外"|

用 MLLM 自动生成包含这些错误的负样本推理链：

```
Step 1: The image shows an outdoor park setting with green grass and trees. [correct]
Step 2: In the foreground, there are two children playing with a blue ball. [HALLUCINATION: ball is red]
Step 3: A dog is running beside the children. [HALLUCINATION: no dog in image]
Step 4: The weather appears sunny with clear blue skies. [correct, but based on flawed context]
```

**Step 3: Step-Level 自动标注**

用 GPT-4o 对每条推理链逐步验证：

- 输入：image + question + reasoning step + previous steps
- 输出：correct/incorrect + error_type (如果 incorrect)

#### 数据规模估计

|数据来源|种子样本数|生成正样本|生成负样本|最终 step-level 标注|
|---|---|---|---|---|
|LLaVA-Instruct 子集|~3000|~3000 条推理链|~6000 条含错推理链|~45000 步|
|ShareGPT4V 子集|~2000|~2000|~4000|~30000 步|
|**合计**|**~5000**|**~5000**|**~10000**|**~75000 步**|

### 2.3 模型训练

**Base Model**: Qwen2.5-VL-7B-Instruct

**SFT 阶段**：

- 用 step-level 标注数据微调
- 输入：image + question + reasoning trace to verify
- 输出：每步的 correct/incorrect 判断 + 简短理由
- 参考 VRPRM 的格式（<think> 推理 + \boxed{0/1} 判断）

**RL 阶段（可选，作为增益）**：

- 参考 VRPRM 的 GRPO 训练
- Reward = 0.9 × R_process + 0.1 × R_format
- 使用 VERL 框架

---

## 三、Baseline 与数据集

### 3.1 Baseline 模型

|#|Model|来源|你需要做什么|
|---|---|---|---|
|B1|Qwen2.5-VL-7B (zero-shot prompt)|HuggingFace 开源|直接跑推理|
|B2|LLaVA-Critic-7B|HuggingFace 开源|直接跑推理|
|B3|VisualPRM-8B|HuggingFace 开源|直接跑推理（验证数学 PRM 在开放域的表现）|
|B4|VRPRM-7B|如开源则直接用，否则引用数据|跑推理或引用|
|B5|MR. Judge-7B|未开源，引用论文数据|引用|
|**Ours**|**OpenPRM-7B**|**你训练**|**SFT (+RL)**|

### 3.2 关键资源链接

**开源模型**:

- Qwen2.5-VL-7B: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- LLaVA-Critic-7B: https://huggingface.co/lmms-lab/llava-critic-7b
- VisualPRM-8B: https://huggingface.co/OpenGVLab/VisualPRM-8B (InternVL 团队)

**开源数据**:

- VisualPRM400K: https://huggingface.co/datasets/OpenGVLab/VisualPRM400K
- VisualProcessBench: https://huggingface.co/datasets/OpenGVLab/VisualProcessBench
- LLaVA-Critic-113K: https://huggingface.co/datasets/lmms-lab/llava-critic-113k
- LLaVA-Instruct-150K: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K

**评估 Benchmark**:

- MM-Vet: https://github.com/yuweihao/MM-Vet
- LLaVA-Bench: 内含在 LLaVA-NeXT 中
- VL-RewardBench: https://github.com/vl-rewardbench/VL_RewardBench
- MathVista: https://huggingface.co/datasets/AI4Math/MathVista

**训练框架**:

- VERL: https://github.com/volcengine/verl
- ms-swift (SFT): https://github.com/modelscope/ms-swift
- LLaMA-Factory (SFT 备选): https://github.com/hiyouga/LLaMA-Factory

### 3.3 论文中需要展示的实验

**主实验 1：开放域 BoN 评估（Table 1）— 你的主战场**

|Critic Model|MM-Vet (BoN=8)|LLaVA-Bench (BoN=8)|
|:--|:-:|:-:|
|No critic (Pass@1)|baseline|baseline|
|Self-Consistency|你跑|你跑|
|LLaVA-Critic-7B|你跑|你跑|
|VisualPRM-8B|你跑（预期表现不佳）|你跑|
|**OpenPRM-7B (ours)**|**你跑**|**你跑**|

VisualPRM 在开放域表现不佳是预期中的——它是在数学数据上训练的。这恰好验证你的论点。

**主实验 2：跨域泛化（Table 2）— 展示你的方法也能在数学上工作**

|Critic Model|MathVista (BoN=8)|MathVerse-VO (BoN=8)|
|:--|:-:|:-:|
|VisualPRM-8B|引用/你跑|引用/你跑|
|**OpenPRM-7B**|你跑|你跑|

不需要超过 VisualPRM，只要不差太多就行。重点是展示开放域训练的模型也有一定的跨域迁移能力。

**实验 3：VisualProcessBench 上的 step verification（Table 3）**

直接在现有 benchmark 上评估你的模型，与 VisualPRM 对比。

**实验 4：开放域 Step Verification 案例分析（Table 4 / Figure）**

展示你的模型在开放域场景中如何逐步验证推理链，定位视觉幻觉。 这是定性分析，用 case study 的形式展示。

**消融实验（Table 5）**

|配置|MM-Vet BoN|说明|
|:--|:-:|:--|
|OpenPRM (full)|best|完整模型|
|− Visual hallucination injection|下降|去掉视觉幻觉注入|
|− Step-level error injection (only outcome-level)|下降|只用 response-level 负样本|
|− SFT warm-up (RL only)|下降|跳过 SFT|
|− RL (SFT only)|略低|只用 SFT|

---

## 四、两个月时间规划

### 周计划

**第 1 周（3/30-4/5）：环境搭建 + 可行性验证**

- [ ] 搭建环境：VERL, vLLM, transformers, ms-swift
- [ ] 下载 Qwen2.5-VL-7B, LLaVA-Critic-7B, VisualPRM-8B
- [ ] 在 MM-Vet 上跑 Pass@1 和 BoN (用 LLaVA-Critic 和 VisualPRM 做 critic)
- [ ] **关键验证：确认 VisualPRM 在 MM-Vet 上表现不佳**（验证你的论文假设）
- 里程碑：baseline 数字就位，假设验证通过

**第 2 周（4/6-4/12）：数据构造 Pipeline**

- [ ] 从 LLaVA-Instruct / ShareGPT4V 中选取种子样本
- [ ] 用 Qwen2.5-VL-32B 生成 reasoning traces
- [ ] 实现反向错误注入 pipeline（视觉幻觉/推理跳跃/不一致）
- [ ] 用 GPT-4o 做 step-level 自动标注
- [ ] 数据质量检查
- 里程碑：~5K 正样本 + ~10K 负样本就位

**第 3 周（4/13-4/19）：SFT 训练**

- [ ] 格式化训练数据（参考 VRPRM 的 CoT-PRM 格式）
- [ ] 用 ms-swift 对 Qwen2.5-VL-7B 做 SFT
- [ ] 在 MM-Vet BoN 上初步评估
- 里程碑：SFT 模型完成，有初步 BoN 数字

**第 4 周（4/20-4/26）：RL 训练（可选）+ 补充实验**

- [ ] 如果 SFT 效果好：尝试 GRPO RL 训练
- [ ] 如果 SFT 效果一般：调整数据构造策略，重新训练
- [ ] 跑 VisualProcessBench 评估
- 里程碑：最终模型确定

**第 5 周（4/27-5/3）：全面实验**

- [ ] 跑所有 baseline 在所有 benchmark 上的数字
- [ ] 跑跨域泛化实验（MathVista）
- [ ] 收集 case study
- 里程碑：所有实验数字就位

**第 6 周（5/4-5/10）：消融实验 + 分析**

- [ ] 消融实验
- [ ] 失败案例分析
- [ ] 整理结果表格和图表
- 里程碑：实验部分完整

**第 7 周（5/11-5/17）：论文撰写**

- [ ] Introduction + Related Work
- [ ] Method
- [ ] Experiments
- [ ] 画框架图
- 里程碑：论文初稿

**第 8-9 周（5/18-截稿）：修改 + 提交**

- [ ] 导师 review + 修改
- [ ] 补充实验
- [ ] 最终提交
- 里程碑：投稿完成

### 风险管理

|风险|应对|
|---|---|
|VisualPRM 在 MM-Vet 上表现不错（假设不成立）|改为比较"在开放域数据上训练 vs 在数学数据上训练"的 PRM，强调数据分布匹配的重要性|
|GPT-4o 标注成本过高|改用 DeepSeek-V3 API 或 Qwen2.5-VL-72B 自标注|
|RL 训练不稳定|只用 SFT 版本，RL 作为消融实验的一部分|
|开放域 BoN 提升不明显|强调 step verification 的定性贡献（case study），加大 qualitative analysis 的篇幅|

### 最低可交付标准

即使时间紧张，只要完成以下四件事就可以投稿：

1. 开放域 process supervision 数据构造 pipeline（方法创新）
2. SFT 训练的 OpenPRM 模型
3. MM-Vet / LLaVA-Bench 上的 BoN 对比实验
4. 与 VisualPRM 的 cross-domain 对比分析

---

## 五、论文结构大纲

### 暂定标题

**"Beyond Math: Process Reward Modeling for Open-Domain Multimodal Reasoning"**

或

**"OpenPRM: Extending Process Supervision to Open-Domain Visual Question Answering without Monte Carlo Sampling"**

### 各节内容

**Abstract** (~200 words)

- 现有多模态 PRM 限制在数学推理
- 我们提出第一个开放域多模态 PRM
- 不依赖 MC sampling 的数据构造方法
- 在 MM-Vet/LLaVA-Bench 上的显著提升

**1. Introduction** (~1.5 pages)

- 多模态 PRM 的重要性
- MC sampling 的局限性 → 只能做数学
- 我们的方案 + 贡献列表

**2. Related Work** (~1 page)

- Process Reward Models (PRM800K → VisualPRM → VRPRM)
- MLLM-as-Judge (MR. Judge, LLaVA-Critic)
- Test-Time Scaling for MLLMs

**3. Method** (~2.5 pages)

- 3.1 Problem: 为什么 MC sampling 在开放域不可用
- 3.2 数据构造: Reverse Step-Level Error Injection
- 3.3 模型训练: SFT + RL
- 3.4 推理: Step-level scoring + BoN selection

**4. Experiments** (~3 pages)

- 4.1 开放域 BoN 评估 (主实验)
- 4.2 跨域泛化分析
- 4.3 Step Verification on VisualProcessBench
- 4.4 消融实验
- 4.5 Case Study

**5. Conclusion + Limitation** (~0.5 page)

---

## 六、关键参考论文

1. VisualPRM (Wang et al., 2025) - 多模态 PRM baseline, arXiv preprint
2. VRPRM (Chen et al., AAAI 2026) - CoT + RL 的多模态 PRM
3. MR. Judge (Pi et al., EMNLP 2025) - 反向负样本策略
4. LLaVA-Critic (Xiong et al., 2024) - 开源多模态 judge
5. URSA (Luo et al., NeurIPS 2025) - 多模态 PRM + PS-GRPO
6. Math-Shepherd (Wang et al., 2023) - MC sampling 方法论
7. PathFinder-PRM (Pala et al., 2025) - Error-type-aware PRM (纯文本)
8. EVPV (2026) - Visual premise verification
9. ThinkWithImages-PRMBench (2026) - 多模态推理错误 taxonomy
10. MMRB2 (Meta, 2026) - 多模态 reward model benchmark