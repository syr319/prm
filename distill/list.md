## Week 1：数据准备 + 最小验证

### Day 1

- [ ]  环境搭建 + 下载所有模型和数据（你已经在做了）
- [ ]  解析GenPRM-MATH-Data，拆成单步数据
- [ ]  统计总步骤数、正确/错误比例

### Day 2（验证insight 1：困难步骤差距是否更大）

- [ ]  按teacher判断的确定性将步骤分3桶：简单/中等/困难
- [ ]  统计每个桶内teacher和hard_label的一致率
- [ ]  统计每个桶内的数据量分布
- [ ]  **判断**：困难桶一致率是否明显低于简单桶
    - ✅ 是（差距>10个点） → motivation成立，继续
    - ❌ 否 → 需要重新思考motivation角度

### Day 3（验证insight 2：错误类型能否可靠提取）

- [ ]  从数据中筛选出被判断为错误的步骤
- [ ]  设计错误类型提取方案（关键词匹配 + 备选GPT分类）
- [ ]  在500条上跑提取
- [ ]  人工检查100条，计算提取准确率
- [ ]  **判断**：准确率是否>80%
    - ✅ 是 → 创新点2可行，继续
    - ⚠️ 60-80% → 改进提取方法或简化错误分类体系（比如从7类减到3-4类）
    - ❌ <60% → 考虑弱化或替换这个创新点

### Day 4（构建soft score）

- [ ]  部署GenPRM-7B
- [ ]  对拆好的单步数据跑GenPRM推理获取soft score
- [ ]  或者直接从模型输出logits计算概率作为soft score
- [ ]  合并为完整蒸馏数据集

### Day 5-6（训练baseline）

- [ ]  训练纯CE baseline
- [ ]  训练CE+KL蒸馏 baseline
- [ ]  监控loss曲线

### Day 7（评测 + Week1总结）

- [ ]  ProcessBench评测
- [ ]  汇总三个验证结果，决定Week2的具体策略
---

## Week 2：加满所有创新模块

### Day 1

- [ ]  扩充蒸馏数据集到200K-300K（或全量PRM800K）
- [ ]  再次做数据质量检查
- [ ]  统计错误类型分布，确认各类别样本量是否均衡，考虑是否需要类别加权

### Day 2

- [ ]  实现完整DistillPRM模型（score_head + error_head + temperature参数）
- [ ]  实现难度计算函数：`difficulty = 1 - 2*|teacher_score - 0.5|`
- [ ]  实现自适应蒸馏损失：简单步骤用CE为主，困难步骤用KL为主

### Day 3

- [ ]  实现错误类型分类损失（cross entropy，加入类别权重处理不均衡）
- [ ]  实现校准损失（soft ECE或label smoothing + 可学习temperature）
- [ ]  组装总损失函数：`L_total = L_adaptive_score + λ2·L_error + λ3·L_cal`

### Day 4-5

- [ ]  训练完整DistillPRM模型
- [ ]  用wandb跟踪各分项loss的变化趋势
- [ ]  调优λ权重（建议先用λ2=0.3，λ3=0.2，观察各loss量级后微调）
- [ ]  如果资源允许，尝试两个student规模：1.5B和7B

### Day 6-7

- [ ]  ProcessBench评测（step acc + error localization）
- [ ]  PRMBench评测（如果可用）
- [ ]  MATH + Best-of-N评测（采样N=8/16/64/256）
- [ ]  ECE校准指标评测 + 绘制reliability diagram
- [ ]  错误类型分类F1评测
- [ ]  推理速度对比（DistillPRM vs GenPRM的tokens/sec或latency）
- [ ]  汇总Week2结果，确认完整模型是否比Week1的CE+KL进一步提升

---

## Week 3：消融实验 + 分析实验

### Day 1-2：消融实验（最核心）

需要训练以下模型变体：

- [ ]  **Full DistillPRM**（所有模块）
- [ ]  **- w/o 难度自适应**（所有步骤用固定0.5权重混合CE和KL）
- [ ]  **- w/o 错误类型头**（去掉error_head和L_error）
- [ ]  **- w/o 校准损失**（去掉L_cal和temperature）
- [ ]  **- w/o 所有蒸馏**（退化为纯CE baseline）

每个变体都在ProcessBench + MATH BoN + ECE上评测，填表：

|配置|ProcessBench|MATH BoN@16|ECE ↓|Error F1|
|---|---|---|---|---|
|Full|||||
|w/o adaptive|||||
|w/o error|||||
|w/o calibration|||||
|CE only|||||

### Day 3：协同效应分析

- [ ]  计算各组件单独提升之和 vs 全部一起的实际提升
- [ ]  如果实际提升 > 各组件之和 → 证明协同效应，写入论文
- [ ]  如果没有协同效应 → 也是有价值的发现，如实报告

### Day 4：难度分桶分析

- [ ]  按teacher_score将步骤分为3桶：简单（>0.8或<0.2）、中等、困难（0.3-0.7）
- [ ]  分别统计每个桶内的准确率提升
- [ ]  **预期结果**：困难桶提升最大，这直接验证了核心motivation
- [ ]  绘制柱状图

### Day 5：错误类型分析

- [ ]  统计各错误类型的分类准确率
- [ ]  分析哪类错误最难识别
- [ ]  展示2-3个case study：模型正确识别错误类型的例子

### Day 6：模型规模实验（如果时间允许）

- [ ]  Student = 1.5B / 7B 两个规模的对比
- [ ]  Teacher = GenPRM-7B固定
- [ ]  分析蒸馏收益是否随student规模变化

### Day 7：汇总所有实验结果

- [ ]  整理所有实验表格
- [ ]  绘制所有图表（reliability diagram、难度分桶图、BoN曲线等）
- [ ]  确认是否有需要补充的实验

---

## Week 4：论文撰写

### Day 1：Introduction + Abstract

- [ ]  写Motivation：GenPRM强但慢，判别式PRM快但弱
- [ ]  写核心Insight：能力差距集中在困难步骤
- [ ]  写一句话贡献总结
- [ ]  列出4个贡献点

### Day 2：Related Work

- [ ]  Process Reward Models（PRM800K、Math-Shepherd、GenPRM等）
- [ ]  Knowledge Distillation in NLP
- [ ]  Model Calibration
- [ ]  强调没有人做过PRM蒸馏

### Day 3：Method

- [ ]  问题定义与符号说明
- [ ]  难度自适应蒸馏策略（配公式+图示）
- [ ]  错误类型感知蒸馏（配错误分类体系表格）
- [ ]  置信度校准（配公式）
- [ ]  总损失函数

### Day 4：Experiments

- [ ]  实验设置（数据集、基座模型、超参数、评测指标）
- [ ]  主实验结果表
- [ ]  消融实验表
- [ ]  分析实验（难度分桶、错误类型、协同效应）

### Day 5：Analysis + Case Study

- [ ]  定性分析（case study展示）
- [ ]  推理效率分析（速度对比表）
- [ ]  Limitation讨论

### Day 6：Conclusion + 整体润色

- [ ]  写Conclusion
- [ ]  统一符号和术语
- [ ]  检查所有表格和图表的编号引用
- [ ]  检查参考文献格式

### Day 7：最终检查

- [ ]  通读全文确认逻辑通顺
- [ ]  检查是否有遗漏的实验
- [ ]  确认投稿目标（ACL/EMNLP/NeurIPS）的格式要求
- [ ]  准备Appendix（完整超参数表、更多case study、补充实验）

---

## 关键里程碑检查点

|时间点|检查内容|通过标准|
|---|---|---|
|Week1 Day4|蒸馏数据集质量|teacher-hardlabel一致率>80%|
|**Week1 Day7**|**CE+KL vs CE_only**|**ProcessBench提升≥2个点**|
|Week2 Day7|完整模型 vs CE+KL|进一步提升≥2个点|
|Week3 Day2|消融实验|每个组件去掉后性能下降|
|Week3 Day4|难度分桶分析|困难桶提升最大|
|Week4 Day7|论文初稿完成|可提交|

---

## 所需资源清单

|资源|最低要求|推荐配置|
|---|---|---|
|GPU|1×A100 80G|2×A100 80G|
|显存需求（teacher推理）|~28G（GenPRM-7B + vllm）|用2卡tensor parallel更快|
|显存需求（student训练）|~12G（1.5B + batch32）|单卡即可|
|磁盘|50G（模型+数据）|100G留余量|
|Teacher推理时间（50K）|~6h（单A100）|~3h（双A100）|
|Teacher推理时间（300K）|~36h（单A100）|~18h（双A100）|
|Student训练时间|~3h/run|消融需要跑5-6次|
