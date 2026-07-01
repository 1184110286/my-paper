# Research Contract：AGF-ST-HGAN-MCBG-MalSnif v1

本 contract 针对你提供的 **MalSnif baseline** 与 **“基于自适应门控融合的 MalSnif 方案一优化改造方案”**。MalSnif baseline 的核心是：审计日志解析、进程事件序列与调用关系抽取、provenance graph 构建与简化、NLP 语义特征提取、GCN 检测，并通过训练阶段下采样缓解类别不平衡；公开元数据也确认该论文主题是 audit-log-based HIDS、semantic-structural fusion、provenance graph 和 GNN。([Beijing Institute of Technology][1])

---

## 1. 研究问题与当前上下文

### 1.1 缺失信息

**必须补充的信息：**

1. **MalSnif 代码与数据处理细节**：是否有官方代码、是否复现论文的事件过滤、路径归一化、序列简化、图简化与下采样逻辑。没有这些，不能把“论文表格数值”直接当成公平 baseline，只能以“本地复现 A0”为主比较对象。
2. **数据集 split 与标签粒度**：LotL、DARPA、StreamSpot 的 train/val/test 划分、攻击节点标签、进程节点标签、图标签来源必须固定。尤其 StreamSpot 在 MalSnif 论文中不提供单个攻击事件标签，只能做 graph-level 是否含恶意活动的检测；若要做 node-level，必须额外定义标签映射。
3. **主任务口径**：本方案声称保持 MalSnif 的节点级检测粒度，即只对 process / attack-candidate 节点输出恶意概率；如果后续改成图级或子图级检测，应记为 v2，不属于当前 contract。
4. **硬件与训练预算**：GPU 型号、显存、最长训练时间、最大允许推理延迟需要确定，否则效率 claim 只能写成“记录结果”，不能写成“满足实时性”。
5. **阈值选择规则**：二分类阈值必须在 validation set 上确定，不能在 test set 上调。

**可以暂时假设的信息：**

1. 主比较对象是 **本地复现的 MalSnif 原模型 A0**，论文表格只作为 sanity check。MalSnif 论文报告的三数据集结果为：LotL F1=0.972，DARPA F1=0.964，StreamSpot F1=0.976；这些数值不能替代本地 A0。
2. 主验证数据集为 **LotL 与 DARPA**；StreamSpot 作为图级或场景级泛化辅助实验，除非补充 node-level 标签。
3. 至少使用 **3 个 random seeds** 做 confirmatory run。
4. 本方案 v1 不引入 LLM embedding、自监督预训练、focal loss、weighted BCE、动态图 TGAT、强化学习或子图重构作为主创新；这些都可以作为 exploratory 或 v2。

### 1.2 当前 task、setting 与 baseline

当前任务是基于 Windows / audit-log / provenance graph 的主机入侵检测，目标是检测 LoTL、Fileless、APT 等隐蔽攻击。MalSnif 原流程可以概括为：Procmon/ETW 审计日志 → 日志清洗、路径归一化、事件筛选、Word2Vec 向量化 → 进程事件序列提取与溯源图构建 → 外围节点裁剪与图简化 → GRU/BiLSTM 语义特征提取 → GCN 结构聚合 → MLP/Sigmoid 节点分类 → 后置下采样参与损失计算。

MalSnif baseline 的检测模型在 provenance graph 中存储 event vectors 与 process event sequence vectors，先从图中提取语义特征，再使用 GCN 进行预测；最终通过 linear layer + sigmoid 将节点映射到 `[0,1]` 的恶意概率。 论文还强调在类别极不平衡时，precision 与 F1 比 accuracy 更关键，因为 precision 反映告警可靠性，F1 反映整体检测性能。

### 1.3 核心瓶颈与 proposed idea 想解决的问题

本 proposed idea 不是泛泛堆模块，而是针对 MalSnif 的四个明确瓶颈做最小侵入式改造：

| 瓶颈  | MalSnif 原机制           | Proposed idea                                                       |
| --- | --------------------- | ------------------------------------------------------------------- |
| 语义侧 | Word2Vec + GRU/BiLSTM | 用 Multi-kernel 1D-CNN + BiGRU + Multi-head Attention 捕获局部攻击载荷与全局上下文 |
| 结构侧 | 同构 GCN                | 用 ST-HGAN 区分节点类型、边类型与时间偏置                                           |
| 图简化 | 外围节点硬裁剪               | 用注意力软修剪 / Top-k 邻居聚合，避免物理删除潜在攻击链                                    |
| 融合侧 | 静态拼接或晚期融合             | 用 adaptive gated fusion 在节点级动态分配语义与结构权重                             |

这些改造目标来自 idea 文档对 MalSnif 的重新定位：语义侧解决局部恶意参数组合稀释，结构侧解决同构聚合弱化类型差异，融合侧解决不同攻击场景下证据来源不一致的问题。

---

## 2. 原始 Idea 与核心不变量

### 2.1 原始 idea 的核心机制

本 idea 的核心机制是：

> 在 MalSnif 的审计日志语义—溯源图结构联合检测框架上，同时学习节点的语义表示 `s_v` 与异构结构上下文表示 `g_v`，再通过可学习向量门控 `z_v` 在节点级动态决定语义证据与结构证据的贡献，最终输出 process / attack-candidate 节点的恶意概率。

完整主模型命名为 **AGF-ST-HGAN-MCBG-MalSnif**。idea 文档定义的结构为：MCBG 语义分支输出 `s_v`，ST-HGAN 结构分支输出 `g_v`，自适应门控 `z_v = sigmoid(W_z[s_v,g_v,|s_v-g_v|,s_v⊙g_v]+b_z)`，融合特征 `f_v = z_v⊙s_v + (1-z_v)⊙g_v`，再由 MLP/Sigmoid 做节点级分类。

### 2.2 不可随意改变的核心不变量

以下内容是 v1 idea 的本质，不应在主实验中随意改变：

1. **任务粒度不变**：保持 MalSnif 的节点级检测，不把主任务改成图级、子图级或攻击故事重构。
2. **语义—结构双分支不变**：必须同时保留 MCBG 语义分支与 ST-HGAN / HGAN 结构分支；只保留一个分支不能声称验证了该 idea。
3. **局部—全局语义机制不变**：MCBG 的本质是 CNN 捕获局部短跨度攻击载荷，BiGRU 捕获上下文，多头注意力聚焦关键事件。若改成单纯 Transformer、纯 CNN 或纯 RNN，属于新 idea。
4. **异构关系建模不变**：结构分支必须显式使用 node type、edge / relation type，完整模型还应包含轻量 time bias；若退回普通 GCN，只能作为 ablation。
5. **软修剪不物理删除原始图**：Top-k 只影响消息聚合，不应删除原始审计图或回溯证据。idea 文档已明确低注意力边仍可能有攻击意义，因此原始图应保留。
6. **向量门控是核心机制**：主模型使用 vector gate，而不是固定拼接、平均、标量权重或投票。标量门控只能作为 A7 ablation。
7. **公平比较不变**：第一版保留 MalSnif 的 after-forward downsampling，以便与 baseline 对齐；改 weighted BCE / focal loss 属于附加实验或 v2。

### 2.3 可以调整的实现形式

以下调整属于合理优化，只要记录即可，不改变原 idea：

* CNN kernel size、channel 数、BiGRU hidden size、HGAN hidden size、attention heads、dropout、Top-k、learning rate、batch size、early stopping patience。
* `LayerNorm` vs `BatchNorm` vs no norm 的稳定性选择。
* 稀疏矩阵、mini-batch graph sampling、cache、mixed precision、gradient clipping 等工程优化。
* 关系类型命名与 time bucket 划分，只要不改变“类型/关系/时间偏置”的机制。
* 分支 dropout 与弱多样性正则可作为稳定性辅助，但必须 ablate；不能在主结果失败后临时加入再声称是原 idea。

### 2.4 已经偏离原 idea 的修改

以下修改必须标记为 **idea revision / v2**：

* 将主任务从节点级检测改为图级、子图级或攻击链重构。
* 用 LLM embedding 替代 Word2Vec + MCBG。
* 用 TGAT / TGN 等完整动态图模型替代轻量 time bias。
* 引入 focal loss、contrastive loss、自监督预训练作为主要性能来源。
* 用 ensemble voting 替代 adaptive gate。
* 物理删除 Top-k 之外的边并在调查阶段无法恢复。
* 使用 test set 选择阈值、Top-k、模型层数或融合策略。

---

## 3. Method 映射与允许优化空间

### 3.1 对 MalSnif 的模块替换关系

| MalSnif 原模块      | v1 处理 | 本方案实现                                              | 对 claim 的影响  |
| ---------------- | ----- | -------------------------------------------------- | ------------ |
| Procmon/ETW 日志采集 | 保留    | 不改变数据源                                             | 不作为创新点       |
| 路径清洗、正则归一化       | 保留    | 复用 MalSnif 预处理                                     | 公平对比要求       |
| 39 类事件筛选         | 保留并增强 | 映射到 node / edge / relation type                    | 支撑 ST-HGAN   |
| Word2Vec         | 保留    | 作为 MCBG 输入嵌入                                       | 避免引入额外语义模型变量 |
| GRU + BiLSTM     | 替换    | Multi-kernel 1D-CNN + BiGRU + Multi-head Attention | 验证 H1        |
| GCN              | 替换    | HGAN / ST-HGAN                                     | 验证 H2        |
| 外围节点硬裁剪          | 弱化或替换 | soft pruning + Top-k 聚合                            | 验证 H2        |
| 静态拼接融合           | 替换    | Adaptive gated fusion                              | 验证 H3        |
| MLP/Sigmoid 分类器  | 保留    | 输入从 concat 改为 gated feature                        | 不单独 claim    |
| 后置下采样训练          | 第一版保留 | forward 后采样计算 BCE                                  | 公平比较要求       |

idea 文档给出的算法也明确要求：先构建 MalSnif graph，再赋 node / relation type，对目标节点用 Word2Vec→CNN→BiGRU→Attention 得到 `s_v`，对异构图做 relation-temporal attention 得到 `g_v`，最后用向量门控融合并在 post-forward sampled process nodes 上计算 BCE loss。

### 3.2 允许优化空间与记录规则

| 优化类型                                         |       是否允许 | 是否必须记录 |              是否需要重跑 baseline | 是否影响 claim        |
| -------------------------------------------- | ---------: | -----: | ---------------------------: | ----------------- |
| bug fix，例如维度错误、mask 错误、label 对齐错误            |         允许 |     必须 |          若影响 A0 共享代码，必须重跑 A0 | 不影响 idea，但影响实验有效性 |
| 代码重构、模块拆分、缓存、稀疏化                             |         允许 |     必须 |               通常不需要，除非改变数值结果 | 不影响               |
| 显存优化、mixed precision、gradient checkpointing  |         允许 |     必须 |        若只用于新模型，需说明；最好同等应用 A0 | 影响效率 claim        |
| gradient clipping、dropout、weight decay 小范围调参 |         允许 |     必须 |      若对新模型大量调参，也要给 A0 合理调参机会 | 可能影响              |
| hidden size、head 数、Top-k、kernel size 小范围选择   |         允许 |     必须 |     不一定，但选择规则必须基于 validation | 不改变机制             |
| loss 加入 `L_div` 或 `L_att`                    |      有条件允许 |     必须 |  需要做 no-regularizer ablation | 只能作为稳定性辅助 claim   |
| focal loss / weighted BCE 作为主损失              | 不作为 v1 主实验 |     必须 | 必须重跑 A0 且标记 exploratory / v2 | 会改变 claim         |
| 改为 graph-level / subgraph-level              |   不允许作为 v1 |     必须 |                        新实验协议 | v2                |
| test set 后调阈值或调模型                            |         禁止 | 必须标记违规 |          结果不可作为 confirmatory | 破坏 contract       |

---

## 4. 核心 Hypothesis

### H1：MCBG 语义分支提升局部攻击载荷识别能力

**机制**：Multi-kernel 1D-CNN 捕获命令参数、路径片段、注册表键、网络动作中的局部 N-gram 恶意模式；BiGRU 建模前后事件上下文；multi-head attention 对关键事件片段赋更高权重。
**影响对象**：主要影响依赖命令行、路径、注册表、脚本参数等语义证据的攻击节点，尤其 LotL / Fileless。
**预期方向**：A1 = MalSnif + MCBG 相对 A0 在 Recall、F1 上提升，Precision 不显著下降。
**指标**：process-node Precision、Recall、F1、AUC；额外记录 attention top-k event 是否覆盖关键日志片段。
**理论理由**：MalSnif 原 GRU/BiLSTM 对长序列建模有效，但局部短跨度攻击载荷可能被长序列稀释；CNN 的局部感受野和 attention pooling 应提高局部恶意组合的显著性。idea 文档将该贡献定义为 local-global semantic encoder。

### H2：ST-HGAN + soft pruning 提升异构因果上下文建模与降低误报

**机制**：type-specific projection 避免不同类型节点被同一 GCN 权重混合；relation-aware attention 区分 read/write/create/connect 等关系；time bias 提供轻量时序线索；Top-k soft pruning 降低噪声邻居对聚合的干扰。
**影响对象**：主要影响依赖结构证据的攻击节点，例如横向移动、父子进程链、异常网络连接、伪装正常路径但依赖关系异常的进程。
**预期方向**：A2 = MalSnif + HGAN 与 A3 = MCBG + HGAN + static concat 相对 A0/A1 在 Precision、FPR、F1 上提升；A5 去掉 time bias 与 A6 改为 hard pruning 后性能下降或解释性变差。
**指标**：Precision、FPR、F1、AUC、relation-level attention 分布、Top-k 邻居覆盖率。
**理论理由**：MalSnif 原 GCN 默认同构聚合，容易弱化进程、文件、注册表、网络等实体和关系的安全语义差异；异构关系注意力应减少不相关邻居引入的误报。

### H3：Adaptive gated fusion 优于静态融合

**机制**：不同攻击阶段依赖的证据来源不同：Fileless / encoded command 更依赖语义，横向移动 / C2 / 父子进程链更依赖结构。向量门控允许每个节点、每个维度动态选择语义或结构证据。
**影响对象**：语义强但结构弱、结构强但语义弱、语义和结构均强的混合攻击节点。
**预期方向**：A4 = MCBG + HGAN + adaptive gate 相对 A3 = static concat 在 F1、Precision 或 FPR 上稳定提升；A7 scalar gate 弱于 vector gate；gate distribution 不应长期塌缩到单一分支。
**指标**：A4-A3 的 F1 / Precision / FPR 差异，`GateSem = mean(z_v)` 与 `GateStruct = 1-mean(z_v)` 的攻击类型分布，gate entropy / branch usage。
**理论理由**：静态融合假设语义与结构证据贡献固定；APT/LoTL 场景中证据来源随攻击阶段变化，动态门控更符合该 setting。idea 文档也将 gate 设计为节点级动态调节语义与结构贡献并提供可解释权重。

---

## 5. Success / Failure / Unclear Signals

### 5.1 Baseline 有效性前置条件

正式比较前必须先确认 A0 = 本地复现 MalSnif 是有效 baseline。MalSnif 论文报告三数据集检测效果为 LotL：Accuracy 0.981 / Precision 0.989 / Recall 0.955 / F1 0.972；DARPA：0.976 / 0.982 / 0.946 / 0.964；StreamSpot：0.992 / 0.952 / 1.000 / 0.976。

若本地 A0 与论文主指标相差超过 **2 个百分点 F1**，需要先判断是数据 split、预处理、标签、下采样、随机种子或实现差异造成。未解释清楚前，不能声称新模型优于 MalSnif，只能声称优于“当前实现的 MalSnif-like baseline”。

### 5.2 Success signal

v1 的 confirmatory success 必须同时满足以下条件：

1. **主指标提升**：在 LotL 与 DARPA 两个主数据集上，A4 相对 A0 的 mean F1 至少提升 **≥ 1.0 个百分点**，或在 A0 已接近饱和时，FPR 相对降低 **≥ 10%** 且 Recall 下降不超过 **1.5 个百分点**。
2. **次指标不恶化**：Precision 不低于 A0，或 Precision 下降不超过 **0.5 个百分点** 且 Recall 有明确提升；AUC 不低于 A0。
3. **机制支持**：A4 > A3，说明 gate 不只是装饰；A1 或 A2 至少有一个对主指标有正贡献；A5/A6/A7 至少有两个 ablation 出现预期下降或可解释变化。
4. **seed 稳定性**：至少 3 个 seeds 中，A4 相对 A0 的 F1 提升在 **≥ 2/3 seeds** 上为正，且 mean improvement 大于 1 个标准差。
5. **训练成本约束**：A4 的训练时间、峰值显存、推理延迟不超过 A0 的 **2×**；若超过 2× 但指标提升明显，只能 claim “accuracy-oriented improvement”，不能 claim practical / efficient improvement。
6. **评估口径一致**：split、标签、threshold、checkpoint selection、下采样位置与 A0 对齐；任何不一致必须作为单独实验报告。

### 5.3 Failure signal

以下任一情况应判定为 failure，而不是包装为成功：

1. A4 相对 A0 在 LotL 与 DARPA 上 mean F1 没有提升，且 Precision/FPR 也没有实质改善。
2. A4 的提升只来自 threshold 调整、test set 调参或更长训练，而非架构机制。
3. A4 优于 A0，但 A3、A5、A6、A7 不支持门控、时间偏置或软修剪机制，说明核心 hypothesis 不成立。
4. Gate 长期塌缩到几乎全语义或全结构，例如 `mean(z_v)` 接近 0 或 1 且方差极小，同时 ablation 不受影响。
5. 新模型 Recall 明显下降，即漏报增加，尤其在 DARPA APT 节点上 Recall 下降超过 **2 个百分点**。
6. 训练/推理成本超过 A0 的 **3×** 且没有显著 Precision/FPR/F1 收益。
7. 发现数据泄漏、标签泄漏、test set 调参、重复样本跨 split 或 baseline 未复现问题。

### 5.4 Unclear / partial signal

以下结果只能写成 unclear 或 partial：

1. A4 主指标提升在 **0.3–1.0 个百分点**之间，但 seed 方差较大。
2. A4 提升 F1，但主要来自 Recall 提升而 Precision/FPR 明显变差；安全检测中这可能增加告警疲劳。
3. A4 在 LotL 有效、DARPA 无效，或反过来；可写作 dataset-specific partial support。
4. Gate 分布看起来有差异，但 A4 与 A3 性能无差异；只能说 gate 可能提供解释线索，不能说 gate 提升检测。
5. StreamSpot 图级结果提升，但 node-level 主任务无提升；只能作为 auxiliary evidence。

---

## 6. 实验协议与 Ablation Plan

### 6.1 数据集与任务

| 数据集        | v1 用途 | 任务口径                                              |
| ---------- | ----- | ------------------------------------------------- |
| LotL       | 主实验   | process-node / attack-candidate node detection    |
| DARPA      | 主实验   | process-node / malicious entity detection         |
| StreamSpot | 辅助实验  | graph-level 是否含攻击；除非补充 node-level 标签，否则不用于主 claim |

MalSnif 论文使用 LotL、DARPA 和 StreamSpot 评估，并在 StreamSpot 上受限于无 individual attack event labels，只能判断图是否含恶意活动。 因此，本 contract 的主结论不得依赖未经定义的 StreamSpot node-level 结果。

### 6.2 Split 与 leakage 控制

1. 优先沿用 MalSnif 论文或官方代码 split。
2. 若无官方 split，则在任何模型调试前固定 train/val/test，并保存 split 文件 hash。
3. 不允许同一攻击链、同一高度重复的进程事件序列、同一 provenance graph 的强相关节点同时出现在 train 与 test 中，除非 baseline 论文明确这样做且为了复现必须保持一致。
4. 所有 threshold、Top-k、hidden size、checkpoint 只能基于 validation set 选择。
5. Test set 只运行最终冻结配置。

### 6.3 Metrics

必须汇报：

```text
Accuracy, Precision, Recall, F1, AUC, FPR, FNR
```

建议额外汇报：

```text
Detection latency, training time, peak GPU memory, model parameters, alert volume
```

主排序指标为 **F1 + Precision/FPR**，不以 Accuracy 作为主要成功依据。MalSnif 论文也说明，在正常与攻击事件严重不平衡时，precision 和 F1 更关键。

### 6.4 Training budget 与 checkpoint

* 每个 confirmatory setting 至少 **3 seeds**。
* Debug 阶段允许单 seed 快速跑，但 debug 结果不得写入主表。
* Checkpoint selection：固定使用 validation F1；若类别极不平衡，也可预注册 validation AUPRC / F1，但不能中途切换。
* Early stopping：固定 patience，例如 10 或 20 epochs。
* 训练日志必须记录 loss、Precision、Recall、F1、AUC、FPR、GPU memory、每 epoch wall-clock time。

### 6.5 Confirmatory ablation plan

| ID | 模型                          | 验证机制     | 移除/替换组件                     | 预期结果              | 相反结果说明                    |
| -- | --------------------------- | -------- | --------------------------- | ----------------- | ------------------------- |
| A0 | MalSnif 原模型                 | baseline | 无                           | 复现论文或合理接近         | baseline 无效则全部暂停          |
| A1 | MalSnif + MCBG              | H1 语义增强  | GRU/BiLSTM → MCBG，GCN 不变    | Recall/F1 提升      | 语义模块可能无效或实现错误             |
| A2 | MalSnif + HGAN              | H2 异构结构  | GCN → HGAN，语义不变             | Precision/FPR 改善  | 同构 GCN 不是瓶颈，或关系定义无效       |
| A3 | MCBG + HGAN + static concat | 双分支主体    | 无 adaptive gate             | 优于 A0/A1/A2       | 双分支组合无收益，可能过拟合            |
| A4 | MCBG + HGAN/ST-HGAN + AGF   | H3 门控融合  | 主模型                         | 优于 A3             | gate 没有贡献，不能 claim gating |
| A5 | A4 去掉 time bias             | 时间偏置     | remove time bias            | A4 > A5           | 时间偏置不是关键或实现无效             |
| A6 | A4 使用 hard pruning          | 软修剪      | soft pruning → hard pruning | A4 > A6，Recall 更稳 | 软修剪无收益或 Top-k 设计错误        |
| A7 | A4 scalar gate              | 向量门控     | vector gate → scalar gate   | A4 > A7           | 向量级门控不是必要机制               |

idea 文档已预注册了 A0–A7 的消融结构，并建议 A4 作为最终主模型。

### 6.6 Gate explainability analysis

必须记录：

```text
GateSem(v) = mean(z_v)
GateStruct(v) = 1 - mean(z_v)
```

至少统计：

1. LotL / Fileless 样本中 GateSem 是否更高。
2. 横向移动 / 网络连接 / 父子进程链场景中 GateStruct 是否更高。
3. 同一攻击链不同阶段 gate 是否变化。
4. Top-k event / relation / neighbor 是否覆盖可解释证据。

该分析只能作为 **mechanism support**，不能单独证明因果解释。idea 文档也提醒注意力解释性可能被质疑，应配合反事实删除 Top-k 事件验证。

### 6.7 Exploratory ablation

以下实验可以做，但必须标注为 exploratory，不得混入 confirmatory plan：

* weighted BCE / focal loss。
* 更深 HGAN，例如 3–4 层。
* 更大 hidden size 或更多 attention heads。
* LLM log embeddings。
* Graph-level / subgraph-level variant。
* Cross-dataset transfer。
* Robustness to synthetic noisy benign activity。
* Adversarial mimicry 或 evasion 测试。

---

## 7. 风险、调试与变更记录规则

### 7.1 主要风险

1. **Baseline 未复现**：如果 A0 与论文结果差距过大，任何提升都可能只是实现差异。
2. **数据泄漏**：相似进程序列、同一攻击链或同一图的节点跨 split 会夸大性能。
3. **标签粒度不一致**：尤其 StreamSpot 无 individual attack labels，不能直接混入 node-level 主实验。
4. **下采样不公平**：MalSnif 使用 forward 后下采样，若新模型改变下采样位置，会破坏比较。MalSnif 论文说明传统采样会在前向传播前造成结构信息损失，因此其策略是在 forward 后下采样，使节点先学习完整图结构，再平衡正常/恶意节点进行 loss 与 backpropagation。
5. **参数量和训练时间增加**：新模型更复杂，性能提升可能来自容量而不是机制。
6. **阈值选择偏差**：若 test set 调阈值，会导致结果不可用。
7. **门控塌缩**：AGF 可能学成固定偏向某一分支。
8. **软修剪漏报**：Top-k 可能排除低频但关键的攻击边，因此原始图必须保留。
9. **Ablation 不独立**：如果 A1/A2/A3/A4 的参数量差异过大，机制解释会被削弱。
10. **评估脚本不一致**：不同模型使用不同 node mask、threshold 或 metric 脚本会造成虚假提升。

### 7.2 Debug 允许范围

允许：

* 修复数据读取、mask、维度、batch、device、overflow、NaN、seed、日志记录等问题。
* 修复明显不符合原 idea 的实现错误，例如 relation type 未被使用、time bias 未接入 attention、gate 输入顺序错误。
* 增加 gradient clipping、dropout、LayerNorm、early stopping 等稳定性措施。
* 优化稀疏矩阵和缓存以降低显存。

不允许在看到主实验结果后：

* 改 success threshold。
* 改主 metric。
* 改 test split。
* 删除不利数据集。
* 把 exploratory 结果包装为 confirmatory。
* 把失败模块替换为新机制但仍称为原 idea。

### 7.3 变更记录模板

每次重要修改必须记录：

```text
Change ID:
Date / time:
Git commit:
修改内容:
修改原因:
修改前是否已经看到相关实验结果: yes/no
影响模块: data / model / loss / training / evaluation
是否影响 hypothesis: no / minor / major
分类: implementation fix / optimization / experimental-control change / idea revision
是否需要重跑 A0: yes/no
是否需要重跑 ablation: yes/no
备注:
```

### 7.4 implementation fix vs idea revision

| 修改                                                   | 分类                          |
| ---------------------------------------------------- | --------------------------- |
| 修正 relation type 未进入 attention                       | implementation fix          |
| 修正 gate 维度广播错误                                       | implementation fix          |
| 将 hard pruning 改回 message-passing Top-k soft pruning | implementation fix          |
| 将 BiGRU hidden 从 128 调到 96 以防 OOM                    | optimization                |
| 加 gradient clipping 防止 NaN                           | optimization                |
| 改为 weighted BCE 主损失                                  | idea revision / exploratory |
| 用 Transformer 替代 MCBG                                | idea revision               |
| 用 TGN/TGAT 替代轻量 time bias                            | idea revision               |
| 改 graph-level 输出                                     | idea revision               |
| test 后重新选 split 或阈值                                  | invalid change              |

---

## 8. 结果判定、Claim 边界与版本规则

### 8.1 实验结束后的判定顺序

实验结束后，必须按以下顺序写结果：

1. A0 baseline 是否有效。
2. A4 是否满足 success signal。
3. H1/H2/H3 分别是否被支持。
4. A0–A7 ablation 是否与机制一致。
5. seed 稳定性是否足够。
6. 成本是否在预算内。
7. 最终判定：success / failure / unclear / partial。
8. 再写原因分析与论文叙述。

不得先写故事，再反向挑选指标支持故事。

### 8.2 Claim 边界

允许的 claim：

* “在与本地复现 MalSnif 对齐的预处理、split 和训练协议下，AGF-ST-HGAN-MCBG 在 LotL / DARPA 上相对 A0 提升了节点级检测性能。”
* “Ablation 显示 MCBG、HGAN/ST-HGAN、adaptive gate 对性能或机制解释有贡献。”
* “Gate distribution 为不同攻击场景下语义/结构证据权重差异提供了辅助解释。”

不允许的 claim：

* “证明了注意力就是因果解释。”
* “证明对所有 APT / LoTL / Fileless 攻击泛化。”
* “证明实时可部署”，除非有明确 latency、throughput、memory 结果。
* “证明 soft pruning 不会漏报”，除非有反事实和召回分析。
* “StreamSpot node-level 检测提升”，除非补充合法 node-level 标签。

### 8.3 版本规则

**v1：当前 contract 对应版本**

```text
MalSnif 原预处理
+ MCBG
+ HGAN/ST-HGAN with relation/time bias
+ soft pruning as message-passing Top-k
+ adaptive vector gated fusion
+ MLP/Sigmoid node classifier
+ after-forward downsampling
```

**v1.1：implementation fix 版本**

只包含 bug fix、稳定性修正、显存优化、小范围超参调整。不改变 hypothesis。

**v2：idea revision 版本**

以下任一修改触发 v2：

* 换主损失为 weighted BCE / focal loss / contrastive loss。
* 引入 LLM embedding 或 Transformer-only semantic encoder。
* 引入动态图 TGAT/TGN 作为主要结构模型。
* 改成 graph-level / subgraph-level / attack-story reconstruction。
* 改用 ensemble / voting 替代 adaptive gate。
* 使用外部威胁知识或 ATT&CK 规则作为主要特征。
* 更换主数据集或主任务定义。

v2 必须说明：改了什么、为什么改、是否已经看过 v1 相关结果、改动是否仍服务原假设，还是新的 causal mechanism。禁止在看到 v1 主结果失败后，把 v2 结果写成 v1 的事前预测。

[1]: https://pure.bit.edu.cn/en/publications/a-novel-host-based-intrusion-detection-approach-leveraging-audit-/ "
        A novel host-based intrusion detection approach leveraging audit logs
      \-  Beijing Institute of Technology"
