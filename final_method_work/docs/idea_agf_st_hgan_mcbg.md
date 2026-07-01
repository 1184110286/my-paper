# 基于自适应门控融合的 MalSnif 方案一优化改造方案

> 方案名称建议：**AGF-ST-HGAN-MCBG**  
> 全称：**Adaptive-Gated Fusion Spatio-Temporal Heterogeneous Graph Attention Network with Multi-head CNN-BiGRU**  
> 中文名：**自适应门控融合的多头注意力 CNN-BiGRU 与时空异构图注意力检测框架**

---

## 0. 核心结论

本方案以创新方案文档中的**方案一 ST-HGAN-MCBG** 为主干，保留其“语义序列增强 + 异构图注意力建模 + 软修剪”的核心逻辑；同时仅吸收方案三中的**自适应门控特征融合**思想，用可学习门控替换方案一原本的静态拼接融合。最终形成一个对 MalSnif 改动清晰、创新点集中、工程代价可控的改造版框架。

相对 MalSnif 原方法，本方案的改造目标不是单纯堆叠更多模型，而是解决三个具体问题：

1. **语义侧**：MalSnif 的 Word2Vec + GRU/BiLSTM 对长日志序列中局部恶意参数组合不够敏感。改为 **1D-CNN + BiGRU + 多头注意力**，同时捕获局部 N-gram 攻击载荷和全局时序上下文。
2. **结构侧**：MalSnif 的 GCN 按同构图做均匀聚合，难以区分进程、文件、注册表、网络等不同实体/事件类型。改为 **时空异构图注意力网络 ST-HGAN**，按节点类型、边类型和时间偏置进行关系感知聚合。
3. **融合侧**：MalSnif 以及原方案一都偏向静态融合，无法针对不同攻击场景动态判断“语义更重要”还是“结构更重要”。加入**自适应门控融合**，让模型在每个待检测节点上自动分配语义分支与结构分支的权重。

最终检测仍保持 MalSnif 的**节点级检测粒度**：仅对进程节点或关键系统实体节点输出恶意概率，不转为图级或子图级粗粒度告警。

---

## 1. 对 MalSnif 基线的重新定位

MalSnif 的原始检测流程可以抽象为：

```text
Procmon/ETW 审计日志
   ↓
日志清洗、路径归一化、事件筛选、Word2Vec 向量化
   ↓
进程事件序列提取 + 溯源图构建
   ↓
外围节点裁剪与图简化
   ↓
GRU/BiLSTM 提取语义特征
   ↓
GCN 聚合图结构特征
   ↓
特征融合 + MLP/Sigmoid 节点分类
   ↓
后置下采样参与损失计算
```

这个流程的优势在于把审计日志语义和溯源图结构结合起来，适合检测 LoTL 与 APT 这类隐蔽攻击。但作为后续改造基线，它存在四个可以直接突破的点：

| 位置 | MalSnif 原机制 | 主要不足 | 本方案替换/优化 |
|---|---|---|---|
| 语义编码 | Word2Vec + GRU/BiLSTM | 对局部命令参数、路径片段、注册表键值等短跨度攻击载荷捕获不足 | 1D-CNN + BiGRU + Multi-Head Attention |
| 图建模 | GCN | 默认同构邻居聚合，弱化节点类型与关系类型差异 | ST-HGAN 异构关系注意力 |
| 图简化 | 外围节点硬裁剪 | 可能切断隐蔽攻击的间接依赖链 | 注意力软修剪 + Top-k 关系邻居选择 |
| 特征融合 | 静态拼接/晚期融合 | 不同攻击场景不能动态调节语义/结构权重 | 自适应门控融合 |

---

## 2. 最终总体架构

推荐最终框架如下：

```text
输入：格式化审计日志 e = <SrcId, SrcType, DstId, DstType, EdgeType, Time, Tag>

      ┌───────────────────────────────────────────────────────────────┐
      │ 1. 类型增强的 MalSnif 图构建                                   │
      │    - 继承路径清洗、事件筛选、Word2Vec 初始化                    │
      │    - 保留 process/event 节点体系                               │
      │    - 为节点、边补充 type / relation / time bias                │
      └───────────────────────────────────────────────────────────────┘
                              ↓
      ┌───────────────────────────────────────────────────────────────┐
      │ 2. 语义分支 MCBG                                                │
      │    Word2Vec → Multi-kernel 1D-CNN → BiGRU → Multi-Head Attention│
      │    输出：语义表示 s_v                                           │
      └───────────────────────────────────────────────────────────────┘
                              ↓
      ┌───────────────────────────────────────────────────────────────┐
      │ 3. 结构分支 ST-HGAN                                             │
      │    type-specific projection                                     │
      │    relation-aware attention                                     │
      │    temporal relation bias                                       │
      │    soft pruning / attention Top-k                               │
      │    输出：结构上下文表示 g_v                                      │
      └───────────────────────────────────────────────────────────────┘
                              ↓
      ┌───────────────────────────────────────────────────────────────┐
      │ 4. 自适应门控特征融合 AGF                                       │
      │    z_v = sigmoid(W_z [s_v, g_v, |s_v-g_v|, s_v⊙g_v] + b_z)      │
      │    f_v = z_v⊙s_v + (1-z_v)⊙g_v                                  │
      └───────────────────────────────────────────────────────────────┘
                              ↓
      ┌───────────────────────────────────────────────────────────────┐
      │ 5. 节点级分类器                                                 │
      │    y_hat_v = sigmoid(MLP(f_v))                                  │
      │    只对 process / attack-candidate 节点计算检测结果              │
      └───────────────────────────────────────────────────────────────┘
```

其中：

- `s_v` 表示节点 `v` 的纯语义特征，主要来自其事件序列、命令参数、路径片段、结果状态和细节字段。
- `g_v` 表示节点 `v` 的结构上下文特征，主要来自其异构邻居、关系类型、事件方向、时间相邻性和局部因果上下文。
- `z_v` 是自适应门控向量，控制每个维度上语义分支与结构分支的贡献比例。

推荐将该模型命名为：

```text
AGF-ST-HGAN-MCBG-MalSnif
```

如果用于论文标题，可以简化为：

```text
AG-MalSnif: Adaptive-Gated Semantic-Structural Heterogeneous Graph Learning for Audit-Log-Based Intrusion Detection
```

---

## 3. 模块一：类型增强的 MalSnif 图构建

### 3.1 继承 MalSnif 的输入格式

MalSnif 已将原始日志格式：

$$
e = \langle TimeOfDay, ProcessName, PID, Operation, Path, Result, Detail \rangle
$$

转化为更适合溯源图建模的格式：

$$
e = \langle SrcId, SrcType, DstId, DstType, EdgeType, Time, Tag \rangle
$$

本方案不推翻这一数据处理流程，而是在其基础上增加**类型语义**和**关系语义**。

### 3.2 节点类型设计

为了避免大幅增加节点规模，推荐采用“最小侵入式异构化”策略：不强制把所有文件、注册表、网络地址都拆成独立资源节点，而是优先保留 MalSnif 的 process/event 节点体系，并将 event 节点细分为类型。

推荐节点类型集合：

$$
\mathcal{T}_V = \{P, E_{proc}, E_{file}, E_{reg}, E_{net}, E_{query}\}
$$

含义如下：

| 节点类型 | 含义 | 示例 |
|---|---|---|
| `P` | 进程节点 | `powershell.exe`, `cmd.exe`, `excel.exe` |
| `E_proc` | 进程/线程相关事件 | `Process Create`, `Thread Create`, `Load Image` |
| `E_file` | 文件相关事件 | `CreateFile`, `ReadFile`, `WriteFile` |
| `E_reg` | 注册表相关事件 | `RegSetValue`, `RegCreateKey` |
| `E_net` | 网络相关事件 | `TCP Connect`, `TCP Send`, `UDP Receive` |
| `E_query` | 查询/系统设置相关事件 | `DeviceIoControl`, `QueryStatInformation` |

这种设计的好处是：

- 与 MalSnif 已筛选出的 39 类关键事件兼容；
- 不会像完整资源图那样显著增加节点规模；
- 能让 HGAN 学到“网络连接”“注册表写入”“文件执行”等事件类别的不同权重。

### 3.3 边关系类型设计

边关系不应只表示是否相连，而应编码系统动作。推荐将边关系定义为：

$$
\mathcal{T}_E = \{r = (src\_type, edge\_type, dst\_type, direction)\}
$$

工程上可将 39 类事件压缩为两级关系：

1. **粗粒度关系类**：`process`, `file`, `registry`, `network`, `query`。
2. **高危细粒度动作**：保留如 `Process Create`, `Load Image`, `WriteFile`, `RegSetValue`, `TCP Connect` 等关键操作作为单独关系类型。

这样既可以避免关系类型过多造成稀疏，也能保留对高危行为的敏感性。

### 3.4 时间偏置设计

方案一强调时空图建模，但不需要引入方案三的 TC-LSTM。推荐在 HGAN 注意力中加入轻量时间偏置：

$$
\Delta t_{uv} = |t_u - t_v|
$$

将时间间隔离散化为桶：

```text
[0, 1s], (1s, 10s], (10s, 1min], (1min, 10min], (10min, 1h], >1h
```

再将其映射为时间嵌入：

$$
\phi(\Delta t_{uv}) = Emb_{time}(bucket(\Delta t_{uv}))
$$

该设计的目的不是完全建模低频长周期攻击，而是在不显著增加计算开销的情况下，让结构注意力知道两个事件在时间上是否接近。

---

## 4. 模块二：MCBG 语义特征提取

MCBG 是方案一中最适合替换 MalSnif 原始 GRU/BiLSTM 语义提取器的部分。

### 4.1 输入序列构造

对每个待检测进程节点 `v`，收集其对应事件序列：

$$
S_v = [e_1, e_2, ..., e_L]
$$

每个事件转为 token 序列：

```text
[OP] Operation [PATH] PathSegments [RESULT] Result [DETAIL] Detail
```

例如：

```text
[OP] RegSetValue [PATH] hklm software microsoft windows currentversion run [RESULT] SUCCESS [DETAIL] ...
```

随后使用 MalSnif 已有的 Word2Vec/Skip-Gram 嵌入获得：

$$
X_v \in \mathbb{R}^{L \times d_w}
$$

其中 `L` 为序列长度，`d_w` 为词向量维度。

### 4.2 局部 N-gram 语义抽取：Multi-kernel 1D-CNN

APT 和 LoTL 攻击中，很多关键特征不是单个词，而是短语组合，例如：

- `ExecutionPolicy Bypass`
- `EncodedCommand`
- `RegSetValue + Run`
- `WriteFile + system32`
- `TCP Connect + uncommon_ip`

因此使用多尺度一维卷积：

$$
C_k = ReLU(Conv1D_k(X_v) + b_k), \quad k \in \{2, 3, 5\}
$$

拼接不同窗口的卷积输出：

$$
C_v = [C_2 \parallel C_3 \parallel C_5]
$$

然后进行最大池化，压缩序列长度：

$$
\tilde{C}_v = MaxPool(C_v)
$$

这一步可以降低后续 BiGRU 的序列长度，缓解原 MalSnif 在长事件序列上的计算冗余。

### 4.3 全局上下文编码：BiGRU

将压缩后的局部特征序列输入 BiGRU：

$$
\overrightarrow{h_t} = GRU_f(\tilde{C}_{v,t}, \overrightarrow{h_{t-1}})
$$

$$
\overleftarrow{h_t} = GRU_b(\tilde{C}_{v,t}, \overleftarrow{h_{t+1}})
$$

$$
H_v = [\overrightarrow{h_t} \parallel \overleftarrow{h_t}]_{t=1}^{L'}
$$

BiGRU 比 BiLSTM 参数更少，适合保持工程开销可控；双向结构仍能捕获攻击动作前后的上下文依赖。

### 4.4 多头注意力聚焦关键事件

对 BiGRU 输出使用多头注意力：

$$
Q_i = H_v W_i^Q, \quad K_i = H_v W_i^K, \quad V_i = H_v W_i^V
$$

$$
Head_i = softmax\left(\frac{Q_i K_i^T}{\sqrt{d_h}}\right)V_i
$$

$$
A_v = [Head_1 \parallel Head_2 \parallel ... \parallel Head_H]W^O
$$

最后进行全局池化得到语义表示：

$$
s_v = Pool(A_v)
$$

这里的注意力权重可用于解释：模型究竟关注了哪些命令参数、路径片段、注册表键或网络动作。

---

## 5. 模块三：ST-HGAN 结构特征提取

### 5.1 异构图定义

将 MalSnif 溯源图改造为异构图：

$$
G = (V, E, \phi, \psi)
$$

其中：

- $V$ 为节点集合；
- $E$ 为边集合；
- $\phi(v) \in \mathcal{T}_V$ 表示节点类型；
- $\psi(e) \in \mathcal{T}_E$ 表示边关系类型。

### 5.2 类型特异性投影

不同类型节点的特征分布不同，不能直接放进同一个 GCN 权重矩阵。先做类型特异性投影：

$$
h_v^{(0)} = W_{\phi(v)}x_v + b_{\phi(v)}
$$

其中 `x_v` 可以由以下内容组成：

```text
x_v = [semantic_init_v || node_type_emb_v || operation_emb_v || time_emb_v]
```

对进程节点，`semantic_init_v` 使用 MCBG 输出的 `s_v`；对事件节点，使用事件自身的 Word2Vec/GRU 简化表示或操作类型嵌入。

### 5.3 关系感知注意力

对中心节点 `v` 与邻居 `u`，在关系 `r` 下计算注意力分数：

$$
e_{vu}^{r,h} = LeakyReLU\left(a_{r,h}^{T}\left[W_{r,h}h_v^{(l)} \parallel W_{r,h}h_u^{(l)} \parallel e_r \parallel \phi(\Delta t_{uv})\right]\right)
$$

其中：

- `h` 表示注意力头；
- `e_r` 是关系类型嵌入；
- `φ(Δt_uv)` 是轻量时间偏置；
- 如果沿用 MalSnif 中已计算的 edge vector，可将 `EdgeAttr_uv` 经线性层压缩后并入注意力计算，但不进行 EGNN 式边状态迭代。

归一化得到：

$$
\alpha_{vu}^{r,h} = \frac{\exp(e_{vu}^{r,h})}{\sum_{k \in \mathcal{N}_v^r}\exp(e_{vk}^{r,h})}
$$

关系内聚合：

$$
m_v^{r,h} = \sum_{u \in \mathcal{N}_v^r}\alpha_{vu}^{r,h}W_{r,h}h_u^{(l)}
$$

多头拼接：

$$
m_v^r = Concat(m_v^{r,1}, ..., m_v^{r,H})
$$

### 5.4 关系级注意力

不同关系对恶意判定的重要性不同。例如，`TCP Connect`、`RegSetValue`、`Process Create` 往往比普通查询动作更敏感。使用关系级权重：

$$
\beta_{v,r} = \frac{\exp(q^T tanh(W_R m_v^r))}{\sum_{r' \in \mathcal{R}_v}\exp(q^T tanh(W_R m_v^{r'}))}
$$

最终节点更新：

$$
h_v^{(l+1)} = \sigma\left(W_O \sum_{r \in \mathcal{R}_v}\beta_{v,r}m_v^r + b_O\right)
$$

最后一层输出结构上下文表示：

$$
g_v = h_v^{(L_g)}
$$

### 5.5 注意力软修剪

MalSnif 的外围节点硬裁剪可以降低复杂度，但存在攻击链断裂风险。本方案采用软修剪：

1. 训练阶段保留图结构，不提前删除潜在相关节点；
2. 聚合阶段仅对每种关系下注意力最高的 Top-k 邻居参与消息传递；
3. 低注意力节点不被物理删除，只是不参与当前层聚合；
4. 推理解释时仍可回溯这些低注意力节点，避免审计证据丢失。

形式化表示为：

$$
\mathcal{N}_{v,topk}^{r} = TopK(\mathcal{N}_v^r, \alpha_{vu}^{r})
$$

$$
m_v^{r,h} = \sum_{u \in \mathcal{N}_{v,topk}^{r}}\alpha_{vu}^{r,h}W_{r,h}h_u^{(l)}
$$

该策略能在保持上下文可追溯性的同时，把实际聚合复杂度控制在可接受范围。

---

## 6. 模块四：自适应门控特征融合

这是本次按要求从方案三中引入的关键逻辑。注意：这里只引入**门控融合机制**，不引入方案三的 EGNN 和 TC-LSTM，以保证最终方案仍以方案一为主体。

### 6.1 为什么不能继续使用拼接

静态拼接的问题在于，它假设语义特征和结构特征对所有攻击类型都同等重要。但实际攻击并非如此：

| 攻击类型 | 语义特征强度 | 结构特征强度 | 例子 |
|---|---:|---:|---|
| Fileless / LoTL | 高 | 中或低 | PowerShell 可疑参数、宏命令、脚本执行 |
| 横向移动 | 中 | 高 | 进程链、远程连接、凭证访问后的网络行为 |
| 勒索软件 | 高 | 高 | 大量文件写入 + 特定路径/扩展名模式 |
| 伪装进程 | 低或中 | 高 | 正常名称进程出现异常父子关系或异常访问路径 |

因此，融合层需要对每个节点动态选择特征来源。

### 6.2 维度对齐

先将语义表示和结构表示映射到同一维度：

$$
\bar{s}_v = LayerNorm(W_s s_v + b_s)
$$

$$
\bar{g}_v = LayerNorm(W_g g_v + b_g)
$$

其中：

- `s_v` 是 MCBG 输出；
- `g_v` 是 ST-HGAN 输出；
- `\bar{s}_v, \bar{g}_v \in \mathbb{R}^{d_f}`。

### 6.3 门控输入

为了让门控判断两种特征之间的一致性与冲突程度，门控输入不应只拼接 `s` 和 `g`，还应加入差分项与交互项：

$$
q_v = [\bar{s}_v \parallel \bar{g}_v \parallel |\bar{s}_v - \bar{g}_v| \parallel (\bar{s}_v \odot \bar{g}_v) \parallel c_v]
$$

其中 `c_v` 是可选上下文统计特征，例如：

```text
c_v = [node_type, in_degree, out_degree, relation_count, time_span_bucket]
```

### 6.4 向量门控

推荐使用向量门控，而不是单个标量门控：

$$
z_v = \sigma(W_z q_v + b_z)
$$

$$
f_v = z_v \odot \bar{s}_v + (1 - z_v) \odot \bar{g}_v
$$

其中：

- `z_v` 越接近 1，表示该维度更依赖语义分支；
- `z_v` 越接近 0，表示该维度更依赖结构分支；
- `f_v` 是最终融合表示。

最终分类：

$$
\hat{y}_v = \sigma(MLP(f_v))
$$

### 6.5 门控解释性

门控值可以直接作为解释信号：

```text
GateSem(v) = mean(z_v)
GateStruct(v) = 1 - mean(z_v)
```

解释示例：

| 节点/场景 | 预期 GateSem | 预期 GateStruct | 解释 |
|---|---:|---:|---|
| `powershell.exe -EncodedCommand ...` | 高 | 中低 | 命令参数是主要证据 |
| `excel.exe → powershell.exe → TCP Connect` | 中 | 高 | 父子进程链与网络行为更关键 |
| `wannacry.exe` 大量 `WriteFile` | 中高 | 中高 | 语义与结构都强 |
| 伪装正常路径的恶意进程 | 中低 | 高 | 名称语义不明显，但依赖关系异常 |

这部分非常适合作为论文中的 case study 或可解释性分析。

### 6.6 防止门控退化

门控层可能退化成长期偏向某一个分支。建议加入两个轻量约束：

**1. 分支 Dropout**

训练时随机屏蔽一小部分语义或结构通道：

```text
p_drop_sem = 0.1
p_drop_struct = 0.1
```

迫使模型不能完全依赖单一模态。

**2. 弱多样性正则**

鼓励语义表示与结构表示不要完全重合：

$$
L_{div} = \frac{1}{|V|}\sum_{v \in V}\left|cos(\bar{s}_v, \bar{g}_v)\right|
$$

最终损失可写为：

$$
L = L_{BCE} + \lambda_{att}L_{att} + \lambda_{div}L_{div}
$$

其中：

- `L_BCE` 保持 MalSnif 的节点级二分类损失；
- `L_att` 是可选注意力稀疏约束，用于促使软修剪更清晰；
- `L_div` 防止语义分支与结构分支表示坍缩。

不建议在当前方案中引入自适应焦点损失，因为那属于方案二的训练机制，会分散“方案一 + 门控融合”的主线创新。

---

## 7. 训练流程设计

### 7.1 推荐训练流程

```text
Step 1. 继承 MalSnif 预处理
        - Procmon/ETW 日志解析
        - 39 类关键事件筛选
        - 路径清洗和 token 化
        - Word2Vec 初始化

Step 2. 构建异构图
        - process/event 节点
        - event 子类型
        - relation type
        - time bucket

Step 3. 训练 MCBG 语义分支
        - CNN 捕获局部恶意短语
        - BiGRU 捕获事件上下文
        - Multi-head attention 输出 s_v

Step 4. 训练 ST-HGAN 结构分支
        - 类型投影
        - 关系注意力
        - 时间偏置
        - 注意力 Top-k 软修剪

Step 5. 自适应门控融合
        - 计算 z_v
        - 得到 f_v
        - 输出节点恶意概率

Step 6. 后置下采样计算损失
        - 保持 MalSnif 的 after-forward sampling 思想
        - 不在前向传播前破坏图结构
```

### 7.2 为什么保留 MalSnif 的后置下采样

虽然创新方案文档指出降采样可能导致良性边界损失，但本次用户要求的主体是方案一和门控融合，不是方案二的训练机制。为了让改造边界清晰，推荐在第一版实验中保留 MalSnif 的后置下采样策略：

- 图前向传播仍使用完整或软修剪后的图；
- 损失计算阶段再采样正常节点；
- 这样可以与原论文结果做公平对比；
- 后续扩展实验可再加入 weighted BCE 或 focal loss 作为附加消融，而不是主创新点。

---

## 8. 与 MalSnif 的具体替换关系

| MalSnif 原模块 | 保留/替换 | 本方案实现 |
|---|---|---|
| Procmon/ETW 日志采集 | 保留 | 不改变数据源 |
| 路径清洗、正则归一化 | 保留 | 继续减少路径稀疏性 |
| 事件筛选 | 保留并增强 | 39 类事件映射到节点/边类型 |
| Word2Vec | 保留 | 作为 MCBG 输入嵌入 |
| GRU + BiLSTM | 替换 | Multi-kernel 1D-CNN + BiGRU + Multi-head Attention |
| GCN | 替换 | ST-HGAN |
| 外围节点硬裁剪 | 弱化/替换 | 注意力软修剪 + Top-k 邻居聚合 |
| 静态拼接融合 | 替换 | 自适应门控融合 AGF |
| MLP/Sigmoid 分类器 | 保留 | 输入从 concat 改为 gated feature |
| 后置下采样训练 | 第一版保留 | 便于与 MalSnif 做公平比较 |

---

## 9. 推荐算法描述

### Algorithm: AGF-ST-HGAN-MCBG for MalSnif

```text
Input:
  Parsed audit logs D
  Process/event labels Y for training
  Relation schema R
  Node type schema T

Output:
  Malicious probability y_hat_v for each target process node v

1. D_clean ← sanitize_path_and_filter_events(D)
2. G ← build_malsnif_graph(D_clean)
3. G_het ← assign_node_types_and_relation_types(G)
4. For each target node v:
      X_v ← Word2Vec(tokenize(event_sequence(v)))
      C_v ← MultiKernelCNN(X_v)
      H_v ← BiGRU(C_v)
      s_v ← MultiHeadAttentionPool(H_v)
5. For each node v in G_het:
      h_v^0 ← TypeSpecificProjection(x_v)
6. For l = 1 ... L_g:
      For each relation r:
          α_vu^r ← RelationTemporalAttention(h_v, h_u, r, Δt_vu)
          N_topk ← select_topk_neighbors(α_vu^r)
          m_v^r ← aggregate(N_topk, α_vu^r)
      g_v ← RelationLevelAttention({m_v^r})
7. For each target process node v:
      s_bar_v ← LinearNorm(s_v)
      g_bar_v ← LinearNorm(g_v)
      z_v ← sigmoid(W_z [s_bar_v || g_bar_v || |s_bar_v-g_bar_v| || s_bar_v⊙g_bar_v] + b_z)
      f_v ← z_v⊙s_bar_v + (1-z_v)⊙g_bar_v
      y_hat_v ← sigmoid(MLP(f_v))
8. Compute BCE loss on post-forward sampled process nodes
9. Backpropagate and update parameters
```

---

## 10. PyTorch 模块级实现草图

下面是核心门控模块的实现思路，便于后续直接嵌入 MalSnif 代码：

```python
import torch
import torch.nn as nn

class AdaptiveGatedFusion(nn.Module):
    def __init__(self, sem_dim, graph_dim, hidden_dim, ctx_dim=0, dropout=0.1):
        super().__init__()
        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        gate_in_dim = hidden_dim * 4 + ctx_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, sem_feat, graph_feat, ctx_feat=None):
        s = self.sem_proj(sem_feat)
        g = self.graph_proj(graph_feat)
        gate_input = [s, g, torch.abs(s - g), s * g]
        if ctx_feat is not None:
            gate_input.append(ctx_feat)
        q = torch.cat(gate_input, dim=-1)
        z = self.gate(q)
        fused = z * s + (1.0 - z) * g
        return fused, z
```

分类器：

```python
class NodeClassifier(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, fused_feat):
        return torch.sigmoid(self.mlp(fused_feat)).squeeze(-1)
```

---

## 11. 实验与消融设计

### 11.1 主实验

沿用 MalSnif 论文中的数据集设置：

| 数据集 | 用途 |
|---|---|
| LotL | 检测 Living-off-the-Land / Fileless 类攻击 |
| DARPA | 检测 APT 攻击 |
| StreamSpot | 图级或场景级泛化验证，可转化为节点级辅助实验 |

核心指标：

```text
Accuracy, Precision, Recall, F1, AUC, FPR, FNR, Detection Latency, GPU Memory
```

对于安全检测，建议重点汇报：

```text
Precision, Recall, F1, FPR, Time-to-Detect
```

因为 Accuracy 在严重类别不平衡下解释力不足。

### 11.2 消融实验

| 实验编号 | 模型 | 目的 |
|---|---|---|
| A0 | MalSnif 原模型 | 基线 |
| A1 | MalSnif + MCBG | 验证语义模块贡献 |
| A2 | MalSnif + HGAN | 验证异构图建模贡献 |
| A3 | MCBG + HGAN + 静态拼接 | 验证方案一主体贡献 |
| A4 | MCBG + HGAN + 自适应门控 | 验证门控融合贡献 |
| A5 | A4 去掉时间偏置 | 验证轻量时间建模贡献 |
| A6 | A4 使用硬裁剪 | 验证软修剪贡献 |
| A7 | A4 标量门控而非向量门控 | 验证向量门控必要性 |

最终论文中可以将 A4 作为主模型。

### 11.3 门控解释实验

对不同攻击类型统计：

$$
GateSem = \frac{1}{|V_m|}\sum_{v \in V_m}mean(z_v)
$$

其中 `V_m` 是检测到的恶意节点集合。

建议可视化：

1. Fileless/LoTL 场景下的 `GateSem` 分布；
2. 横向移动场景下的 `GateStruct` 分布；
3. 同一攻击链中不同阶段的门控变化；
4. 被注意力聚焦的 Top-k 事件、Top-k 关系和 Top-k 邻居。

这能把“门控融合”从一个工程模块提升为具有解释性的学术贡献。

---

## 12. 复杂度分析

设：

- `N` 为节点数；
- `M` 为边数；
- `L` 为进程事件序列长度；
- `d` 为隐藏维度；
- `H` 为注意力头数；
- `K` 为每种关系保留的 Top-k 邻居数。

### 12.1 语义分支复杂度

原 MalSnif 主要 RNN 复杂度可近似为：

$$
O(Ld^2)
$$

本方案加入 CNN，但池化后 RNN 序列长度变为 $L' < L$：

$$
O(Lkd) + O(L'd^2) + O(HL'd^2)
$$

当 `L'` 明显小于 `L` 时，整体推理延迟可以保持在可接受范围。

### 12.2 图分支复杂度

原 GCN 每层约为：

$$
O(Md + Nd^2)
$$

ST-HGAN 每层约为：

$$
O(HMd + |R|Nd)
$$

若启用 Top-k 软修剪，则实际聚合复杂度变为：

$$
O(HNKd)
$$

其中 `K` 可设置为 10、20 或 30，通过验证集选择。

### 12.3 门控复杂度

门控层复杂度约为：

$$
O(Nd^2)
$$

相对 HGAN 与 BiGRU 开销很小，几乎不是瓶颈。

### 12.4 工程开销评估

推荐参数起点：

| 参数 | 推荐值 |
|---|---:|
| Word2Vec 维度 | 64 或 128 |
| CNN kernel size | 2, 3, 5 |
| CNN channel | 64 |
| BiGRU hidden | 128 |
| HGAN hidden | 128 |
| HGAN heads | 4 |
| HGAN layers | 2 |
| Top-k neighbors per relation | 10~20 |
| Gate hidden | 128 |
| Dropout | 0.1~0.3 |

第一版不建议把层数堆得太深，2 层 HGAN 足以覆盖多数局部因果关系，也能降低过平滑风险。

---

## 13. 预期创新点表述

论文或开题中可以将贡献写成以下三点：

### Contribution 1: Local-global semantic encoder

提出 MCBG 语义编码器，将多尺度 1D-CNN、BiGRU 和多头注意力结合，用于从审计日志中同时捕获局部攻击载荷和全局行为上下文，缓解原 MalSnif 在长日志序列中的关键特征稀释问题。

### Contribution 2: Type- and relation-aware provenance graph learning

将 MalSnif 的同构 GCN 改造为 ST-HGAN，通过节点类型、关系类型和轻量时间偏置执行异构消息传递，使模型能够区分文件、注册表、网络和进程行为的不同威胁意义。

### Contribution 3: Adaptive gated semantic-structural fusion

提出自适应门控融合模块，在节点级别动态调节语义特征与结构特征的贡献比例，使模型能够根据攻击类型自动选择更可靠的证据来源，并提供可解释的门控权重。

---

## 14. 可能的审稿质疑与应对

| 可能质疑 | 风险 | 应对策略 |
|---|---|---|
| 模型是否过于复杂 | Reviewer 可能认为只是堆模块 | 强调每个模块对应 MalSnif 一个明确瓶颈，并用消融证明必要性 |
| HGAN 与已有异构 GNN 是否重复 | 创新性被质疑 | 突出与 MalSnif process/event 图的最小侵入式适配，以及关系时间偏置和门控解释 |
| 门控是否真的有效 | 可能被认为只是普通 fusion | 做 gate distribution、case study、gate ablation |
| 注意力是否能代表解释 | 注意力解释性常被质疑 | 不把注意力等同因果解释，只作为辅助证据；配合反事实删除 Top-k 事件验证 |
| 软修剪是否影响漏报 | 低注意力边可能仍有攻击意义 | 保留原始审计图，不物理删除；只在消息聚合中 Top-k，回溯时仍可访问全图 |
| 保留后置下采样是否仍有局限 | FPR 可能受影响 | 主实验保留以公平对比；附加实验可加入 weighted BCE，但不作为主创新 |

---

## 15. 最终落地版本建议

推荐按两个阶段实现。

### 阶段一：最小可复现改造

```text
MalSnif 原预处理
+ MCBG 替换 GRU/BiLSTM
+ HGAN 替换 GCN
+ 静态拼接融合
```

目标：先验证方案一主体相对 MalSnif 的增益。

### 阶段二：完整创新版

```text
MalSnif 原预处理
+ MCBG
+ ST-HGAN with relation/time bias
+ Soft pruning
+ Adaptive gated fusion
+ Gate explainability analysis
```

目标：形成最终论文主模型。

---

## 16. 建议最终方法小节结构

如果后续写论文，方法章节可以这样组织：

```text
3. Methodology
  3.1 Problem Definition
  3.2 Typed Provenance Graph Construction
  3.3 Multi-head CNN-BiGRU Semantic Encoder
  3.4 Spatio-Temporal Heterogeneous Graph Attention Encoder
  3.5 Adaptive Gated Semantic-Structural Fusion
  3.6 Training Objective and Complexity Analysis
```

实验章节：

```text
4. Evaluation
  4.1 Experimental Setup
  4.2 Overall Detection Performance
  4.3 Ablation Study
  4.4 Gate Interpretability Study
  4.5 Robustness to Noisy Benign Activities
  4.6 Efficiency Analysis
```

---

## 17. 一句话总结

本优化版方案可以概括为：

> 在 MalSnif 的审计日志语义—溯源图结构联合检测框架上，用 MCBG 强化长日志语义抽取，用 ST-HGAN 替代同构 GCN 以建模节点/关系异构性，再用自适应门控融合在节点级动态选择语义证据或结构证据，从而提升对 LoTL、Fileless、横向移动和多阶段 APT 的鲁棒检测能力，同时保持可解释性和可复现实验边界。
