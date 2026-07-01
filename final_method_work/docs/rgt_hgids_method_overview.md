# RGT-HGIDS: Redundancy-aware Gated Temporal Heterogeneous Graph Intrusion Detection

> **推荐论文方法名**：**RGT-HGIDS**  
> **英文全称**：**Redundancy-aware Gated Temporal Heterogeneous Graph Intrusion Detection System**  
> **中文名**：**冗余感知门控时序异构图入侵检测框架**  
> **工程别名**：`rgt_hgids`  
> **核心组合**：`TBB-RR + RGD-BiGRU-MCBG + ST-HGAN + EHA`

## 1. 定位

RGT-HGIDS 是以 MalSnif 风格的审计日志溯源图检测为基础、面向 DARPA TC 过程节点级检测的完整方法。它保留 MalSnif 的“审计日志语义 + 溯源图结构”主线，同时将最终方法收敛到一个可解释、低耦合、实验表现稳定的组合：

```text
格式化审计日志
  -> TBB-RR 序列压缩
  -> 类型/关系/时间增强溯源图
  -> RGD-BiGRU-MCBG 语义编码
  -> ST-HGAN 结构聚合
  -> EHA hop 自适应融合
  -> 节点级恶意概率输出
```

与早期 AGF-ST-HGAN-MCBG 方案相比，RGT-HGIDS 不再引入额外的自适应门控融合分支，而是采用已经在项目中验证更稳的 `E1_eha_only` 路线：只启用 EHA，不启用 ETS/EAW。这使方法边界更清晰，论文叙述也更集中。

## 2. 与 MalSnif 的关系

MalSnif 的核心思想是从审计日志中提取进程事件序列与进程/事件关系，构建带语义信息的溯源图，再用 NLP 序列模型和 GCN 进行检测。RGT-HGIDS 继承这个主线，但针对三个问题做最小必要改造：

| 环节 | MalSnif/早期基线 | RGT-HGIDS 改造 |
|---|---|---|
| 长序列压缩 | prefix-tree redundancy reduction | TBB-RR：目标预算边界压缩，显式控制压缩率 |
| 语义编码 | GRU/BiLSTM 或普通 MCBG | RGD-BiGRU-MCBG：残差门控膨胀卷积 + BiGRU + 注意力池化 |
| 图结构建模 | GCN 或基础异构聚合 | ST-HGAN + EHA：关系/时间感知聚合与 hop 自适应 |
| 实验粒度 | 节点级/图级均可 | 当前论文主张 process-node-level detection |

## 3. 方法名称选择理由

`RGT-HGIDS` 的四个关键词分别对应完整方法的四个论文贡献点：

- **Redundancy-aware**：使用 TBB-RR 在预处理阶段控制过程事件序列冗余；
- **Gated Temporal**：使用残差门控膨胀卷积和 BiGRU 建模局部事件片段与顺序依赖；
- **Heterogeneous Graph**：使用 ST-HGAN 区分事件关系、时间偏置和异构邻接；
- **Intrusion Detection System**：保持 HIDS 审计日志入侵检测任务定位。

建议论文中首次出现写法：

> We propose **RGT-HGIDS**, a Redundancy-aware Gated Temporal Heterogeneous Graph Intrusion Detection System for audit-log-based host intrusion detection.

中文写法：

> 本文提出 **RGT-HGIDS**，一种面向审计日志的冗余感知门控时序异构图入侵检测框架。

## 4. 完整流程

### 4.1 日志解析与事件序列构造

原始审计事件记为：

\[
e = \langle SrcId, SrcType, DstId, DstType, EdgeType, Time, Tag \rangle
\]

对每个进程节点 \(v\)，收集按时间排序的事件序列：

\[
S_v = [e_1,e_2,\ldots,e_L]
\]

事件经过 token 化与 Word2Vec/embedding 后得到：

\[
x_t = \frac{1}{m_t}\sum_{j=1}^{m_t} Emb(w_{t,j}), \quad X_v=[x_1,\ldots,x_L]\in\mathbb{R}^{L\times d_w}
\]

### 4.2 TBB-RR 序列压缩

TBB-RR，即 **Target-Budget Boundary Redundancy Reduction**，用于替代默认 prefix-tree 压缩。设目标压缩率为 \(\rho\)，原序列长度为 \(L\)，目标保留预算为：

\[
B = \max(1, \lceil (1-\rho)L \rceil)
\]

TBB-RR 的思想是优先保留序列边界与目标预算内的代表性事件片段，并为保留事件产生权重 \(w_t\)。压缩输出为：

\[
\tilde{S}_v = [(e_{i_1},w_{i_1}),\ldots,(e_{i_B},w_{i_B})]
\]

其中 \(w_t\) 可作为后续语义池化的证据权重：

\[
a_t = q^\top \tanh(W h_t+b) + \beta\log(w_t+\epsilon)
\]

\[
\alpha_t = softmax(a_t)
\]

### 4.3 RGD-BiGRU-MCBG 语义编码

输入投影：

\[
H^{(0)} = LayerNorm(X_v W_x+b_x)
\]

残差门控膨胀卷积块：

\[
A^{(l)} = DWConv_{k,r_l}(H^{(l-1)})W_A^{(l)} + b_A^{(l)}
\]

\[
G^{(l)} = \sigma(DWConv_{k,r_l}(H^{(l-1)})W_G^{(l)}+b_G^{(l)})
\]

\[
\tilde{H}^{(l)} = GELU(A^{(l)})\odot G^{(l)}
\]

\[
H^{(l)} = LayerNorm(H^{(l-1)} + \lambda_l Dropout(\tilde{H}^{(l)}W_O^{(l)}))
\]

默认使用 \(k=3\)、\(r=[1,2]\)、\(\lambda_l=0.1\)。小初始化残差系数让新卷积分支具备“可退化”能力：当某个数据集不需要卷积增强时，模型可近似回到 BiGRU 主路径。

BiGRU 顺序建模：

\[
Y_t=[\overrightarrow{GRU}(H^{(L_c)}_t)\parallel \overleftarrow{GRU}(H^{(L_c)}_t)]
\]

残差对齐：

\[
\bar{Y}_t = LayerNorm(W_yY_t + W_sH^{(L_c)}_t)
\]

注意力池化：

\[
s_v=\sum_t \alpha_t \bar{Y}_t
\]

其中 \(s_v\) 是进程节点的语义表示。

### 4.4 ST-HGAN 结构编码

溯源图定义为：

\[
G=(V,E,\phi,\psi,\tau)
\]

其中 \(\phi(v)\) 为节点类型，\(\psi(e)\) 为关系类型，\(\tau(e)\) 为时间信息。

节点类型投影：

\[
h_v^{(0)} = W_{\phi(v)}x_v+b_{\phi(v)}
\]

关系/时间感知注意力：

\[
e_{vu}^{r,h}=LeakyReLU\left(a_{r,h}^{\top}[W_{r,h}h_v \parallel W_{r,h}h_u \parallel e_r \parallel \phi(\Delta t_{uv})]\right)
\]

\[
\alpha_{vu}^{r,h}=softmax_{u\in\mathcal{N}_v^r}(e_{vu}^{r,h})
\]

\[
m_v^{r,h}=\sum_{u\in\mathcal{N}_v^r}\alpha_{vu}^{r,h}W_{r,h}h_u
\]

多关系聚合输出结构表示 \(g_v\)。

### 4.5 EHA hop 自适应

RGT-HGIDS 采用 EHA-only 路线。对不同 hop 的结构表示 \(g_v^{(0)}, g_v^{(1)}, g_v^{(2)}\)，学习 hop 权重：

\[
\gamma_{v,k}=softmax_k(q_h^\top\tanh(W_h g_v^{(k)}+b_h))
\]

\[
g_v=\sum_k\gamma_{v,k}g_v^{(k)}
\]

实验中 RGD-BiGRU-MCBG 往往会让 EHA 更偏向 hop0/hop1，说明语义分支增强后，图分支对远跳补偿的依赖降低。

### 4.6 节点级分类

最终对进程节点输出恶意概率：

\[
\hat{y}_v=\sigma(MLP(g_v))
\]

训练损失保持节点级二分类：

\[
\mathcal{L}_{BCE}=-\frac{1}{|\mathcal{V}_{train}|}\sum_{v}\left[y_v\log\hat{y}_v+(1-y_v)\log(1-\hat{y}_v)\right]
\]

## 5. 推荐论文贡献表述

1. 提出 TBB-RR，对过程事件序列进行目标预算边界压缩，在保留边界行为和关键事件权重的同时降低冗余。
2. 提出 RGD-BiGRU-MCBG，以残差门控膨胀卷积增强 MCBG 的局部攻击片段建模能力，并保留 BiGRU 对稀疏正例场景的顺序兜底。
3. 将增强语义编码接入 ST-HGAN + EHA，形成冗余感知、语义增强、异构结构自适应的节点级 HIDS 框架。
4. 在 CADETS/THEIA 上使用 paired seeds 验证 RGT-HGIDS 相对原 MCBG 的非劣性与召回提升。

## 6. 运行入口

快速验证：

```bash
DEVICE=1 EVAL_DEVICE=1 bash scripts/run_rgt_hgids_quick.sh
```

论文主实验建议：

```bash
DEVICE=1 EVAL_DEVICE=1 bash scripts/run_rgt_hgids_rigorous.sh
```

底层仍调用：

```text
scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets.sh
scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh
```

## 7. 实验报告口径

主表建议报告：

```text
Precision, Recall, F1, MCC, Average Precision, ROC-AUC, FP, FN
```

安全检测中应强调 Recall/FN 与 MCC，不建议只报告 Accuracy。AP 用于观察不平衡数据下的排序质量，固定阈值 F1/Recall 用于观察实际告警效果。
