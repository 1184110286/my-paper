# RGT-HGIDS 论文方法章节草稿

## Method Overview

We propose **RGT-HGIDS**, a Redundancy-aware Gated Temporal Heterogeneous Graph Intrusion Detection System for audit-log-based host intrusion detection. Given formatted audit logs, RGT-HGIDS first constructs process-centric event sequences and provenance graphs. It then compresses redundant process event sequences with TBB-RR, extracts semantic representations with RGD-BiGRU-MCBG, models heterogeneous provenance dependencies with ST-HGAN, and adaptively aggregates multi-hop structural contexts with EHA. The final detector performs process-node-level maliciousness prediction.

## Processing Pipeline

For each audit event, we use the normalized format

\[
e=\langle SrcId,SrcType,DstId,DstType,EdgeType,Time,Tag\rangle.
\]

For a process node \(v\), the chronological event sequence is denoted as

\[
S_v=[e_1,e_2,\ldots,e_L].
\]

TBB-RR compresses \(S_v\) into a budgeted sequence \(\tilde{S}_v\) while producing event weights. The sequence is embedded and passed to RGD-BiGRU-MCBG. The residual gated dilated convolution block is defined as

\[
H^{(l)}=LayerNorm(H^{(l-1)}+\lambda_l Dropout((GELU(A^{(l)})\odot G^{(l)})W_O^{(l)})),
\]

where \(A^{(l)}\) is the candidate temporal feature, \(G^{(l)}\) is the sigmoid gate, and \(\lambda_l\) is a learnable residual scale initialized to a small value. The output is further encoded by BiGRU and pooled by attention, producing the semantic representation \(s_v\).

The structural branch uses ST-HGAN to aggregate relation-aware and time-aware provenance contexts. Given relation \(r\), attention head \(h\), and neighbor \(u\), the attention logit is

\[
e_{vu}^{r,h}=LeakyReLU(a_{r,h}^{\top}[W_{r,h}h_v\parallel W_{r,h}h_u\parallel e_r\parallel \phi(\Delta t_{uv})]).
\]

EHA then learns hop weights and outputs the final graph representation \(g_v\). The maliciousness score is

\[
\hat{y}_v=\sigma(MLP(g_v)).
\]

## Why RGD-BiGRU-MCBG

The semantic encoder is deliberately conservative. A purely convolutional replacement can improve parallel local pattern extraction but may reduce recall on sparse-positive datasets. RGD-BiGRU-MCBG therefore keeps BiGRU as the sequential fallback and only strengthens the CNN front-end with residual gated dilated blocks. This design improves local attack-fragment sensitivity while preserving temporal dependency modeling.

## Why TBB-RR

Audit event sequences are highly redundant. TBB-RR provides a one-parameter compression-controlled reducer. Unlike ad-hoc truncation, it explicitly targets a compression budget and keeps an event-weight side channel. This side channel can bias attention pooling toward retained representative events.

## Evaluation Protocol

The recommended primary experiment compares the original MCBG encoder against RGD-BiGRU-MCBG under the same graph cache, same TBB-RR configuration, same dataset split, and paired random seeds. The script is:

```bash
DEVICE=1 EVAL_DEVICE=1 bash scripts/run_rgt_hgids_rigorous.sh
```

Primary metrics are F1, Recall, Precision, MCC, Average Precision, ROC-AUC, FP, and FN. The strongest claim should be based on paired seed deltas rather than a single run.
