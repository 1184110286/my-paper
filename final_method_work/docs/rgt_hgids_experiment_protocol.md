# RGT-HGIDS 实验协议

## 主实验

目的：验证 RGT-HGIDS 中的 RGD-BiGRU-MCBG 是否优于原 E1_eha_only MCBG，同时固定 TBB-RR、ST-HGAN、EHA、数据集与 seed。

```bash
DEVICE=1 EVAL_DEVICE=1 bash scripts/run_rgt_hgids_rigorous.sh
```

默认配置：

```text
RIGOR_LEVEL=balanced
CADETS_EA_PRESET=calib8m
SEEDS="42 43 44"
EPOCHS=8
VAL_EVERY=1
RUN_DATASETS="cadets theia"
RUN_ENCODERS="mcbg rgd_bigru"
REDUNDANCY_MODE=target_boundary
TBB_RR_TARGET_COMPRESSION=0.90
```

## 输出

结果根目录：

```text
runs/rgt_hgids_rigorous_<timestamp>/
```

需要发给分析者的目录：

```text
analysis_bundle/
```

其中包含日志、配置、指标、图像、post-check、paired deltas 和 summary。

## 成功判据

主判据：

```text
mean F1_RGT >= mean F1_MCBG - 0.002
mean Recall_RGT >= mean Recall_MCBG - 0.005
mean FN_RGT <= mean FN_MCBG + 0.5
```

强正向判据：

```text
paired seeds 中至少 2/3 的 F1 或 Recall 提升；
FN 平均下降；
AP/MCC 不出现明显退化。
```

## 消融实验建议

| 实验 | 目的 |
|---|---|
| MCBG + TBB-RR | 原语义编码器基线 |
| RGD-BiGRU-MCBG + TBB-RR | 主方法 |
| RGD-BiGRU-MCBG + prefix_tree | 分离 TBB-RR 影响 |
| MCBG + prefix_tree | MalSnif 风格压缩基线 |
| RGD-BiGRU-MCBG without event-weight pooling | 验证 TBB-RR 权重侧通道 |

## 图像分析建议

重点看：

- `history.png`：loss、val_f1、val_mcc、val_average_precision 是否稳定；
- `scores_test.png`：正负分数是否分离，低分正例是否减少；
- EHA hop 权重：RGD 是否让 hop0/hop1 增强、hop2 依赖下降。
