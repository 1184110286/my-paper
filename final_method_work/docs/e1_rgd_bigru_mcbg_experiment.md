# E1-RGD-BiGRU-MCBG with TBB-RR

## Purpose

This experiment adds `rgd_bigru_mcbg` as a low-coupling semantic encoder for the existing `E1_eha_only` configuration.  It keeps the graph branch, ST-HGAN, EHA-only adaptivity, edge-weight flow, TBB-RR sequence compression, node-level loss, and evaluation pipeline unchanged.

## Encoder

`RGDBiGRUMCBGEncoder` keeps the original MCBG interface:

```text
forward_nested(nested_ids, max_events, max_tokens, device, nested_weights) -> [num_items, behavior_dim]
```

The internal flow is:

```text
Word2Vec event mean
  -> residual gated dilated CNN blocks
  -> residual BiGRU
  -> multi-head attention + attention pooling
```

The key design goal is to improve local audit-event fragment modeling without fully removing BiGRU.  This is more conservative than the earlier `gdtc_mcbg` encoder and is intended to reduce the chance of losing recall on sparse-positive datasets such as THEIA.

## Main script

Run both CADETS and THEIA with paired seeds:

```bash
DEVICE=1 EVAL_DEVICE=1 SEEDS="42 43 44" EPOCHS=5 \
  bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets.sh
```

Quick smoke run:

```bash
DEVICE=1 EVAL_DEVICE=1 CADETS_EA_PRESET=smoke SEEDS="42" EPOCHS=1 \
  bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets.sh
```

Default comparison:

```text
E1_eha_only_mcbg      : semantic_encoder=mcbg
E1_eha_only_rgd_bigru : semantic_encoder=rgd_bigru_mcbg
```

Default RGD parameters:

```text
RGD_KERNEL_SIZE=3
RGD_DILATIONS=1,2
RGD_DROPOUT=0.2
RGD_RESIDUAL_SCALE_INIT=0.1
RGD_DEPTHWISE_SEPARABLE=1
RGD_USE_EVENT_WEIGHT_POOLING=1
```

## Outputs

The script writes:

```text
runs/e1_rgd_bigru_tbb_rr_theia_cadets_<timestamp>/
  E1_RGD_BIGRU_TBB_RR_EXPERIMENT_PLAN.md
  E1_RGD_BIGRU_TBB_RR_SUMMARY.csv
  E1_RGD_BIGRU_TBB_RR_REPORT.md
  analysis_bundle/
```

`analysis_bundle/` is the directory to send for follow-up analysis.  It includes the top-level plan/summary/report and per-dataset/per-encoder seed bundles containing key logs, metrics, configs, reports, and plots.  It intentionally excludes graph caches, checkpoints, raw data, and large arrays.

## Interpretation

Use paired seed deltas in `E1_RGD_BIGRU_TBB_RR_REPORT.md`:

```text
delta = rgd_bigru - mcbg
```

Primary metrics:

```text
F1, Recall, Precision, MCC, Average Precision, ROC-AUC, FP, FN
```

A conservative success criterion is that `rgd_bigru` should not reduce mean F1/Recall relative to `mcbg`, while improving or stabilizing CADETS performance.
