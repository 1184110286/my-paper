# Rigorous E1-RGD-BiGRU-MCBG TBB-RR experiment

This protocol is the stricter follow-up to the quick RGD-BiGRU-MCBG validation.
It keeps the same experimental boundary but increases data coverage, seeds,
epochs, and reporting rigor.

## Main command

```bash
DEVICE=1 EVAL_DEVICE=1 \
  bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh
```

The default `EXPERIMENT_LEVEL=rigorous` uses:

```text
CADETS_EA_PRESET=calib12m
SEEDS="42 43 44 45 46"
EPOCHS=15
PATIENCE=8
VAL_EVERY=1
MAX_EVENTS_PER_NODE=64
REDUNDANCY_MODE=target_boundary
TBB_RR_TARGET_COMPRESSION=0.90
```

Quick check:

```bash
EXPERIMENT_LEVEL=smoke DEVICE=1 EVAL_DEVICE=1 \
  bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh
```

Larger run:

```bash
EXPERIMENT_LEVEL=full DEVICE=1 EVAL_DEVICE=1 \
  bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh
```

## Comparison

The default comparison is intentionally narrow and paired:

```text
mcbg      -> semantic_encoder=mcbg
rgd_bigru -> semantic_encoder=rgd_bigru_mcbg
```

The following controls are fixed:

```text
model_variant=ea_st_hgan_mcbg
RUN_E1=1 only
EHA enabled
ETS/EAW disabled
ST-HGAN unchanged
redundancy_mode=target_boundary (TBB-RR)
node_scope=process
threshold_strategy=val_f1_min_recall
threshold_min_recall=0.95
model_selection_metric=val_average_precision
```

## Output

The run writes:

```text
runs/e1_rgd_bigru_tbb_rr_theia_cadets_RIGOROUS_<level>_<timestamp>/
  ENVIRONMENT.txt
  E1_RGD_BIGRU_TBB_RR_RIGOROUS_EXPERIMENT_PLAN.md
  E1_RGD_BIGRU_TBB_RR_RIGOROUS_SUMMARY.csv
  E1_RGD_BIGRU_TBB_RR_RIGOROUS_AGG.csv
  E1_RGD_BIGRU_TBB_RR_RIGOROUS_PAIRED_DELTAS.csv
  E1_RGD_BIGRU_TBB_RR_RIGOROUS_PAIRED_AGG.csv
  E1_RGD_BIGRU_TBB_RR_RIGOROUS_REPORT.md
  analysis_bundle/
```

Send `analysis_bundle/` for follow-up analysis. It contains the files most useful
for diagnosis: environment, plan, aggregate metrics, paired deltas, paired
verdicts, per-seed logs, config files, compact metrics, train summaries,
run-analysis JSON, `history.png`, and `scores_test.png`. It excludes graph caches,
checkpoints, raw data, model weights, and large arrays.

## Interpretation

Use paired seed deltas rather than comparing unpaired means:

```text
delta = candidate - mcbg
```

Primary metrics:

```text
F1, Recall, Precision, MCC, Average Precision, ROC-AUC, FP, FN,
train_seconds, cuda_peak_allocated_mb
```

Report verdicts use conservative practical thresholds:

```text
non-inferior if mean_delta_f1 >= -0.002 and mean_delta_recall >= -0.005
positive if mean_delta_f1 >= 0.001, mean_delta_recall >= 0, and paired F1 wins are a majority
```

These thresholds are not mathematical proof; they are practical decision rules for
whether RGD-BiGRU-MCBG should become the next default E1_eha_only semantic encoder.
