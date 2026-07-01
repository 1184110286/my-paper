# MalSnif E1 TBB-RR Verdict Project

This package focuses on strict, reproducible **TBB-RR** redundancy experiments for MalSnif-style process-node detection.  The current main entries evaluate **TBB-RR: Target-Budget Boundary Redundancy Reduction** against MalSnif's original `prefix_tree` sequence reduction.

TBB-RR is a compression-budgeted one-parameter reducer.  Given a target compression ratio `C*`, it splits each process event sequence into fixed-size consecutive blocks and keeps only the first and last event of each block.  It introduces no new event tokens, uses no prefix tree, and keeps the original event order.

## Main CADETS experiment

```bash
DEVICE=0 bash scripts/run_cadets_tbb_rr_verdict.sh
```

The script runs **E1_eha_only** and compares only:

```text
off
prefix_tree
target_boundary
```

Default seeds:

```text
42 43 44 45 46
```

Default TBB-RR target compression:

```text
TBB_RR_TARGET_COMPRESSION=0.90
```

Quick smoke run:

```bash
RUN_OFF=0 SEEDS="42 43 44" EPOCHS=3 DEVICE=0 bash scripts/run_cadets_tbb_rr_verdict.sh
```

## THEIA experiment

To run the same TBB-RR verdict on DARPA TC THEIA-E3:

```bash
DEVICE=0 bash scripts/run_theia_tbb_rr_verdict.sh
```

This script keeps the same fixed model and paired comparison:

```text
off
prefix_tree
target_boundary
```

Default THEIA inputs:

```text
data/raw/darpa_tc/theia/e3/cdm/ta1-theia-e3-official*.json*
data/raw/darpa_tc/theia/e3/labels/
```

Accepted THEIA label files include `theia.json`, `theia.txt`, `malicious_uuids.txt`, `malicious_paths.txt`, `malicious_event_types.txt`, `malicious_time_ranges.csv`, and `malicious_events.csv`.

Fast smoke run:

```bash
RUN_OFF=0 SEEDS="42 43 44" EPOCHS=3 DEVICE=0 bash scripts/run_theia_tbb_rr_verdict.sh
```

The THEIA script creates an independent graph cache for each redundancy mode and validates that the effective `preprocess_metadata.json` really uses the requested mode.  A run fails if `prefix_tree` or `target_boundary` accidentally reuses an `off` cache.

## Output

CADETS output:

```text
runs/cadets_tbb_rr_verdict_<timestamp>_autostop_win<WINDOW_EVENTS>/experiment/
  run_matrix.tsv
  summary_tbb_rr.csv
  summary_tbb_rr_agg.csv
  paired_vs_prefix.csv
  paired_vs_prefix_agg.csv
  TBB_RR_VERDICT_REPORT.md
```

THEIA output:

```text
runs/theia_tbb_rr_verdict_<timestamp>_autostop_win<WINDOW_EVENTS>/experiment/
  run_matrix.tsv
  summary_theia_tbb_rr.csv
  summary_theia_tbb_rr_agg.csv
  paired_vs_prefix.csv
  paired_vs_prefix_agg.csv
  THEIA_TBB_RR_VERDICT_REPORT.md
```

The upload-friendly single-folder bundle is:

```text
runs/<run_name>/analysis_bundle/collected/
```

The script prints this path as:

```text
[send me] .../analysis_bundle/collected
```

## CADETS data layout

Expected CADETS files:

```text
data/raw/darpa_tc/cadets/e3/cdm/ta1-cadets-e3-official*.json*
data/raw/darpa_tc/cadets/e3/labels/
```

Override with:

```bash
RAW_DIR=/path/to/cdm LABEL_DIR=/path/to/labels DEVICE=0 bash scripts/run_cadets_tbb_rr_verdict.sh
```

## Notes

- `target_boundary` / `tbb_rr` is the TBB-RR mode.
- `prefix_tree` remains the MalSnif Algorithm 1 control.
- TBB-RR is a candidate reducer, not a confirmed improvement until paired deltas satisfy the report's conservative success rule.
- Heavy caches, checkpoints, raw data, and large arrays are excluded from the collected analysis bundle.


## E1-GDTC-MCBG TBB-RR experiment

This package includes a low-coupling semantic encoder replacement for `E1_eha_only`:
`semantic_encoder: gdtc_mcbg`.  To compare the existing E1 MCBG encoder with
the new gated dilated temporal convolution encoder on CADETS and THEIA using
TBB-RR sequence compression, run:

```bash
bash scripts/run_e1_gdtc_tbb_rr_theia_cadets.sh
```

See `docs/e1_gdtc_mcbg_experiment.md` for the design, configuration fields,
and output interpretation.

### E1-RGD-BiGRU-MCBG TBB-RR experiment

Run the paired MCBG vs RGD-BiGRU-MCBG E1_eha_only experiment on CADETS and THEIA:

```bash
DEVICE=1 EVAL_DEVICE=1 SEEDS="42 43 44" EPOCHS=5 \
  bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets.sh
```

Outputs are placed under `runs/e1_rgd_bigru_tbb_rr_theia_cadets_<timestamp>/`.  Send the generated `analysis_bundle/` directory for result inspection; it contains key logs, metrics, configs, reports, and plots while excluding graph caches, checkpoints, raw data, and large arrays.

## Rigorous E1-RGD-BiGRU-MCBG TBB-RR validation

Run the stricter CADETS/THEIA paired comparison on GPU 1:

```bash
DEVICE=1 EVAL_DEVICE=1 \
  bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh
```

This compares `E1_eha_only_mcbg` and `E1_eha_only_rgd_bigru` with TBB-RR
(`redundancy_mode=target_boundary`) on CADETS and THEIA. The default rigorous
setting uses `calib12m`, five paired seeds, 15 epochs, fixed threshold policy,
and an analysis bundle for follow-up diagnosis.

Useful variants:

```bash
EXPERIMENT_LEVEL=smoke DEVICE=1 EVAL_DEVICE=1 \
  bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh

EXPERIMENT_LEVEL=full DEVICE=1 EVAL_DEVICE=1 \
  bash scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh
```

The output directory contains `analysis_bundle/`; send that folder for result
analysis because it includes the key logs, configs, metrics and plots while
excluding graph caches, checkpoints and raw data.

## Paper method: RGT-HGIDS

This project now packages the final paper-oriented method as **RGT-HGIDS**:

```text
Redundancy-aware Gated Temporal Heterogeneous Graph Intrusion Detection System
中文名：冗余感知门控时序异构图入侵检测框架
Pipeline: TBB-RR + RGD-BiGRU-MCBG + ST-HGAN + EHA
```

Recommended paper experiment entrypoint:

```bash
DEVICE=1 EVAL_DEVICE=1 bash scripts/run_rgt_hgids_rigorous.sh
```

Quick sanity-check entrypoint:

```bash
DEVICE=1 EVAL_DEVICE=1 bash scripts/run_rgt_hgids_quick.sh
```

Key documentation:

```text
docs/rgt_hgids_method_overview.md
docs/rgt_hgids_paper_section_draft.md
docs/rgt_hgids_experiment_protocol.md
docs/rgt_hgids_references.md
docs/paper_assets/rgt_hgids_mermaid_flow.mmd
configs/method_profiles/rgt_hgids_balanced.env
```

Use `analysis_bundle/` generated by the rigorous script for follow-up result analysis.
