# Changelog

## v4.11.0-tbb-rr-verdict

- Added TBB-RR (`target_boundary`, `tbb_rr`, `tbb`, `target_budget`, `target_budget_boundary`) as a target-budget block-boundary reducer.
- Added `tbb_rr_target_compression` config option, defaulting to `0.90`.
- Added CADETS and THEIA one-key verdict scripts that compare only `off`, `prefix_tree`, and `target_boundary` under E1_eha_only.
- Added per-mode cache validation for the THEIA TBB-RR script to prevent reduced modes from reusing off-mode graph caches.
- Added TBB-RR protocol documentation and tests.

## v4.10.1-theia-wra-per-mode-cache-fix

- Fixed THEIA WRA-RR cache isolation: `run_theia_wra_rr_verdict.sh` now passes a distinct per-mode `CADETS_CACHE_ROOT` to the generic child runner for `off`, `prefix_tree`, and `winnowing_anchor`.
- Added post-run cache validation for every mode via `CACHE_MODE_VALIDATION.json`; the script fails if effective `preprocess_metadata.json` reports the wrong `redundancy_mode`, near-zero reduction for reduced modes, or a mismatched WRA window.
- Added the cache root to `run_matrix.tsv` so analysis can trace which graph cache produced each result.
- Updated THEIA protocol and README to document the per-mode cache policy and validation checks.

## v4.10.0-theia-wra-rr-verdict

- Added `scripts/run_theia_wra_rr_verdict.sh` for a THEIA-E3 E1_eha_only paired comparison among `off`, `prefix_tree`, and `winnowing_anchor` only.
- Added `scripts/check_theia_data_layout.sh` for THEIA CDM/label placement validation.
- Added `scripts/collect_theia_wra_rr_analysis_bundle.sh` to flatten THEIA WRA-RR results into `analysis_bundle/collected/`.
- Added `docs/theia_wra_rr_protocol.md` and updated README with THEIA commands, defaults, and upload instructions.
- Extended the generic CDM layout checker so the existing child runner can accept THEIA label files when `RAW_DIR`/`LABEL_DIR` point to THEIA.

## v4.9.0-wra-rr-verdict

- Added WRA-RR (`winnowing_anchor`, `wra_rr`, `wra`, `representative_anchor`, `winnowing`) as a one-parameter local representative-anchor reducer.
- Added `wra_rr_window` config option, defaulting to `11` to target a compression ratio close to the historical prefix_tree level.
- Added `scripts/run_cadets_wra_rr_verdict.sh` for E1_eha_only paired comparison among `off`, `prefix_tree`, and `winnowing_anchor` only.
- Added `scripts/collect_wra_rr_analysis_bundle.sh`, which always flattens analysis-critical metrics, logs, configs, plots, and reports into `analysis_bundle/collected/`.
- Added WRA-RR tests and protocol documentation.

## E1-RGD-BiGRU-MCBG TBB-RR experiment add-on

- Added `rgd_bigru_mcbg` semantic encoder as a low-coupling E1_eha_only drop-in.
- Added residual gated dilated CNN blocks with small initial residual scale before BiGRU.
- Preserved BiGRU and attention pooling to reduce recall risk on sparse-positive datasets.
- Added one-key paired CADETS/THEIA TBB-RR experiment script:
  `scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets.sh`.
- Added robust top-level summary/report generation and an `analysis_bundle/` aggregation directory for logs, metrics, configs, plots, and reports.

## E1-RGD-BiGRU-MCBG rigorous experiment add-on

- Added `scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh` for stricter paired CADETS/THEIA validation.
- Default rigorous protocol uses `calib12m`, five paired seeds, more epochs, TBB-RR, and fixed E1_eha_only controls.
- Added deterministic-mode environment support via `MALSNIF_DETERMINISTIC=1` in `malsnif/utils/seed.py`.
- Added robust top-level reporting: summary, aggregate means/stds, paired deltas, exact sign-test p-values, bootstrap confidence intervals, and practical non-inferiority verdicts.
- Added an expanded `analysis_bundle/` with environment, plan, reports, per-seed logs/metrics/configs/plots, and a `key_files_flat/` directory for easy upload.

## RGT-HGIDS paper-method packaging

- Named the final method **RGT-HGIDS**: Redundancy-aware Gated Temporal Heterogeneous Graph Intrusion Detection System.
- Added paper-facing method documentation covering the full pipeline: TBB-RR, RGD-BiGRU-MCBG, ST-HGAN, EHA, classifier, formulas, experiment protocol, and reporting criteria.
- Added alias entrypoints:
  - `scripts/run_rgt_hgids_rigorous.sh`
  - `scripts/run_rgt_hgids_quick.sh`
- Added recommended experiment profile: `configs/method_profiles/rgt_hgids_balanced.env`.
- Added Mermaid flow diagram: `docs/paper_assets/rgt_hgids_mermaid_flow.mmd`.
