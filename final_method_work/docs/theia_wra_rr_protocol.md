# THEIA WRA-RR verdict protocol

This protocol evaluates **WRA-RR: Winnowing Representative Anchor Redundancy Reduction** on DARPA TC THEIA-E3 while keeping the downstream model fixed to `E1_eha_only`.

## Goal

Compare only three sequence-reduction modes on the same strict-qualified THEIA event prefix:

```text
off
prefix_tree
winnowing_anchor
```

`prefix_tree` is the MalSnif Algorithm 1 control. `winnowing_anchor` is the one-parameter WRA-RR candidate.

## Data layout

Place THEIA-E3 CDM shards under:

```text
data/raw/darpa_tc/theia/e3/cdm/ta1-theia-e3-official*.json*
```

Place at least one label source under:

```text
data/raw/darpa_tc/theia/e3/labels/
```

Accepted label sources include:

```text
theia.json
theia.txt
malicious_uuids.txt
malicious_paths.txt
malicious_event_types.txt
malicious_time_ranges.csv
malicious_events.csv
_raw/*.json*
```

Run `bash scripts/check_theia_data_layout.sh` to write a manifest and validate basic placement.

## One-key run

```bash
DEVICE=0 bash scripts/run_theia_wra_rr_verdict.sh
```

Default parameters:

```text
WRA_RR_WINDOW=11
WINDOW_EVENTS=200000
STRICT_MAX_EVENTS_CAP=12000000
SEEDS="42 43 44 45 46"
```

Fast smoke test:

```bash
RUN_OFF=0 SEEDS="42 43 44" EPOCHS=3 DEVICE=0 bash scripts/run_theia_wra_rr_verdict.sh
```


## Cache isolation and validation

Each redundancy mode uses a separate graph-cache root:

```text
experiment/<mode>/cache/<mode>_events<MAX_EVENTS>_win<WINDOW_EVENTS>_mode<MODE>_wra<WRA_RR_WINDOW>_gs<GRAPH_SIMPLIFY_MODE>/
```

The script exports this path through `CADETS_CACHE_ROOT`, because the generic child runner uses `CADETS_CACHE_ROOT` for graph-cache reuse. After each mode finishes, the wrapper reads:

```text
<cache_root>/analysis/preprocess/metadata.json
```

and fails the run unless:

```text
config.redundancy_mode == requested mode
stats[*].redundancy_mode == requested mode
off reduction ratio == 0
prefix_tree / winnowing_anchor reduction ratio > 0
winnowing_anchor wra_rr_window == requested WRA_RR_WINDOW
```

The validation result is saved as `experiment/<mode>/CACHE_MODE_VALIDATION.json` and copied into the final collected bundle. This prevents the previous failure mode where all three THEIA experiments silently reused the same off-mode cache.

## Output

The outer summary is written to:

```text
runs/theia_wra_rr_verdict_<timestamp>_autostop_win<WINDOW_EVENTS>/experiment/
```

The upload-friendly analysis folder is:

```text
runs/theia_wra_rr_verdict_<timestamp>_autostop_win<WINDOW_EVENTS>/analysis_bundle/collected/
```

The script prints this folder as `[send me] .../analysis_bundle/collected`.

## Decision rule

A WRA-RR positive signal requires all of the following against `prefix_tree`:

```text
mean delta F1  >= +0.003
mean delta MCC >= +0.003
mean delta Recall >= -0.001
F1 wins on at least 60% of paired seeds
compression ratio within 5 percentage points of prefix_tree
```

Otherwise the result remains `tie_or_inconclusive` or `negative_or_recall_risk`.
