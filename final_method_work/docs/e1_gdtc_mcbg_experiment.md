# E1-GDTC-MCBG integration and TBB-RR experiment

## What changed

This revision adds `semantic_encoder: gdtc_mcbg` as a low-coupling drop-in replacement for the existing MCBG semantic branch.  It keeps the same public encoder API:

```python
forward_nested(nested_ids, max_events, max_tokens, device, nested_weights=None) -> Tensor[num_items, behavior_dim]
```

Therefore the graph builder, TBB-RR preprocessing, ST-HGAN, EHA, training loop, checkpointing, and evaluation code remain unchanged.

## Encoder flow

Existing E1 semantic branch:

```text
Word2Vec event mean -> multi-kernel 1D CNN -> BiGRU -> multi-head attention pooling
```

New E1-GDTC-MCBG branch:

```text
Word2Vec event mean -> gated dilated temporal convolution blocks -> evidence-aware pooling
```

Key configuration fields:

```yaml
semantic_encoder: gdtc_mcbg
gdtc_kernel_size: 3
gdtc_dilations: "1,2,4"
gdtc_dropout: 0.2
gdtc_use_event_weight_pooling: true
mw_prr_attention_beta: 1.0
```

The TBB-RR event weight side channel is consumed in the final pooling score:

```text
score_t = q^T tanh(W h_t + b) + beta * log(weight_t + eps)
```

## One-key experiment

Run both CADETS and THEIA, comparing current E1_eha_only MCBG against E1-GDTC-MCBG on the same TBB-RR graph cache:

```bash
bash scripts/run_e1_gdtc_tbb_rr_theia_cadets.sh
```

Common overrides:

```bash
DEVICE=0 SEEDS="42 43 44" EPOCHS=5 \
  bash scripts/run_e1_gdtc_tbb_rr_theia_cadets.sh
```

Fast smoke run:

```bash
CADETS_EA_PRESET=smoke SEEDS="42" EPOCHS=1 \
  bash scripts/run_e1_gdtc_tbb_rr_theia_cadets.sh
```

Outputs:

```text
runs/e1_gdtc_tbb_rr_theia_cadets_<timestamp>/
  E1_GDTC_TBB_RR_EXPERIMENT_PLAN.md
  E1_GDTC_TBB_RR_SUMMARY.csv
  E1_GDTC_TBB_RR_REPORT.md
  cadets/mcbg/...
  cadets/gdtc/...
  theia/mcbg/...
  theia/gdtc/...
```

## Interpretation

Use paired seed deltas in `E1_GDTC_TBB_RR_REPORT.md`:

- `delta_f1_gdtc_minus_mcbg > 0`: candidate improves F1;
- `delta_recall > 0` with stable/acceptable precision: candidate is better at catching positives;
- lower `train_seconds` or `cuda_peak_allocated_mb`: candidate reduces semantic-branch cost.

The experiment isolates the semantic encoder because both arms share:

- identical TBB-RR preprocessing (`redundancy_mode=target_boundary`);
- identical graph simplification;
- identical ST-HGAN + EHA settings;
- identical train/val/test graph cache per dataset.
