from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class Config:
    raw_dir: str = "data/raw/cadets"
    processed_dir: str = "data/processed/default"
    run_dir: str = "runs/debug"
    checkpoint_dir: str | None = None
    # Optional directory for analysis-side metadata. When set, processed_dir can
    # contain only graph cache files (graph_*.pkl and vocab.pkl), while
    # metadata/config diagnostics live under analysis/. Backward compatible:
    # if None, metadata is written/read from processed_dir as before.
    metadata_dir: str | None = None
    seed: int = 42

    # Data/parser settings
    dataset_name: str = "cadets"
    input_format: str = "auto"  # auto, csv, jsonl, json, cdm_json
    raw_glob: str | None = None
    raw_file_sort: str = "auto"  # auto/cdm_shards/natural/name/mtime
    label_dir: str | None = None
    cdm_information_flow: bool = True
    node_label_policy: str = "process_event_endpoints"  # process_event_endpoints/matched_endpoints/event_endpoints
    process_label_projection: str = "none"  # none/adaptive/labeled_events
    process_label_min_events: int = 2
    process_label_max_positive_ratio: float = 0.75
    process_label_min_positive_processes: int = 1
    split_ratio: tuple[float, float, float] = (0.6, 0.2, 0.2)
    window_events: int = 50000
    max_events: int | None = None
    filter_selected_events: bool = False
    simplify_graph: bool = True
    # Graph simplification preprocessing mode. leaf keeps the MalSnif paper's
    # singleton non-process node condensation. risk_aware keeps high-risk
    # singleton evidence as graph nodes and condenses only low-risk leaves.
    graph_simplify_mode: str = "leaf"  # leaf/risk_aware
    graph_simplify_risk_threshold: float = 0.62
    graph_simplify_topk_per_process: int = 0
    graph_simplify_temporal_window_ns: int = 1_000_000_000
    graph_simplify_repeat_norm: int = 8
    sanitize_paths: bool = True
    reduce_sequences: bool = True
    # Redundant sequence reduction. prefix_tree is the original MalSnif
    # Algorithm 1 behavior. target_boundary (TBB-RR) is a one-parameter,
    # compression-budgeted block-boundary replacement candidate.
    redundancy_mode: str = "prefix_tree"  # prefix_tree/target_boundary
    redundancy_risk_threshold: float = 2.5
    redundancy_preserve_risk_events: int = 1
    redundancy_repeat_summary: bool = True
    redundancy_repeat_min: int = 3
    mw_prr_alpha: float = 0.2
    mw_prr_attention_beta: float = 1.0
    btr_rr_max_block_len: int = 16
    btr_rr_min_gain: int = 2
    btr_rr_repeat_cap: int = 32
    btr_rr_alpha: float = 0.3
    lz_sr_min_phrase_len: int = 4
    lz_sr_max_phrase_len: int = 24
    lz_sr_window: int = 512
    lz_sr_min_gain: int = 2
    lz_sr_alpha: float = 0.25
    frc_rr_cap_size: int = 3
    frc_rr_repeat_cap: int = 32
    frc_rr_alpha: float = 0.25
    flb_rr_repeat_cap: int = 32
    flb_rr_alpha: float = 0.25
    wra_rr_window: int = 11
    tbb_rr_target_compression: float = 0.90
    lowercase_tokens: bool = True

    # Vocabulary/embedding
    word_dim: int = 64
    min_token_freq: int = 1
    skipgram_epochs: int = 1
    skipgram_window: int = 2
    skipgram_negative: int = 5
    skipgram_lr: float = 0.01
    skipgram_batch_size: int = 4096
    skipgram_max_sentences: int | None = 1500000
    skipgram_max_pairs: int | None = 3000000
    freeze_word_embeddings: bool = False
    max_tokens_per_event: int = 12
    max_events_per_node: int = 96
    max_events_per_edge: int = 16

    # Model
    # model_variant=baseline keeps the original MalSnif reproduction.
    # model_variant=agf_st_hgan_mcbg enables the v1 late-fusion idea.
    # model_variant=edge_gated_st_hgan_mcbg enables the v2 MalSnif-aligned
    # edge-gated message passing idea.
    # model_variant=ea_st_hgan_mcbg enables the v3 MalSnif-aligned
    # EA-THGN-inspired EHA/ETS/EAW node-adaptive mechanisms, without AGF gate.
    model_variant: str = "baseline"  # baseline/agf_st_hgan_mcbg/edge_gated_st_hgan_mcbg/ea_st_hgan_mcbg
    hidden_dim: int = 64
    semantic_dim: int = 64
    behavior_dim: int = 64
    # Semantic branch selection. Baseline uses GRU+BiLSTM; MCBG uses
    # multi-kernel CNN + BiGRU + multi-head attention over event vectors.
    # gdtc_mcbg is the E1-GDTC-MCBG drop-in semantic encoder: Word2Vec
    # event means -> gated dilated temporal conv -> evidence-aware pooling.
    # rgd_bigru_mcbg is the conservative E1-RGD-BiGRU-MCBG encoder:
    # residual gated dilated CNN -> residual BiGRU -> attention pooling.
    semantic_encoder: str = "baseline"  # baseline/mcbg/gdtc_mcbg/rgd_bigru_mcbg
    mcbg_kernel_sizes: str = "2,3,5"
    mcbg_conv_dim: int = 0  # 0 => semantic_dim
    mcbg_attention_heads: int = 4
    mcbg_dropout: float = 0.2
    gdtc_kernel_size: int = 3
    gdtc_dilations: str = "1,2,4"
    gdtc_dropout: float = 0.2
    gdtc_use_event_weight_pooling: bool = True
    rgd_kernel_size: int = 3
    rgd_dilations: str = "1,2"
    rgd_dropout: float = 0.2
    rgd_residual_scale_init: float = 0.1
    rgd_depthwise_separable: bool = True
    rgd_use_event_weight_pooling: bool = True
    # Graph message passing backend. MalSnif paper uses GCN; this reproduction
    # can use a GraphSAGE mean aggregator as a memory-friendly, interface-compatible
    # replacement while keeping the same semantic encoders, edge weights, loss and
    # evaluation flow. Set graph_encoder=gcn to recover the previous backend.
    graph_encoder: str = "graphsage"  # graphsage/gcn/st_hgan/hgan
    gcn_layers: int = 2
    edge_chunk_size: int = 200000
    graphsage_normalize: bool = False
    graphsage_use_root: bool = True
    # ST-HGAN branch controls. They are ignored by baseline GCN/GraphSAGE.
    hgan_num_relations: int = 128
    hgan_num_time_buckets: int = 16
    hgan_use_node_types: bool = True
    hgan_use_relation_types: bool = True
    hgan_use_time_bias: bool = True
    hgan_topk: int = 20  # <=0 disables Top-k attention focusing
    hgan_pruning_mode: str = "soft"  # none/soft/hard
    hgan_soft_pruning_floor: float = 0.05
    hgan_attention_dropout: float = 0.1
    hgan_leaky_relu_negative_slope: float = 0.2
    hgan_use_residual: bool = True
    return_attention_stats: bool = True
    # Fusion controls. agf uses vector gate by default; scalar_gate is A7.
    fusion_mode: str = "baseline"  # baseline/agf/static_concat/scalar_gate/semantic_only/structure_only/mean
    # v2 MalSnif-aligned edge gate controls.  These are used only by
    # model_variant=edge_gated_st_hgan_mcbg.  The gate is applied per edge during
    # message passing; it is not a final semantic/structure late-fusion gate.
    edge_gate_mode: str = "vector"  # vector/scalar/none/fixed_one/fixed_half
    edge_gate_hidden_dim: int = 0  # 0 => hidden_dim
    edge_gate_dropout: float = 0.1
    edge_gate_temperature: float = 1.0
    edge_gate_use_edge_semantics: bool = True

    # v3 EA-THGN-inspired node-adaptive mechanisms. Used only by
    # model_variant=ea_st_hgan_mcbg. These replace the original AGF/edge gate
    # idea with MalSnif-aligned node-wise adaptivity inside message passing.
    ea_use_eha: bool = False  # Elastic Hop Aggregation: node-wise hop depth
    ea_use_ets: bool = False  # Elastic Temporal Softmax: node-wise attention temperature
    ea_use_eaw: bool = False  # Elastic Attention Width: node-wise head bandwidth
    ea_num_heads: int = 4
    ea_hidden_dim: int = 0  # 0 => hidden_dim
    ea_tau_min: float = 0.1
    ea_tau_max: float = 5.0
    ea_dropout: float = 0.0

    gate_hidden_dim: int = 0  # 0 => hidden_dim
    gate_dropout: float = 0.1
    gate_temperature: float = 1.0
    dropout: float = 0.2
    use_semantics: bool = True
    use_edge_weights: bool = True
    # Edge weights remain learned from edge event sequences as in MalSnif.
    # legacy_sigmoid matches the early implementation (0..1). centered_sigmoid
    # starts from the unweighted baseline (1.0) and learns multiplicative weights
    # in (0, 2), avoiding systematic under-weighting of graph edges relative to
    # self-loops. This changes only calibration, not the model flow.
    edge_weight_mode: str = "legacy_sigmoid"  # legacy_sigmoid/centered_sigmoid/softplus
    edge_weight_init_zero: bool = False
    graph_level: bool = False
    graph_readout: str = "max"  # max, mean
    # process is closest to MalSnif; auto falls back to all nodes only when process labels are single-class.
    node_scope: str = "auto"  # auto/process/all

    # Train
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    downsample_after_forward: bool = True
    downsample_weight: int = 10
    loss_sampling: str = "paper"  # paper/balanced/all
    balanced_loss: bool = True
    loss_positive_weight: float | None = None
    model_selection_metric: str = "val_auc_pr"  # val_auc_pr/val_auc_roc/val_mcc/val_f1
    # Secondary metrics used only when the primary selection metric ties within model_selection_epsilon.
    # This prevents early checkpoints with identical F1 but much worse ranking quality from being selected.
    model_selection_tie_breakers: str = "val_average_precision,val_mcc,val_balanced_accuracy"
    model_selection_epsilon: float = 1e-9
    threshold_strategy: str = "fixed"  # fixed/val_f1/val_mcc/val_balanced/val_balanced_accuracy
    threshold_min_recall: float = 0.95
    threshold: float | str = 0.5
    allow_unlabeled_training: bool = False
    grad_clip: float = 5.0
    patience: int = 20

    # Resource guards for first quick version
    max_nodes_per_graph: int | None = None
    max_edges_per_graph: int | None = None

    # Misc
    num_workers: int = 0

    # Fast diagnostic controls. These are experiment-budget controls only;
    # strict reproduction should keep val_every=1 and top_alerts_per_graph=50.
    val_every: int = 1  # validate every N epochs; always validates epoch 1 and final epoch
    top_alerts_per_graph: int = 50  # analysis-only top alerts saved per graph; metrics are unchanged
    cache_graphs_in_memory: bool = False  # cache split graph pickles in RAM across epochs when the fast subset is small
    # Plot generation policy.  essential keeps only history.png and scores_test.png,
    # which are sufficient for most analysis; all reproduces the old behavior and
    # writes one png per metric; none disables plot files.
    plot_mode: str = "essential"  # essential/all/none
    plot_metric_keys: str = "loss,val_f1,val_mcc,val_average_precision,val_threshold"

    # Speed / visibility options for long CADETS runs.
    # These do not change the MalSnif method pipeline; they control how much
    # of the already preprocessed chronological graph cache is used by a fast
    # experiment, and whether CUDA AMP/progress bars are enabled.
    graph_limit_train: int | None = None
    graph_limit_val: int | None = None
    graph_limit_test: int | None = None
    train_progress: bool = True
    show_progress: bool = True
    use_amp: bool = False
    amp_dtype: str = "float16"  # float16/bfloat16
    # If an AMP-only dtype/device issue occurs in a custom operation, fall back
    # to full precision instead of aborting a long run.  This preserves the
    # MalSnif algorithm; it only changes numerical execution precision.
    amp_fallback_to_fp32: bool = True

    # CUDA memory hygiene. PyTorch's CUDA caching allocator may reserve much
    # more memory than is actively allocated, which makes nvidia-smi look high
    # even when tensors only need a few GB.  These options periodically release
    # unused cached blocks. They do not change model math or MalSnif flow.
    cuda_empty_cache_interval: int = 0  # 0 disables; 1 releases after every graph
    cuda_empty_cache_after_epoch: bool = False
    cuda_empty_cache_after_eval: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        flat: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value
        # split_ratio in YAML may be list
        if "split_ratio" in flat:
            flat["split_ratio"] = tuple(flat["split_ratio"])
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        clean = {k: v for k, v in flat.items() if k in valid}
        return cls(**clean)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["split_ratio"] = list(self.split_ratio)
        return d

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
