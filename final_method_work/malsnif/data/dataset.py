from __future__ import annotations

from pathlib import Path
import json
from tqdm.auto import tqdm

from malsnif.config import Config
from malsnif.data.parsers import discover_input_files, iter_events
from malsnif.data.cdm import CdmLabeler, CdmParseStats, iter_cdm_events
from malsnif.data.build_graph import GraphBuilder, encode_graph_tokens
from malsnif.data.vocab import Vocabulary, train_skipgram
from malsnif.utils.io import ensure_dir, save_json, save_pickle, load_pickle


def chunk_events(events, window_events: int):
    if window_events <= 0 or len(events) <= window_events:
        return [events]
    return [events[i : i + window_events] for i in range(0, len(events), window_events)]


def split_indices(n: int, ratio=(0.6, 0.2, 0.2)):
    """Deterministic chronological split with usable small-dataset behavior.

    The previous implementation could produce an empty validation split for
    quick CDM checks, e.g. 4 windows with 0.6/0.2/0.2 became 2/0/2.  Training
    would then silently select checkpoints by train metrics.  For n>=3 we keep
    all train/val/test non-empty.
    """
    if n <= 0:
        return {"train": [], "val": [], "test": []}
    idx = list(range(n))
    if n == 1:
        return {"train": [0], "val": [], "test": []}
    if n == 2:
        return {"train": [0], "val": [], "test": [1]}

    total = sum(float(x) for x in ratio) or 1.0
    r0, r1, r2 = [max(0.0, float(x) / total) for x in ratio]
    n_train = max(1, int(round(n * r0)))
    n_val = max(1, int(round(n * r1))) if r1 > 0 else 0
    n_test = n - n_train - n_val
    if r2 > 0 and n_test < 1:
        # Borrow from the largest split while keeping train and val valid.
        while n_test < 1:
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
            n_test = n - n_train - n_val
    if n_train + n_val + n_test != n:
        n_test = max(0, n - n_train - n_val)
    return {"train": idx[:n_train], "val": idx[n_train:n_train+n_val], "test": idx[n_train+n_val:]}


def _is_cdm_format(fmt: str) -> bool:
    return fmt.lower() in {"cdm", "cdm_json", "darpa_cdm", "darpa_cdm_json"}


def metadata_path_for(processed_dir: str | Path, metadata_dir: str | Path | None = None) -> Path:
    """Resolve metadata.json for a processed graph cache.

    v2.0 separates heavy graph cache files from analysis artifacts.  The
    preferred location is metadata_dir/metadata.json; old projects with
    processed_dir/metadata.json remain supported.  For convenience, when a
    user passes runs/.../processed/<name>, we also look for sibling
    runs/.../analysis/<name>/metadata.json and
    runs/.../analysis/preprocess/metadata.json.
    """
    processed_dir = Path(processed_dir)
    candidates: list[Path] = []
    if metadata_dir:
        md = Path(metadata_dir)
        candidates.extend([md / "metadata.json", md / "preprocess_metadata.json"])
    candidates.append(processed_dir / "metadata.json")
    # New v2.0 layout fallback: runs/<run>/processed/<cache> and analysis/preprocess.
    try:
        parts = processed_dir.parts
        if "processed" in parts:
            idx = len(parts) - 1 - list(reversed(parts)).index("processed")
            run_root = Path(*parts[:idx]) if idx > 0 else Path(processed_dir.anchor)
            exp_name = processed_dir.parts[idx + 1] if idx + 1 < len(parts) else processed_dir.name
            candidates.extend([
                run_root / "analysis" / exp_name / "metadata.json",
                run_root / "analysis" / exp_name / "preprocess_metadata.json",
                run_root / "analysis" / "preprocess" / "metadata.json",
                run_root / "analysis" / "preprocess" / "preprocess_metadata.json",
            ])
    except Exception:
        pass
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "metadata.json not found. Checked: " + ", ".join(str(c) for c in candidates)
    )

def metadata_output_dir(cfg: Config) -> Path:
    return ensure_dir(getattr(cfg, "metadata_dir", None) or cfg.processed_dir)


def _event_iterator_with_aux(cfg: Config, files: list[Path]):
    if _is_cdm_format(cfg.input_format):
        cdm_stats = CdmParseStats()
        labeler = CdmLabeler.from_dir(cfg.label_dir)
        events = iter_cdm_events(
            files,
            label_dir=cfg.label_dir,
            sanitize=cfg.sanitize_paths,
            information_flow=cfg.cdm_information_flow,
            stats=cdm_stats,
            labeler=labeler,
        )
        return events, cdm_stats, labeler
    return iter_events(
        files,
        input_format=cfg.input_format,
        sanitize=cfg.sanitize_paths,
        filter_selected_events=cfg.filter_selected_events,
        label_dir=cfg.label_dir,
        cdm_information_flow=cfg.cdm_information_flow,
    ), None, None


def _build_token_graphs_streaming(cfg: Config, files: list[Path]) -> tuple[list[dict], dict, list[str]]:
    builder = GraphBuilder(cfg)
    token_graphs: list[dict] = []
    warnings: list[str] = []
    buf = []
    total_events = 0
    labeled_events = 0
    event_iter, cdm_stats, labeler = _event_iterator_with_aux(cfg, files)
    show_progress = bool(getattr(cfg, "show_progress", True))
    total_hint = cfg.max_events if cfg.max_events is not None else None
    for ev in tqdm(event_iter, desc="preprocess events", unit="event", total=total_hint, dynamic_ncols=True, disable=not show_progress):
        buf.append(ev)
        total_events += 1
        labeled_events += int(ev.tag > 0)
        if cfg.max_events is not None and total_events >= cfg.max_events:
            warnings.append(
                f"预处理因 max_events={cfg.max_events} 提前截断；当前 metadata 只代表快速验证子集，不代表完整 DARPA/CADETS。"
            )
            break
        if cfg.window_events > 0 and len(buf) >= cfg.window_events:
            print(f"[preprocess] build graph window {len(token_graphs):05d}: events={len(buf)} labeled={sum(int(x.tag > 0) for x in buf)}", flush=True)
            token_graphs.append(builder.build_tokens_graph(buf))
            buf = []
    if buf:
        print(f"[preprocess] build graph window {len(token_graphs):05d}: events={len(buf)} labeled={sum(int(x.tag > 0) for x in buf)}", flush=True)
        token_graphs.append(builder.build_tokens_graph(buf))
    stats = {
        "raw_events_consumed": total_events,
        "labeled_events_consumed": labeled_events,
        "positive_event_ratio": labeled_events / max(total_events, 1),
    }
    if cdm_stats is not None:
        stats["cdm"] = {
            "records_seen": cdm_stats.records_seen,
            "objects_seen": cdm_stats.objects_seen,
            "events_seen": cdm_stats.events_seen,
            "events_emitted": cdm_stats.events_emitted,
            "events_missing_subject": cdm_stats.events_missing_subject,
            "events_missing_object": cdm_stats.events_missing_object,
            "events_without_label": cdm_stats.events_without_label,
        }
    if labeler is not None:
        stats["labeler"] = labeler.summary()
        if not labeler.has_labels:
            warnings.append(
                "未加载到任何有效标签：可以构图，但训练/评估不会有真实恶意样本；请检查 labels/_raw/cadets.json 或 malicious_*.txt/csv。"
            )
        elif labeled_events == 0:
            warnings.append(
                "已加载标签文件，但当前已消费事件中未命中恶意标签：可能是 max_events 截断在攻击发生前，也可能是标签格式/UUID 不匹配。"
            )
    elif labeled_events == 0:
        warnings.append("当前预处理结果没有任何正样本标签，训练指标不具备论文复现意义。")
    return token_graphs, stats, warnings



def _split_label_stats(graphs: list[dict], split: dict) -> dict:
    out: dict[str, dict] = {}
    for name, idxs in split.items():
        selected = [graphs[i] for i in idxs]
        process_nodes = int(sum(sum(1 for is_proc in g.get("process_mask", []) if is_proc) for g in selected))
        positive_process_nodes = int(sum(
            sum(int(lbl) for lbl, is_proc in zip(g.get("node_labels", []), g.get("process_mask", [])) if is_proc)
            for g in selected
        ))
        positive_nodes = int(sum(sum(int(x) for x in g.get("node_labels", [])) for g in selected))
        total_nodes = int(sum(len(g.get("node_labels", [])) for g in selected))
        out[name] = {
            "num_graphs": len(selected),
            "positive_graphs": int(sum(int(g.get("graph_label", 0) > 0) for g in selected)),
            "process_nodes": process_nodes,
            "positive_nodes": positive_nodes,
            "negative_nodes": max(total_nodes - positive_nodes, 0),
            "positive_node_ratio": positive_nodes / max(total_nodes, 1),
            "positive_process_nodes": positive_process_nodes,
            "negative_process_nodes": max(process_nodes - positive_process_nodes, 0),
            "positive_process_ratio": positive_process_nodes / max(process_nodes, 1),
            "positive_events": int(sum(int(g.get("stats", {}).get("original_positive_events", 0) or 0) for g in selected)),
        }
    return out


def _graph_diagnostics(graphs: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for i, g in enumerate(graphs):
        process_mask = list(g.get("process_mask", []))
        labels = list(g.get("node_labels", []))
        pos_proc = int(sum(int(lbl) for lbl, is_proc in zip(labels, process_mask) if is_proc))
        proc = int(sum(1 for x in process_mask if x))
        rows.append({
            "graph_index": i,
            "graph_label": int(g.get("graph_label", 0)),
            "nodes": len(labels),
            "edges": len(g.get("edge_index", [])),
            "process_nodes": proc,
            "positive_nodes": int(sum(int(x) for x in labels)),
            "positive_process_nodes": pos_proc,
            "positive_process_ratio": pos_proc / max(proc, 1),
            "events": int(g.get("stats", {}).get("original_events", 0) or 0),
            "labeled_events": int(g.get("stats", {}).get("labeled_events", g.get("stats", {}).get("original_positive_events", 0)) or 0),
        })
    return rows

def _collect_split_metadata(cfg: Config, graphs: list[dict], files: list[Path], parse_stats: dict, warnings: list[str] | None = None) -> dict:
    split = split_indices(len(graphs), cfg.split_ratio)
    split_label_stats = _split_label_stats(graphs, split)
    graph_diagnostics = _graph_diagnostics(graphs)
    validation_warnings = list(warnings or [])
    # A very high process-level positive ratio usually means label propagation is too broad.
    for row in graph_diagnostics:
        if row["process_nodes"] >= 20 and row["positive_process_ratio"] > 0.85:
            validation_warnings.append(
                f"图 {row['graph_index']} 的正样本进程比例为 {row['positive_process_ratio']:.3f}，接近全正；"
                "若模型预测全为恶意，请优先检查 node_label_policy 或标签语义。"
            )
    if len(graphs) >= 3 and not split.get("val"):
        validation_warnings.append("验证集为空；建议增加 max_events 或降低 window_events。")
    if len(graphs) >= 2 and not split.get("test"):
        validation_warnings.append("测试集为空；建议增加 max_events 或降低 window_events。")
    if parse_stats.get("labeled_events_consumed", 0) == 0:
        validation_warnings.append("本次输出图全部为负样本；请勿用该 processed_dir 进行正式训练/评估。")
    elif int(split_label_stats.get("train", {}).get("positive_graphs", 0)) == int(split_label_stats.get("train", {}).get("num_graphs", 0)) and int(split_label_stats.get("test", {}).get("positive_graphs", 0)) == int(split_label_stats.get("test", {}).get("num_graphs", 0)):
        validation_warnings.append("所有 train/test 图窗口都含正样本；graph-level 结果无负图对照，node-level 指标也容易被正样本占比主导。建议增大 max_events 或使用完整配置以覆盖更多良性窗口。")
    if parse_stats.get("labeled_events_consumed", 0) > 0 and not cfg.graph_level:
        total_train_proc = sum(sum(1 for x in graphs[i].get("process_mask", []) if x) for i in split.get("train", []))
        pos_train_proc = split_label_stats.get("train", {}).get("positive_process_nodes", 0)
        if total_train_proc and pos_train_proc / max(total_train_proc, 1) > 0.8:
            validation_warnings.append(f"训练进程节点正样本占比过高({pos_train_proc}/{total_train_proc})；F1/Accuracy 可能接近 all-positive 基线，请重点看 MCC、Balanced Accuracy、ROC-AUC 和 Precision@K。")
        train_stats = split_label_stats.get("train", {})
        if train_stats.get("positive_process_nodes", 0) == 0:
            validation_warnings.append("训练切分没有正样本进程节点；请增大 max_events、调整 window_events，或使用完整 CADETS 预处理/标签。")
        if train_stats.get("negative_process_nodes", 0) == 0:
            validation_warnings.append("训练切分没有负样本进程节点；监督分类会退化为单类学习。")
        for split_name in ["val", "test"]:
            sstats = split_label_stats.get(split_name, {})
            if sstats.get("num_graphs", 0) and sstats.get("positive_process_nodes", 0) == 0:
                validation_warnings.append(f"{split_name} 切分没有正样本进程节点；该切分的节点级指标可能无效。")
            if sstats.get("num_graphs", 0) and sstats.get("negative_process_nodes", 0) == 0:
                validation_warnings.append(f"{split_name} 切分没有负样本进程节点；Specificity/FPR/MCC 将无法反映误报能力。")
        for split_name, sstats in split_label_stats.items():
            if sstats.get("num_graphs", 0) and sstats.get("positive_process_ratio", 0) > 0.8:
                validation_warnings.append(f"{split_name} 切分正样本进程占比 {sstats.get('positive_process_ratio'):.3f}，节点级 Accuracy/F1 容易被 all-positive 基线抬高。")
    return {
        "split": split,
        "split_label_stats": split_label_stats,
        "graph_diagnostics": graph_diagnostics,
        "validation_warnings": validation_warnings,
        "raw_files": [str(p) for p in files],
        "raw_file_sort": cfg.raw_file_sort,
    }

def strict_split_precheck(cfg: Config) -> dict:
    """Run a lightweight split-rigor check without building vocab/cache files.

    This keeps the costly and method-relevant part of preprocessing
    (chronological event parsing, graph-window construction, process-label
    projection, and graph simplification), but skips corpus construction,
    skip-gram training, graph token encoding, and graph pickle writes.
    """
    meta_out = metadata_output_dir(cfg)
    files = discover_input_files(cfg.raw_dir, raw_glob=cfg.raw_glob, sort_mode=cfg.raw_file_sort)
    if not files:
        raise FileNotFoundError(f"No raw CSV/JSON/JSONL/CDM files found under {cfg.raw_dir}")

    token_graphs, parse_stats, warnings = _build_token_graphs_streaming(cfg, files)
    if not token_graphs:
        raise ValueError("Parsed zero events. Check input_format/field names/raw_glob/filter_selected_events.")

    common = _collect_split_metadata(cfg, token_graphs, files, parse_stats, warnings)
    meta = {
        "dataset_name": cfg.dataset_name,
        "num_graphs": len(token_graphs),
        "parse_stats": parse_stats,
        **common,
        "stats": [g["stats"] for g in token_graphs],
        "config": cfg.to_dict(),
        "precheck_only": True,
    }
    save_json(meta, meta_out / "metadata.json")
    return meta

def strict_split_autostop_precheck(
    cfg: Config,
    *,
    required_splits: list[str] | None = None,
    require_graph_mix: bool = True,
    require_node_mix: bool = True,
    min_graphs_per_split: int | None = None,
    check_every_windows: int = 1,
) -> dict:
    """Scan events once and stop as soon as the strict split becomes usable.

    This is a stronger optimization than running multiple prefix prechecks:
    we build chronological windows only once, recompute split diagnostics after
    checkpoints, and stop at the first prefix that already satisfies the
    requested graph/process-node class mixture.
    """
    meta_out = metadata_output_dir(cfg)
    files = discover_input_files(cfg.raw_dir, raw_glob=cfg.raw_glob, sort_mode=cfg.raw_file_sort)
    if not files:
        raise FileNotFoundError(f"No raw CSV/JSON/JSONL/CDM files found under {cfg.raw_dir}")

    required_splits = [s for s in (required_splits or ["train", "val", "test"]) if s]
    if min_graphs_per_split is None:
        min_graphs_per_split = 2 if require_graph_mix else 1
    check_every_windows = max(int(check_every_windows or 1), 1)

    builder = GraphBuilder(cfg)
    token_graphs: list[dict] = []
    warnings: list[str] = []
    buf = []
    total_events = 0
    labeled_events = 0
    event_iter, cdm_stats, labeler = _event_iterator_with_aux(cfg, files)
    show_progress = bool(getattr(cfg, "show_progress", True))
    total_hint = cfg.max_events if cfg.max_events is not None else None
    evaluation_history: list[dict] = []
    pass_snapshot: dict | None = None

    def evaluate_now() -> dict:
        split = split_indices(len(token_graphs), cfg.split_ratio)
        split_label_stats = _split_label_stats(token_graphs, split)
        rows = []
        overall_pass = True
        for split_name in required_splits:
            s = split_label_stats.get(split_name, {}) or {}
            num_graphs = int(s.get("num_graphs", 0) or 0)
            pos_graphs = int(s.get("positive_graphs", 0) or 0)
            neg_graphs = max(num_graphs - pos_graphs, 0)
            pos_proc = int(s.get("positive_process_nodes", 0) or 0)
            neg_proc = int(s.get("negative_process_nodes", 0) or 0)
            enough_graphs = num_graphs >= int(min_graphs_per_split)
            graph_ok = (pos_graphs > 0 and neg_graphs > 0) if require_graph_mix else True
            node_ok = (pos_proc > 0 and neg_proc > 0) if require_node_mix else True
            split_ok = enough_graphs and graph_ok and node_ok
            overall_pass = overall_pass and split_ok
            rows.append({
                "split": split_name,
                "num_graphs": num_graphs,
                "positive_graphs": pos_graphs,
                "negative_graphs": neg_graphs,
                "positive_process_nodes": pos_proc,
                "negative_process_nodes": neg_proc,
                "enough_graphs": enough_graphs,
                "graph_mix": graph_ok,
                "node_mix": node_ok,
                "split_ok": split_ok,
            })
        return {
            "overall_pass": overall_pass,
            "windows": len(token_graphs),
            "raw_events_consumed": total_events,
            "rows": rows,
            "split": split,
            "split_label_stats": split_label_stats,
        }

    for ev in tqdm(event_iter, desc="preprocess events", unit="event", total=total_hint, dynamic_ncols=True, disable=not show_progress):
        buf.append(ev)
        total_events += 1
        labeled_events += int(ev.tag > 0)
        if cfg.max_events is not None and total_events >= cfg.max_events:
            warnings.append(
                f"预处理因 max_events={cfg.max_events} 提前截断；当前 metadata 只代表快速验证子集，不代表完整 DARPA/CADETS。"
            )
            break
        if cfg.window_events > 0 and len(buf) >= cfg.window_events:
            print(f"[precheck-autostop] build graph window {len(token_graphs):05d}: events={len(buf)} labeled={sum(int(x.tag > 0) for x in buf)}", flush=True)
            token_graphs.append(builder.build_tokens_graph(buf))
            buf = []
            if len(token_graphs) % check_every_windows == 0:
                snapshot = evaluate_now()
                evaluation_history.append({
                    "windows": snapshot["windows"],
                    "raw_events_consumed": snapshot["raw_events_consumed"],
                    "overall_pass": snapshot["overall_pass"],
                    "rows": snapshot["rows"],
                })
                if snapshot["overall_pass"]:
                    pass_snapshot = snapshot
                    break
    if pass_snapshot is None and buf:
        print(f"[precheck-autostop] build graph window {len(token_graphs):05d}: events={len(buf)} labeled={sum(int(x.tag > 0) for x in buf)}", flush=True)
        token_graphs.append(builder.build_tokens_graph(buf))
        snapshot = evaluate_now()
        evaluation_history.append({
            "windows": snapshot["windows"],
            "raw_events_consumed": snapshot["raw_events_consumed"],
            "overall_pass": snapshot["overall_pass"],
            "rows": snapshot["rows"],
        })
        if snapshot["overall_pass"]:
            pass_snapshot = snapshot
    stats = {
        "raw_events_consumed": total_events,
        "labeled_events_consumed": labeled_events,
        "positive_event_ratio": labeled_events / max(total_events, 1),
    }
    if cdm_stats is not None:
        stats["cdm"] = {
            "records_seen": cdm_stats.records_seen,
            "objects_seen": cdm_stats.objects_seen,
            "events_seen": cdm_stats.events_seen,
            "events_emitted": cdm_stats.events_emitted,
            "events_missing_subject": cdm_stats.events_missing_subject,
            "events_missing_object": cdm_stats.events_missing_object,
            "events_without_label": cdm_stats.events_without_label,
        }
    if labeler is not None:
        stats["labeler"] = labeler.summary()
        if not labeler.has_labels:
            warnings.append(
                "未加载到任何有效标签：可以构图，但训练/评估不会有真实恶意样本；请检查 labels/_raw/cadets.json 或 malicious_*.txt/csv。"
            )
        elif labeled_events == 0:
            warnings.append(
                "已加载标签文件，但当前已消费事件中未命中恶意标签：可能是 max_events 截断在攻击发生前，也可能是标签格式/UUID 不匹配。"
            )
    elif labeled_events == 0:
        warnings.append("当前预处理结果没有任何正样本标签，训练指标不具备论文复现意义。")

    if not token_graphs:
        raise ValueError("Parsed zero events. Check input_format/field names/raw_glob/filter_selected_events.")

    common = _collect_split_metadata(cfg, token_graphs, files, stats, warnings)
    search_result = pass_snapshot or evaluate_now()
    stop_reason = "strict_mix_satisfied" if search_result.get("overall_pass") else "input_exhausted_without_strict_mix"
    meta = {
        "dataset_name": cfg.dataset_name,
        "num_graphs": len(token_graphs),
        "parse_stats": stats,
        **common,
        "stats": [g["stats"] for g in token_graphs],
        "config": cfg.to_dict(),
        "precheck_only": True,
        "strict_search": {
            "mode": "autostop",
            "required_splits": required_splits,
            "require_graph_mix": bool(require_graph_mix),
            "require_node_mix": bool(require_node_mix),
            "min_graphs_per_split": int(min_graphs_per_split),
            "check_every_windows": int(check_every_windows),
            "overall_pass": bool(search_result.get("overall_pass")),
            "stop_reason": stop_reason,
            "selected_max_events": int(search_result.get("raw_events_consumed", total_events) or total_events),
            "selected_num_graphs": int(search_result.get("windows", len(token_graphs)) or len(token_graphs)),
            "evaluation_history": evaluation_history[-32:],
        },
    }
    save_json(meta, meta_out / "metadata.json")
    return meta

def preprocess(cfg: Config) -> dict:
    out = ensure_dir(cfg.processed_dir)
    meta_out = metadata_output_dir(cfg)
    files = discover_input_files(cfg.raw_dir, raw_glob=cfg.raw_glob, sort_mode=cfg.raw_file_sort)
    if not files:
        raise FileNotFoundError(f"No raw CSV/JSON/JSONL/CDM files found under {cfg.raw_dir}")

    token_graphs, parse_stats, warnings = _build_token_graphs_streaming(cfg, files)
    if not token_graphs:
        raise ValueError("Parsed zero events. Check input_format/field names/raw_glob/filter_selected_events.")

    # Build corpus from all event token sequences to reproduce word2vec/Skip-Gram embedding.
    corpus: list[list[str]] = []
    for g in tqdm(token_graphs, desc="preprocess token corpus", unit="graph", dynamic_ncols=True, disable=not getattr(cfg, "show_progress", True)):
        for seq in g["node_event_tokens"]:
            corpus.extend(seq)
        for seq in g["edge_event_tokens"]:
            corpus.extend(seq)

    vocab = Vocabulary.build(corpus, min_freq=cfg.min_token_freq, dim=cfg.word_dim, seed=cfg.seed)
    vocab = train_skipgram(
        vocab, corpus, epochs=cfg.skipgram_epochs, window=cfg.skipgram_window,
        negative=cfg.skipgram_negative, lr=cfg.skipgram_lr, seed=cfg.seed,
        batch_size=getattr(cfg, "skipgram_batch_size", 4096),
        max_sentences=getattr(cfg, "skipgram_max_sentences", None),
        max_pairs=getattr(cfg, "skipgram_max_pairs", None),
        show_progress=getattr(cfg, "show_progress", True),
    )
    graphs = [encode_graph_tokens(g, vocab, cfg) for g in tqdm(token_graphs, desc="preprocess encode graphs", unit="graph", dynamic_ncols=True, disable=not getattr(cfg, "show_progress", True))]
    common = _collect_split_metadata(cfg, graphs, files, parse_stats, warnings)

    # Save individual graphs to avoid loading huge datasets into memory unnecessarily.
    graph_files = []
    for i, g in enumerate(tqdm(graphs, desc="preprocess save graphs", unit="graph", dynamic_ncols=True, disable=not getattr(cfg, "show_progress", True))):
        path = out / f"graph_{i:05d}.pkl"
        save_pickle(g, path)
        graph_files.append(path.name)
    save_pickle(vocab, out / "vocab.pkl")
    meta = {
        "dataset_name": cfg.dataset_name,
        "num_graphs": len(graphs),
        "graph_files": graph_files,
        "split": common["split"],
        "vocab_size": len(vocab.idx_to_token),
        "parse_stats": parse_stats,
        **common,
        "stats": [g["stats"] for g in graphs],
        "config": cfg.to_dict(),
    }
    save_json(meta, meta_out / "metadata.json")
    # Backward compatibility only when metadata_dir is not used.  In v2.0
    # pipeline runs, metadata_dir points to analysis/, so the processed graph
    # cache remains free of metadata/config files.
    if not getattr(cfg, "metadata_dir", None):
        save_json(meta, out / "metadata.json")
    return meta


class ProcessedGraphs:
    def __init__(self, processed_dir: str | Path, split: str = "train", metadata_dir: str | Path | None = None, limit: int | None = None, cache_in_memory: bool = False):
        self.processed_dir = Path(processed_dir)
        self.metadata_path = metadata_path_for(self.processed_dir, metadata_dir=metadata_dir)
        self.meta = json.load(open(self.metadata_path, "r", encoding="utf-8"))
        self.split = split
        self.indices = list(self.meta["split"].get(split, []))
        if limit is not None and int(limit) > 0:
            # Deterministic chronological prefix of this split.  This is used
            # only by fast/pilot scripts to shorten runtime while keeping the
            # same MalSnif preprocessing/model/training flow.
            self.indices = self.indices[: int(limit)]
        self.files = [self.meta["graph_files"][i] for i in self.indices]
        self.cache_in_memory = bool(cache_in_memory)
        self._cache: list[dict] | None = None
        if self.cache_in_memory:
            self._cache = [self._load_graph(idx, fn) for idx, fn in zip(self.indices, self.files)]

    def __len__(self):
        return len(self.files)

    def _load_graph(self, idx: int, fn: str):
        g = load_pickle(self.processed_dir / fn)
        if isinstance(g, dict):
            g = dict(g)
            g["_graph_index"] = idx
            g["_graph_file"] = fn
        return g

    def __iter__(self):
        if self._cache is not None:
            # Yield a shallow copy so per-graph transient keys cannot leak
            # between epochs, while avoiding repeated disk reads of large pickle
            # files in fast diagnostic runs. Tensor/list payloads remain shared.
            for g in self._cache:
                yield dict(g) if isinstance(g, dict) else g
            return
        for idx, fn in zip(self.indices, self.files):
            yield self._load_graph(idx, fn)

    def all(self):
        return list(iter(self))


def load_vocab(processed_dir: str | Path) -> Vocabulary:
    return load_pickle(Path(processed_dir) / "vocab.pkl")
