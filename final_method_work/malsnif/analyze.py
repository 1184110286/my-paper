from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - yaml is an optional runtime guard here
    yaml = None


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _dedupe_warnings(items: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item is None:
            continue
        text = str(item)
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out

def _candidate_metadata_paths(run_dir: Path) -> list[Path]:
    candidates = [
        run_dir / 'preprocess_metadata.json',
        run_dir.parent.parent / 'preprocess_metadata.json',  # analysis_bundle/seed_X/experiments/<exp> layout
        run_dir.parent / 'preprocess' / 'metadata.json',
        run_dir.parent.parent / 'analysis' / 'preprocess' / 'metadata.json',
        run_dir.parent.parent / 'processed' / 'graph_cache' / 'metadata.json',
        run_dir.parent.parent / 'processed' / 'metadata.json',
    ]
    cfg_path = run_dir / 'config.resolved.yaml'
    if cfg_path.exists() and yaml is not None:
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
            for key in ('metadata_dir', 'processed_dir'):
                raw = cfg.get(key)
                if not raw:
                    continue
                base = Path(str(raw))
                candidates.append(base / 'metadata.json')
                if not base.is_absolute():
                    candidates.append(Path.cwd() / base / 'metadata.json')
        except Exception:
            pass
    # Preserve order while dropping duplicates.
    out = []
    seen = set()
    for cand in candidates:
        key = str(cand)
        if key not in seen:
            out.append(cand)
            seen.add(key)
    return out


def load_preprocess_metadata(run_dir: Path) -> dict[str, Any]:
    for cand in _candidate_metadata_paths(run_dir):
        meta = load_json(cand, None)
        if isinstance(meta, dict) and meta:
            meta = dict(meta)
            meta.setdefault('_metadata_path', str(cand))
            return meta
    return {}


def analyze_run_dir(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = _load_yaml_dict(run_dir / 'config.resolved.yaml')
    meta = load_preprocess_metadata(run_dir)
    hist = load_json(run_dir / 'history.json', [])
    metrics = load_json(run_dir / 'metrics_test.json', {})
    summary = load_json(run_dir / 'train_summary.json', {})
    m = metrics.get('metrics', {}) if isinstance(metrics, dict) else {}
    all_pos = metrics.get('all_positive_baseline', {}) if isinstance(metrics, dict) else {}
    compact = load_json(run_dir / 'metrics_test_compact.json', {})
    warnings = []
    parse = meta.get('parse_stats', {}) if isinstance(meta, dict) else {}
    if parse.get('raw_events_consumed') and meta.get('config', {}).get('max_events'):
        warnings.append('本次为 max_events 截断运行，只能验证链路，不能作为论文复现指标。')
    cfg_max_events = cfg.get('max_events') if isinstance(cfg, dict) else None
    meta_max_events = meta.get('config', {}).get('max_events') if isinstance(meta, dict) else None
    if cfg_max_events is not None and meta_max_events is not None and str(cfg_max_events) != str(meta_max_events):
        warnings.append(
            f'当前 run 配置 max_events={cfg_max_events}，但加载的图缓存 metadata max_events={meta_max_events}；'
            '这通常表示启用了 REUSE_RUN，当前 MAX_EVENTS 不会改变已有 graph_cache，只影响重新预处理的新缓存。'
        )
    if m.get('predicted_positive_rate') is not None and float(m.get('predicted_positive_rate')) >= 0.999:
        warnings.append('测试集几乎全预测为正类；当前 F1/accuracy 主要等价于正样本占比。')
    if m.get('auc_roc') is not None and float(m.get('auc_roc')) < 0.5:
        warnings.append('测试 ROC-AUC < 0.5，分数排序方向异常或模型尚未学到有效判别。')
    if all_pos and abs(float(all_pos.get('f1', -1)) - float(m.get('f1', -2))) < 1e-9:
        warnings.append('模型 F1 与 all-positive baseline 相同，说明当前阈值/训练结果没有实际区分能力。')
    if meta.get('vocab_size', 999999) < 100:
        warnings.append('词表仍偏小；建议确认 CDM predicateObjectPath/对象 display 已进入 semantic tokens，并用更大 max_events 重新预处理。')
    graph_diag = meta.get('graph_diagnostics') or []
    if graph_diag:
        ratios = []
        for g in graph_diag:
            p = g.get('positive_process_nodes', 0)
            n = g.get('process_nodes') or g.get('num_process_nodes') or 0
            if n:
                ratios.append(p / n)
        if ratios and sum(r > 0.8 for r in ratios) == len(ratios):
            warnings.append('所有图窗口的正进程节点占比均超过 80%；建议检查 node_label_policy 或扩大评估范围，避免 node-level 指标被正样本占比主导。')
    graph_subset = metrics.get('graph_subset') if isinstance(metrics, dict) else None
    if not graph_subset and isinstance(compact, dict):
        graph_subset = compact.get('graph_subset')
    if graph_subset and graph_subset.get('uses_chronological_prefix'):
        warnings.append(
            f"本次测试只使用 {graph_subset.get('graphs_used')}/{graph_subset.get('total_split_graphs')} 个 {graph_subset.get('split')} 图窗口的时间前缀；"
            '这适合快速验证趋势，但不能等价于完整 split 指标。'
        )
    split_stats = meta.get('split_label_stats') if isinstance(meta, dict) else None
    if isinstance(split_stats, dict):
        for split_name, stats in split_stats.items():
            try:
                num_graphs = int(stats.get('num_graphs') or 0)
                positive_graphs = int(stats.get('positive_graphs') or 0)
                pos_ratio = float(stats.get('positive_process_ratio') or 0.0)
            except Exception:
                continue
            if num_graphs > 0 and positive_graphs == num_graphs:
                warnings.append(
                    f'{split_name} split 的所有 {num_graphs} 个图窗口都含正样本；graph-level 口径无负图，'
                    'node-level 指标也需要结合更长良性时间窗解释。'
                )
            if pos_ratio >= 0.25:
                warnings.append(
                    f'{split_name} split 的 positive_process_ratio={pos_ratio:.3f}，明显高于真实 APT 稀有度；'
                    '当前快速子集更适合检验工程链路和模型可分性，不宜直接外推到部署场景。'
                )
    projection_rows = meta.get('process_label_projection') if isinstance(meta, dict) else None
    if projection_rows:
        warnings.append(
            '本次启用了 process_label_projection=adaptive；请在论文级实验中单独报告标签投影规则，'
            '并用固定 split/hash 防止标签策略成为主要性能来源。'
        )
    train_summary = summary if isinstance(summary, dict) else {}
    if train_summary.get('train_graphs_used') and meta.get('split', {}).get('train'):
        total_train = len(meta.get('split', {}).get('train', []))
        used_train = int(train_summary.get('train_graphs_used') or 0)
        if used_train and total_train and used_train < total_train:
            warnings.append(f'训练只使用 {used_train}/{total_train} 个 train 图窗口；这是快速实验预算，不是严格复现训练。')
    last = hist[-1] if hist else {}
    if last and m:
        val_f1 = last.get('val_f1')
        test_f1 = m.get('f1')
        if val_f1 is not None and test_f1 is not None and float(val_f1) - float(test_f1) > 0.03:
            warnings.append(f'验证 F1({float(val_f1):.4f}) 高于测试 F1({float(test_f1):.4f}) 超过 0.03；建议扩大 graph_limit_val/test 或运行完整 split 验证阈值稳定性。')
        val_ap = last.get('val_average_precision')
        test_ap = m.get('average_precision') or m.get('auc_pr')
        if val_ap is not None and test_ap is not None and float(val_ap) - float(test_ap) > 0.03:
            warnings.append(f'验证 AP({float(val_ap):.4f}) 高于测试 AP({float(test_ap):.4f}) 超过 0.03；当前验证窗口可能偏容易。')
    return {
        'run_dir': str(run_dir),
        'dataset_name': meta.get('dataset_name'),
        'num_graphs': meta.get('num_graphs'),
        'vocab_size': meta.get('vocab_size'),
        'parse_stats': parse,
        'split': meta.get('split'),
        'train_summary': summary,
        'test_metrics': m,
        'all_positive_baseline': all_pos,
        'graph_subset': graph_subset,
        'run_warnings': _dedupe_warnings(warnings + list(metrics.get('warnings', [])) + list(meta.get('validation_warnings', []))),
        'last_history_row': hist[-1] if hist else None,
    }
