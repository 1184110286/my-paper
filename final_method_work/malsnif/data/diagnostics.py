from __future__ import annotations

from pathlib import Path
import json
from malsnif.data.dataset import metadata_path_for


def diagnose_metadata(processed_dir: str | Path, metadata_dir: str | Path | None = None) -> dict:
    processed_dir = Path(processed_dir)
    meta_path = metadata_path_for(processed_dir, metadata_dir=metadata_dir)
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    parse = meta.get("parse_stats", {})
    cdm = parse.get("cdm", {}) or {}
    labeler = parse.get("labeler", {}) or {}
    split_stats = meta.get("split_label_stats", {}) or {}
    stats = meta.get("stats", []) or []
    checks = []
    def add(level: str, message: str):
        checks.append({"level": level, "message": message})

    raw_events = int(parse.get("raw_events_consumed", 0) or 0)
    pos_events = int(parse.get("labeled_events_consumed", 0) or 0)
    if raw_events <= 0:
        add("ERROR", "未消费任何事件；请检查 raw_dir/raw_glob/input_format。")
    if labeler and not labeler.get("has_labels"):
        add("WARN", "没有加载到标签；只能验证构图，不能训练/评估有监督检测。")
    elif pos_events <= 0:
        add("WARN", "已加载标签或配置了标签目录，但没有正样本事件命中。")
    else:
        add("OK", f"标签命中 {pos_events}/{raw_events} 个事件，比例 {pos_events / max(raw_events, 1):.4f}。")

    if int(cdm.get("events_missing_subject", 0) or 0) > 0 or int(cdm.get("events_missing_object", 0) or 0) > 0:
        add("WARN", "存在缺失 subject/object 映射的 CDM 事件；若比例较高，应检查分片顺序或 CDM 文件完整性。")
    elif cdm:
        add("OK", "CDM subject/object 映射完整：events_missing_subject=0 且 events_missing_object=0。")

    if meta.get("config", {}).get("max_events") is not None:
        add("INFO", f"max_events={meta['config']['max_events']}：这是快速验证子集，不代表完整论文数据。")
    if meta.get("num_graphs", 0) < 3:
        add("WARN", "图窗口少于 3 个，无法稳定划分 train/val/test。")

    for split, s in split_stats.items():
        if s.get("num_graphs", 0) and s.get("positive_process_nodes", 0) == 0:
            add("WARN", f"{split} 切分没有正样本进程节点；该切分的节点级指标可能无效。")
        if s.get("num_graphs", 0) and s.get("negative_process_nodes", 0) == 0:
            add("WARN", f"{split} 切分没有负样本进程节点；误报能力无法评估。")
        if s.get("num_graphs", 0) and s.get("positive_process_ratio", 0) > 0.8:
            add("WARN", f"{split} 正样本进程占比 {s.get('positive_process_ratio'):.3f}，F1/Accuracy 容易接近 all-positive 基线。")

    if stats:
        reductions = [x.get("node_reduction_ratio") for x in stats if x.get("node_reduction_ratio") is not None]
        if reductions:
            avg = sum(float(x) for x in reductions) / len(reductions)
            add("INFO", f"平均节点简化比例约 {avg:.3f}。")
        if any(x.get("first_event_time_ns") and x.get("last_event_time_ns") and x["first_event_time_ns"] > x["last_event_time_ns"] for x in stats):
            add("WARN", "存在窗口 first_event_time_ns > last_event_time_ns，说明时间字段解析或文件顺序可能异常。")

    return {"metadata": meta, "metadata_path": str(meta_path), "checks": checks}


def print_diagnosis(processed_dir: str | Path, metadata_dir: str | Path | None = None) -> dict:
    result = diagnose_metadata(processed_dir, metadata_dir=metadata_dir)
    meta = result["metadata"]
    print("== MalSnif processed metadata diagnosis ==")
    print(f"processed_dir: {Path(processed_dir)}")
    print(f"metadata_path: {meta_path if False else result.get('metadata_path', '')}")
    print(f"dataset: {meta.get('dataset_name')}")
    print(f"num_graphs: {meta.get('num_graphs')}")
    print(f"split: {meta.get('split')}")
    print(f"raw_file_sort: {meta.get('raw_file_sort')}")
    raw_files = meta.get("raw_files") or []
    if raw_files:
        print("raw_files(first 10):")
        for f in raw_files[:10]:
            print(f"  - {f}")
    print("split_label_stats:")
    print(json.dumps(meta.get("split_label_stats", {}), ensure_ascii=False, indent=2))
    print("checks:")
    for c in result["checks"]:
        print(f"[{c['level']}] {c['message']}")
    return result
