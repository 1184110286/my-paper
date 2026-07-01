from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


KEY_FIELDS = [
    "seed", "experiment", "model_variant", "f1", "precision", "recall", "mcc", "roc_auc", "average_precision",
    "tp", "fp", "tn", "fn", "specificity", "threshold", "best_f1", "best_f1_threshold", "best_f1_gap", "predicted_positive_rate", "train_seconds", "ets_tau_mean", "eaw_head_mean", "eha_entropy_mean",
]
METRICS_FOR_CORE = ["f1", "mcc", "average_precision"]
CANONICAL_ORDER = [
    "B0_baseline_gcn", "B1_mcbg_semantic_only", "E0_mcbg_sthgan_no_adapt", "E1_eha_only",
    "E2_eaw_only", "E3_ets_only", "E4_eha_eaw", "E5_eha_ets", "E6_eaw_ets", "E7_eha_ets_eaw",
    "B1_mcbg_semantic_control", "M0_mcbg_sthgan_no_adapt", "M1_context_eha", "M2_calibrated_ets",
    "M3_context_eha_calibrated_ets", "M4_m3_delayed_eaw", "M5_m3_hard_curriculum",
    "M6_m4_hard_curriculum", "M7_m2_ets_hard_curriculum",
]

V3_EXPERIMENTS = {
    "B1_mcbg_semantic_only", "E0_mcbg_sthgan_no_adapt", "E1_eha_only", "E2_eaw_only",
    "E3_ets_only", "E4_eha_eaw", "E5_eha_ets", "E6_eaw_ets", "E7_eha_ets_eaw",
}
V4_EXPERIMENTS = {
    "B1_mcbg_semantic_control", "M0_mcbg_sthgan_no_adapt", "M1_context_eha", "M2_calibrated_ets",
    "M3_context_eha_calibrated_ets", "M4_m3_delayed_eaw", "M5_m3_hard_curriculum",
    "M6_m4_hard_curriculum", "M7_m2_ets_hard_curriculum",
}


def _to_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x):
    try:
        if x is None or x == "":
            return 0
        return int(float(x))
    except Exception:
        return 0


def _read_seed_summaries(root: Path) -> list[dict]:
    rows: list[dict] = []
    for seed_dir in sorted(root.glob("seed_*")):
        summary = seed_dir / "analysis" / "summary.csv"
        # analysis_bundle layout stores summary.csv directly under seed_*;
        # run-root layout stores it under seed_*/analysis/.
        if not summary.exists():
            bundle_summary = seed_dir / "summary.csv"
            summary = bundle_summary if bundle_summary.exists() else summary
        if not summary.exists():
            continue
        seed = seed_dir.name.replace("seed_", "")
        with summary.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["seed"] = seed
                rows.append(row)
    return rows


def _write_summary(rows: list[dict], path: Path) -> None:
    fields = []
    preferred = ["seed", "experiment"] + KEY_FIELDS[2:]
    for k in preferred:
        if any(k in r for r in rows) and k not in fields:
            fields.append(k)
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r.get("experiment", ""), []).append(r)
    return out


def _metric_values(rows: list[dict], metric: str) -> list[float]:
    vals = []
    for r in rows:
        v = _to_float(r.get(metric))
        if v is not None:
            vals.append(v)
    return vals


def _mean_metric(groups: dict[str, list[dict]], name: str, metric: str) -> float | None:
    if name not in groups:
        return None
    vals = _metric_values(groups[name], metric)
    return mean(vals) if vals else None


def _sum_field(rows: list[dict], field: str) -> int:
    return sum(_to_int(r.get(field)) for r in rows)


def _agg_line(name: str, rows: list[dict]) -> str:
    parts = [f"| {name} | {len(rows)}"]
    for m in ["f1", "precision", "recall", "mcc", "average_precision", "train_seconds"]:
        vals = _metric_values(rows, m)
        if vals:
            sd = stdev(vals) if len(vals) > 1 else 0.0
            parts.append(f"{mean(vals):.6f} ± {sd:.6f}")
        else:
            parts.append("—")
    tp, fp, tn, fn = (_sum_field(rows, k) for k in ["tp", "fp", "tn", "fn"])
    parts.append(f"{tp}/{fp}/{tn}/{fn}")
    return " | ".join(parts) + " |"


def _paired_diffs(groups: dict[str, list[dict]], a: str, b: str, metric: str) -> list[tuple[str, float]]:
    if a not in groups or b not in groups:
        return []
    av = {r.get("seed"): _to_float(r.get(metric)) for r in groups[a]}
    bv = {r.get("seed"): _to_float(r.get(metric)) for r in groups[b]}
    diffs = []
    for seed in sorted(set(av) & set(bv)):
        if av[seed] is None or bv[seed] is None:
            continue
        diffs.append((seed, av[seed] - bv[seed]))
    return diffs


def _compare(groups: dict[str, list[dict]], a: str, b: str, metric: str = "f1", eps: float = 0.001, include_missing: bool = False) -> str | None:
    if a not in groups or b not in groups:
        return f"- {a} vs {b}: missing." if include_missing else None
    diffs = _paired_diffs(groups, a, b, metric)
    if not diffs:
        return f"- {a} vs {b}: no paired {metric}." if include_missing else None
    wins = sum(1 for _, d in diffs if d > eps)
    ties = sum(1 for _, d in diffs if abs(d) <= eps)
    losses = sum(1 for _, d in diffs if d < -eps)
    md = mean(d for _, d in diffs)
    verdict = "PASS" if wins >= max(1, len(diffs) * 2 // 3) and md > eps else ("TIE" if ties >= max(wins, losses) else "FAIL")
    detail = ", ".join(f"seed{s}:{d:+.6f}" for s, d in diffs)
    return f"- **{a} vs {b}** on {metric}: **{verdict}**, mean Δ={md:+.6f}; wins/ties/losses={wins}/{ties}/{losses}; {detail}."


def _ordered_names(groups: dict[str, list[dict]]) -> list[str]:
    seen = set(groups)
    ordered = [x for x in CANONICAL_ORDER if x in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _detect_protocol(groups: dict[str, list[dict]]) -> str:
    names = set(groups)
    if names & V4_EXPERIMENTS:
        return "v4"
    return "v3"


def _semantic_control_name(protocol: str) -> str:
    return "B1_mcbg_semantic_control" if protocol == "v4" else "B1_mcbg_semantic_only"


def _nonadaptive_name(protocol: str) -> str:
    return "M0_mcbg_sthgan_no_adapt" if protocol == "v4" else "E0_mcbg_sthgan_no_adapt"


def _ceiling_diagnostics(groups: dict[str, list[dict]], eps: float, protocol: str) -> list[str]:
    lines = []
    names = [n for n in _ordered_names(groups) if n != "B0_baseline_gcn"]
    near = []
    for n in names:
        f1 = _mean_metric(groups, n, "f1")
        ap = _mean_metric(groups, n, "average_precision")
        if f1 is not None and ap is not None and f1 >= 1.0 - eps and ap >= 1.0 - eps:
            near.append(n)
    if len(near) >= 2:
        lines.append("- **Ceiling warning:** multiple non-baseline configurations are at or near perfect F1/AP. Treat mechanism comparisons as error-allocation diagnostics, not as evidence that every adaptive mechanism helps.")
        lines.append(f"- Near-ceiling configs: {', '.join(near)}.")
    # Compare B1 against all EA configs on time and errors.
    semantic_control = _semantic_control_name(protocol)
    if semantic_control in groups:
        b1 = groups[semantic_control]
        b1_time = _mean_metric(groups, semantic_control, "train_seconds") or 0.0
        for n in names:
            if protocol == "v4" and not n.startswith("M"):
                continue
            if protocol == "v3" and not n.startswith("E"):
                continue
            nd = groups[n]
            dt = (_mean_metric(groups, n, "train_seconds") or 0.0) - b1_time
            d_fn = _sum_field(nd, "fn") - _sum_field(b1, "fn")
            d_fp = _sum_field(nd, "fp") - _sum_field(b1, "fp")
            if abs((_mean_metric(groups, n, "f1") or 0) - (_mean_metric(groups, semantic_control, "f1") or 0)) <= eps and dt > 0:
                lines.append(f"- {n} ties B1 on F1 but costs about {dt:.1f}s more on average; do not claim an efficiency or performance gain unless it reduces FP/FN in harder splits.")
            elif d_fn > 0:
                lines.append(f"- {n} introduces {d_fn} extra FN versus B1 across paired seeds; for high-recall APT detection this is a negative signal unless compensated by a large FP reduction.")
    return lines




def _protocol_diagnostics(groups: dict[str, list[dict]], eps: float) -> list[str]:
    lines: list[str] = []
    if not groups:
        return lines
    seed_counts = {name: len({r.get("seed") for r in rows}) for name, rows in groups.items()}
    min_seeds = min(seed_counts.values()) if seed_counts else 0
    if min_seeds < 3:
        lines.append(f"- **Seed warning:** at least one compared configuration has only {min_seeds} seed(s). Treat this as a pilot; confirmatory claims require >=3 seeds.")
    # Warn if validation-selected threshold is much worse than the diagnostic best threshold.
    unstable = []
    overpositive = []
    for name, rows in groups.items():
        gaps = []
        pred_rates = []
        prevalences = []
        for r in rows:
            f1 = _to_float(r.get("f1"))
            best = _to_float(r.get("best_f1"))
            if f1 is not None and best is not None:
                gaps.append(best - f1)
            pr = _to_float(r.get("predicted_positive_rate"))
            pv = _to_float(r.get("prevalence"))
            if pr is not None:
                pred_rates.append(pr)
            if pv is not None:
                prevalences.append(pv)
        if gaps and mean(gaps) > max(0.02, eps * 5):
            unstable.append(f"{name}(mean best-F1 gap={mean(gaps):.4f})")
        if pred_rates and prevalences and mean(pred_rates) > mean(prevalences) * 2.0 and mean(pred_rates) > 0.5:
            overpositive.append(f"{name}(predicted-positive-rate={mean(pred_rates):.3f}, prevalence={mean(prevalences):.3f})")
    if unstable:
        lines.append("- **Threshold warning:** validation-selected thresholds are unstable for " + "; ".join(unstable) + ". Use AP/MCC/best-F1 diagnostics and consider precision_at_recall before claiming a mechanism win.")
    if overpositive:
        lines.append("- **Alert-volume warning:** these configurations predict far more positives than the base rate: " + "; ".join(overpositive) + ". In PIDS settings this can translate into alert fatigue even when recall is high.")
    return lines


def build_report(root: Path, rows: list[dict], eps: float, include_missing: bool = False) -> str:
    groups = _group(rows)
    protocol = _detect_protocol(groups)
    semantic_control = _semantic_control_name(protocol)
    nonadaptive = _nonadaptive_name(protocol)
    lines = []
    lines.append("# EA-THGN Node-Adaptive Mechanism Decision Report")
    lines.append("")
    if protocol == "v4":
        lines.append("This report is for v4 MalSnif-aligned MCBG + ST-HGAN experiments that evaluate Context-EHA, Calibrated-ETS, Delayed-EAW, and hard-negative curriculum variants.")
    else:
        lines.append("This report is for v3 MalSnif-aligned MCBG + ST-HGAN experiments that remove AGF/edge-gate fusion and evaluate EHA/ETS/EAW node-adaptive mechanisms.")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Experiment | Seeds | F1 | Precision | Recall | MCC | AP | Train seconds | TP/FP/TN/FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in _ordered_names(groups):
        lines.append(_agg_line(name, groups[name]))
    lines.append("")
    lines.append("## Ceiling and efficiency diagnostics")
    lines.append("")
    diag = _ceiling_diagnostics(groups, eps, protocol)
    if diag:
        lines.extend(diag)
    else:
        lines.append("- No broad ceiling pattern detected under the configured epsilon.")
    proto = _protocol_diagnostics(groups, eps)
    if proto:
        lines.extend(proto)
    lines.append("")
    lines.append("## Core comparisons")
    lines.append("")
    if protocol == "v4":
        comparisons = [
            (semantic_control, "B0_baseline_gcn", "v4 H1: MCBG semantic control vs MalSnif-like GCN"),
            (nonadaptive, semantic_control, "v4 M0 graph propagation vs semantic control"),
            ("M1_context_eha", nonadaptive, "v4 Context-EHA"),
            ("M2_calibrated_ets", nonadaptive, "v4 Calibrated-ETS"),
            ("M3_context_eha_calibrated_ets", nonadaptive, "v4 Context-EHA + Calibrated-ETS"),
            ("M3_context_eha_calibrated_ets", "M1_context_eha", "v4 interaction over Context-EHA"),
            ("M3_context_eha_calibrated_ets", "M2_calibrated_ets", "v4 interaction over Calibrated-ETS"),
            ("M4_m3_delayed_eaw", "M3_context_eha_calibrated_ets", "v4 Delayed-EAW on top of M3"),
            ("M5_m3_hard_curriculum", "M3_context_eha_calibrated_ets", "v4 hard-negative curriculum on M3"),
            ("M7_m2_ets_hard_curriculum", "M2_calibrated_ets", "v4 hard-negative curriculum on ETS-only"),
            ("M7_m2_ets_hard_curriculum", semantic_control, "v4 ETS+hard curriculum vs semantic control"),
        ]
    else:
        comparisons = [
            (semantic_control, "B0_baseline_gcn", "H1: MCBG semantic encoder vs MalSnif-like GCN"),
            (nonadaptive, semantic_control, "Graph propagation add-on vs semantic-only"),
            ("E1_eha_only", nonadaptive, "EHA single mechanism"),
            ("E2_eaw_only", nonadaptive, "EAW single mechanism"),
            ("E3_ets_only", nonadaptive, "ETS single mechanism"),
            ("E5_eha_ets", nonadaptive, "EHA+ETS vs non-adaptive ST-HGAN"),
            ("E5_eha_ets", "E1_eha_only", "EHA+ETS vs EHA-only"),
            ("E5_eha_ets", "E3_ets_only", "EHA+ETS vs ETS-only"),
            ("E7_eha_ets_eaw", "E5_eha_ets", "Does adding EAW to EHA+ETS help?"),
            ("E7_eha_ets_eaw", semantic_control, "Full EHA+ETS+EAW vs semantic-only"),
        ]
    for a, b, label in comparisons:
        entries = [_compare(groups, a, b, m, eps, include_missing) for m in METRICS_FOR_CORE]
        entries = [e for e in entries if e]
        if not entries:
            continue
        lines.append(f"### {label}")
        lines.extend(entries)
        lines.append("")
    lines.append("## Mechanism interpretation rules")
    lines.append("")
    if protocol == "v4":
        lines.append("- Context-EHA is supported only if M1 > M0 or M3/M4/M5 consistently beat M0 with lower FN or better MCC/AP; otherwise treat deeper adaptive context as unsupported on this split.")
        lines.append("- Calibrated-ETS is supported only if M2 > M0 or M7 consistently beats M0/M2; if the gain is only in AP while F1/MCC are flat or worse, keep the claim narrow.")
        lines.append("- Delayed-EAW is supported only if M4 > M3 or M6 > M5; if it adds FN or extra runtime without cleaner alerts, keep it exploratory.")
        lines.append("- If B1 >= all M-configurations with lower runtime, the main claim should stay with the MCBG semantic encoder and the adaptive graph mechanisms should be reported as non-improving under this split.")
    else:
        lines.append("- EHA is supported only if E1 > E0 or E5/E7 consistently beat E0 with lower FN or better MCC/AP; if B1 and E0/E1 tie, do not claim graph adaptivity helped.")
        lines.append("- ETS is supported only if E3 > E0 or E5/E7 consistently beat E0; if all are at ceiling, inspect FP/FN and selected thresholds rather than small F1 differences.")
        lines.append("- EAW is supported only if E2 > E0 or E7 > E5; if E7 adds FN relative to E5/B1, keep EAW exploratory.")
        lines.append("- If B1 >= all E-configurations with lower runtime, the main claim should stay with the MCBG semantic encoder and EA mechanisms should be reported as non-improving under this split.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--include-missing", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    rows = _read_seed_summaries(root)
    _write_summary(rows, Path(args.summary))
    Path(args.report).write_text(build_report(root, rows, args.eps, include_missing=args.include_missing), encoding="utf-8")
    print({"rows": len(rows), "summary": args.summary, "report": args.report})


if __name__ == "__main__":
    main()
