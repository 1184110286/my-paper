from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

DEFAULT_FIELDS = [
    'seed', 'experiment', 'model_variant', 'semantic_encoder', 'fusion_mode',
    'f1', 'precision', 'recall', 'specificity', 'mcc', 'roc_auc', 'average_precision',
    'threshold', 'num_samples', 'prevalence', 'predicted_positive_rate',
    'best_f1', 'best_f1_threshold', 'best_f1_precision', 'best_f1_recall', 'best_f1_gap',
    'tp', 'fp', 'tn', 'fn', 'best_epoch', 'best_val_f1', 'train_seconds',
    'max_cuda_peak_allocated_mb', 'max_cuda_peak_reserved_mb',
    'gate_semantic_mean', 'gate_semantic_std', 'edge_gate_mean', 'edge_gate_std',
    'attention_kept_ratio', 'warning_count', 'warnings', 'run_root',
]

CORE_EXPERIMENTS = [
    'A0_baseline_gcn', 'A0_baseline_graphsage', 'A1_mcbg_semantic_only',
    'A2_hgan_structure_only', 'A3_static_concat', 'A4_agf_st_hgan_mcbg',
    'A5_no_time_bias', 'A6_hard_pruning', 'A7_scalar_gate',
    'B0_baseline_gcn', 'B1_mcbg_semantic_only', 'B2_mcbg_sthgan_no_gate',
    'B3_edge_gated_sthgan_mcbg', 'B4_edge_gated_no_time',
    'B5_edge_gated_no_relation', 'B6_edge_scalar_gate', 'B7_edge_gate_no_edge_semantics',
]

V2_COMPARISONS = [
    ('B1_mcbg_semantic_only', 'B0_baseline_gcn', 'v2 H1: MCBG semantic-only vs B0-GCN'),
    ('B2_mcbg_sthgan_no_gate', 'B0_baseline_gcn', 'v2 H2: ST-HGAN no-gate vs B0-GCN'),
    ('B2_mcbg_sthgan_no_gate', 'B1_mcbg_semantic_only', 'v2 structural add-on: B2 no-gate vs B1 semantic-only'),
    ('B3_edge_gated_sthgan_mcbg', 'B2_mcbg_sthgan_no_gate', 'v2 edge gate: B3 vs B2'),
    ('B3_edge_gated_sthgan_mcbg', 'B1_mcbg_semantic_only', 'v2 full edge-gated model vs semantic-only'),
    ('B3_edge_gated_sthgan_mcbg', 'B4_edge_gated_no_time', 'v2 time bias contribution'),
    ('B3_edge_gated_sthgan_mcbg', 'B5_edge_gated_no_relation', 'v2 relation type contribution'),
    ('B3_edge_gated_sthgan_mcbg', 'B6_edge_scalar_gate', 'v2 vector edge gate vs scalar edge gate'),
]

V1_COMPARISONS = [
    ('A4_agf_st_hgan_mcbg', 'A0_baseline_gcn', 'A4 vs A0-GCN baseline'),
    ('A4_agf_st_hgan_mcbg', 'A3_static_concat', 'H3: vector gate vs static concat'),
    ('A4_agf_st_hgan_mcbg', 'A5_no_time_bias', 'time bias contribution'),
    ('A4_agf_st_hgan_mcbg', 'A6_hard_pruning', 'soft pruning contribution'),
    ('A4_agf_st_hgan_mcbg', 'A7_scalar_gate', 'vector gate vs scalar gate'),
]


def _active_comparisons(experiments: set[str]) -> list[tuple[str, str, str]]:
    has_v2 = any(e.startswith('B') for e in experiments)
    has_v1 = any(e.startswith('A') for e in experiments)
    comps: list[tuple[str, str, str]] = []
    if has_v2:
        comps.extend(V2_COMPARISONS)
    if has_v1:
        comps.extend(V1_COMPARISONS)
    return comps


def _as_float(x: Any) -> float | None:
    if x is None or x == '':
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _fmt_float(x: float | None, digits: int = 6) -> str:
    if x is None:
        return ''
    return f'{x:.{digits}f}'


def load_rows(root: str | Path) -> list[dict[str, str]]:
    root = Path(root)
    rows: list[dict[str, str]] = []
    for seed_dir in sorted(root.glob('seed_*')):
        seed = seed_dir.name.replace('seed_', '')
        summary = seed_dir / 'analysis' / 'summary.csv'
        if not summary.exists():
            summary = seed_dir / 'summary.csv'
        if not summary.exists():
            continue
        with summary.open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                row = {str(k): '' if v is None else str(v) for k, v in row.items()}
                row['seed'] = seed
                row.setdefault('run_root', str(seed_dir))
                rows.append(row)
    return rows


def write_summary(rows: list[dict[str, str]], out_path: str | Path, fields: list[str] | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or DEFAULT_FIELDS
    with out_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fields})
    return out_path


def _by_seed(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        out[str(row.get('seed', ''))][str(row.get('experiment', ''))] = row
    return dict(out)


def _values(rows: list[dict[str, str]], exp: str, metric: str) -> list[float]:
    vals = []
    for row in rows:
        if row.get('experiment') != exp:
            continue
        v = _as_float(row.get(metric))
        if v is not None:
            vals.append(v)
    return vals


def _compare(exps: dict[str, dict[str, str]], a: str, b: str, metric: str = 'f1', eps: float = 1e-3) -> tuple[str, float | None]:
    av = _as_float(exps.get(a, {}).get(metric))
    bv = _as_float(exps.get(b, {}).get(metric))
    if av is None or bv is None:
        missing = []
        if av is None:
            missing.append(a)
        if bv is None:
            missing.append(b)
        return f'not run / missing metric: {", ".join(missing)}', None
    delta = av - bv
    if delta > eps:
        state = 'PASS'
    elif delta < -eps:
        state = 'FAIL'
    else:
        state = 'TIE'
    return f'{state}: {a} - {b} = {delta:+.6f} ({metric})', delta


def _mean_metric(rows: list[dict[str, str]], exp: str, metric: str) -> float | None:
    vals = _values(rows, exp, metric)
    return mean(vals) if vals else None


def _wins(vals: list[float], eps: float) -> tuple[int, int, int]:
    return sum(v > eps for v in vals), sum(abs(v) <= eps for v in vals), sum(v < -eps for v in vals)


def write_report(rows: list[dict[str, str]], out_path: str | Path, eps: float = 1e-3, include_missing: bool = False) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_seed = _by_seed(rows)
    experiments = sorted({r.get('experiment', '') for r in rows if r.get('experiment')})
    exp_set = set(experiments)
    active_comparisons = [c for c in _active_comparisons(exp_set) if include_missing or (c[0] in exp_set and c[1] in exp_set)]

    with out_path.open('w', encoding='utf-8') as f:
        f.write('# Mechanism decision report\n\n')
        f.write('This report is generated from validation-selected test metrics. It is diagnostic, not a paper-level conclusion.\n\n')
        f.write(f'- comparison_epsilon={eps}\n')
        f.write(f'- seeds={", ".join(sorted(by_seed)) if by_seed else "none"}\n')
        f.write(f'- experiments_run={", ".join(experiments) if experiments else "none"}\n\n')

        f.write('## Mean metrics across available seeds\n\n')
        f.write('| experiment | seeds | mean F1 | std F1 | mean best-F1 | mean Precision | mean Recall | mean AP | mean MCC | mean train seconds |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for exp in CORE_EXPERIMENTS:
            if exp not in experiments:
                continue
            f1 = _values(rows, exp, 'f1')
            bf1 = _values(rows, exp, 'best_f1')
            prec = _values(rows, exp, 'precision')
            rec = _values(rows, exp, 'recall')
            ap = _values(rows, exp, 'average_precision')
            mcc = _values(rows, exp, 'mcc')
            sec = _values(rows, exp, 'train_seconds')
            f.write(
                f'| {exp} | {len(f1)} | {_fmt_float(mean(f1) if f1 else None)} | '
                f'{_fmt_float(pstdev(f1) if len(f1) > 1 else 0.0 if f1 else None)} | '
                f'{_fmt_float(mean(bf1) if bf1 else None)} | {_fmt_float(mean(prec) if prec else None)} | '
                f'{_fmt_float(mean(rec) if rec else None)} | {_fmt_float(mean(ap) if ap else None)} | '
                f'{_fmt_float(mean(mcc) if mcc else None)} | {_fmt_float(mean(sec) if sec else None, 2)} |\n'
            )
        f.write('\n')

        f.write('## Threshold and imbalance diagnostics\n\n')
        any_diag = False
        threshold_unstable = False
        sparse_eval = False
        for exp in CORE_EXPERIMENTS:
            if exp not in experiments:
                continue
            prev = _values(rows, exp, 'prevalence')
            gap = _values(rows, exp, 'best_f1_gap')
            pp = _values(rows, exp, 'predicted_positive_rate')
            if not prev and not gap and not pp:
                continue
            any_diag = True
            if prev and mean(prev) < 0.01:
                sparse_eval = True
            if gap and mean(gap) > 0.05:
                threshold_unstable = True
            f.write(
                f'- {exp}: mean_prevalence={_fmt_float(mean(prev) if prev else None)}, '
                f'mean_predicted_positive_rate={_fmt_float(mean(pp) if pp else None)}, '
                f'mean_oracle_best_f1_gap={_fmt_float(mean(gap) if gap else None)}'
                f'{"  ⚠ threshold-sensitive" if gap and mean(gap) > 0.05 else ""}\n'
            )
        if not any_diag:
            f.write('- no prevalence/best-F1 diagnostics available in summary.csv.\n')
        f.write('\n')
        if sparse_eval:
            f.write('**Caution:** at least one evaluated split has prevalence < 1%; treat validation-selected F1 as an operating-point diagnostic, and compare AP/ROC/MCC/ranking metrics before making mechanism claims.\n\n')
        if threshold_unstable:
            f.write('**Caution:** at least one model has a large current-F1 vs oracle best-F1 gap on the test split. Do not convert current-threshold wins into mechanism claims without AP/MCC/best-F1 agreement.\n\n')

        f.write('## Per-seed mechanism comparisons\n\n')
        if not active_comparisons:
            f.write('- no comparable experiment pairs are available in this run.\n\n')
        aggregate: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for seed, exps in sorted(by_seed.items()):
            f.write(f'### seed {seed}\n\n')
            for a, b, label in active_comparisons:
                line, delta = _compare(exps, a, b, metric='f1', eps=eps)
                f.write(f'- {label}: {line}\n')
                if delta is not None:
                    aggregate[f'{a}__minus__{b}']['f1'].append(delta)
                for metric, name in [('best_f1', 'oracle/best-F1 check'), ('average_precision', 'AP/ranking check'), ('mcc', 'MCC check')]:
                    m_line, m_delta = _compare(exps, a, b, metric=metric, eps=eps)
                    if m_delta is not None:
                        f.write(f'  - {name}: {m_line}\n')
                        aggregate[f'{a}__minus__{b}'][metric].append(m_delta)
            b3 = exps.get('B3_edge_gated_sthgan_mcbg')
            if b3:
                gm = b3.get('edge_gate_mean') or b3.get('gate_semantic_mean', '')
                gs = b3.get('edge_gate_std') or b3.get('gate_semantic_std', '')
                f.write(f'- B3 edge_gate_mean={gm}, edge_gate_std={gs}\n')
            f.write('\n')

        f.write('## Aggregate decision hints\n\n')
        for a, b, label in active_comparisons:
            key = f'{a}__minus__{b}'
            vals = aggregate.get(key, {}).get('f1', [])
            if not vals:
                f.write(f'- {label}: not enough data.\n')
                continue
            wins, ties, losses = _wins(vals, eps)
            msg = (f'- {label}: current-threshold F1 mean_delta={mean(vals):+.6f}, '
                   f'std_delta={(pstdev(vals) if len(vals)>1 else 0.0):.6f}, wins/ties/losses={wins}/{ties}/{losses}.')
            # Add robustness checks if available.
            parts = []
            for metric in ['best_f1', 'average_precision', 'mcc']:
                mvals = aggregate.get(key, {}).get(metric, [])
                if mvals:
                    mw, mt, ml = _wins(mvals, eps)
                    parts.append(f'{metric}: mean_delta={mean(mvals):+.6f}, W/T/L={mw}/{mt}/{ml}')
            if parts:
                msg += ' Robustness checks -> ' + '; '.join(parts) + '.'
            f.write(msg + '\n')
        f.write('\n')

        exps_all = {r.get('experiment') for r in rows}
        has_v2 = any(str(e).startswith('B') for e in exps_all)
        has_v1 = any(str(e).startswith('A') for e in exps_all)
        f.write('## Conservative interpretation\n\n')

        def ag_delta(a: str, b: str, metric: str) -> list[float]:
            return aggregate.get(f'{a}__minus__{b}', {}).get(metric, [])

        def strong_positive(a: str, b: str) -> bool:
            # In sparse/threshold-unstable runs, require current-F1, AP and MCC to agree.
            fvals = ag_delta(a, b, 'f1')
            if not fvals:
                return False
            fw, ft, fl = _wins(fvals, eps)
            if fw < max(1, math.ceil(len(fvals) * 2 / 3)):
                return False
            if sparse_eval or threshold_unstable:
                # Sparse THEIA pilots are highly sensitive to the selected operating
                # threshold.  A mechanism should not be called positive unless the
                # current-threshold F1, the oracle/best-F1 diagnostic, AP ranking and
                # MCC all move in the same direction when those columns are present.
                for metric in ['best_f1', 'average_precision', 'mcc']:
                    mvals = ag_delta(a, b, metric)
                    if mvals:
                        mw, mt, ml = _wins(mvals, eps)
                        if mw < max(1, math.ceil(len(mvals) * 2 / 3)):
                            return False
            return True

        if has_v2:
            if strong_positive('B1_mcbg_semantic_only', 'B0_baseline_gcn'):
                f.write('- v2 H1 is supported: MCBG semantic-only improves over B0-GCN under available metrics.\n')
            else:
                f.write('- v2 H1 is not yet robust under the current sparse/calibrated diagnostics; inspect B1 vs B0 AP/MCC and threshold stability.\n')
            if strong_positive('B2_mcbg_sthgan_no_gate', 'B0_baseline_gcn'):
                f.write('- v2 H2 has a positive signal versus the weak B0-GCN baseline: MCBG + ST-HGAN no-gate improves over B0-GCN.\n')
            else:
                f.write('- v2 H2 is not robust yet: B2 no-gate does not clearly beat B0-GCN across robustness checks.\n')
            if strong_positive('B2_mcbg_sthgan_no_gate', 'B1_mcbg_semantic_only'):
                f.write('- ST-HGAN no-gate adds value beyond MCBG semantic-only under robust checks.\n')
            else:
                f.write('- ST-HGAN no-gate is not robustly better than MCBG semantic-only; treat graph propagation as auxiliary unless later high-recall/full-window runs reverse this.\n')
            if 'B3_edge_gated_sthgan_mcbg' in exps_all:
                if strong_positive('B3_edge_gated_sthgan_mcbg', 'B2_mcbg_sthgan_no_gate'):
                    f.write('- Edge-gated message passing is better than no-gate propagation across available robust checks; continue stricter validation.\n')
                else:
                    f.write('- Edge gate is not robustly better than no-gate propagation; do not claim edge-gate performance contribution yet.\n')
                if strong_positive('B3_edge_gated_sthgan_mcbg', 'B1_mcbg_semantic_only'):
                    f.write('- Full edge-gated graph propagation is robustly better than semantic-only on available checks.\n')
                else:
                    f.write('- Full edge-gated graph propagation is not robustly better than semantic-only; keep MCBG semantic encoder as the safer claim unless later runs reverse this.\n')
            else:
                f.write('- Edge-gated B3 was intentionally not evaluated in this run; no edge-gate performance claim should be made from this report.\n')
            if len(by_seed) < 3:
                f.write('- Only one seed is available; conclusions are pilot diagnostics, not confirmatory evidence.\n')
        if has_v1:
            if strong_positive('A4_agf_st_hgan_mcbg', 'A0_baseline_gcn'):
                f.write('- A4 is better than A0-GCN under robust checks.\n')
            else:
                f.write('- A4 is not robustly better than A0-GCN.\n')
            if strong_positive('A4_agf_st_hgan_mcbg', 'A3_static_concat'):
                f.write('- A4 is better than A3 static concat under robust checks.\n')
            else:
                f.write('- A4 is not robustly better than A3 static concat; do not claim adaptive-gate performance contribution.\n')
    return out_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description='Build cross-seed summary and mechanism decision report.')
    p.add_argument('--root', required=True, help='Parent run directory containing seed_* subdirectories.')
    p.add_argument('--summary', default=None, help='Output combined CSV path.')
    p.add_argument('--report', default=None, help='Output mechanism report path.')
    p.add_argument('--eps', type=float, default=1e-3, help='Small delta treated as a tie.')
    p.add_argument('--include-missing', action='store_true', help='Report missing planned comparisons instead of skipping them.')
    args = p.parse_args(argv)

    root = Path(args.root)
    rows = load_rows(root)
    summary = Path(args.summary) if args.summary else root / 'next_summary.csv'
    report = Path(args.report) if args.report else root / 'MECHANISM_DECISION_REPORT.md'
    write_summary(rows, summary)
    write_report(rows, report, eps=float(args.eps), include_missing=bool(args.include_missing))
    print(json.dumps({'next_summary': str(summary), 'mechanism_report': str(report), 'rows': len(rows)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
