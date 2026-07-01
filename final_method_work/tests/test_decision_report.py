import csv
from pathlib import Path

from malsnif.decision_report import load_rows, write_report, write_summary


def test_decision_report_marks_tie_and_missing(tmp_path: Path):
    seed = tmp_path / 'seed_42' / 'analysis'
    seed.mkdir(parents=True)
    summary = seed / 'summary.csv'
    with summary.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'f1', 'precision', 'recall'])
        w.writeheader()
        w.writerow({'experiment': 'A0_baseline_gcn', 'f1': '0.80', 'precision': '0.9', 'recall': '0.7'})
        w.writerow({'experiment': 'A3_static_concat', 'f1': '0.9000', 'precision': '0.9', 'recall': '0.9'})
        w.writerow({'experiment': 'A4_agf_st_hgan_mcbg', 'f1': '0.9004', 'precision': '0.9', 'recall': '0.9'})
    rows = load_rows(tmp_path)
    assert len(rows) == 3
    out_csv = write_summary(rows, tmp_path / 'next_summary.csv')
    assert out_csv.exists()
    report = write_report(rows, tmp_path / 'report.md', eps=0.001, include_missing=True)
    text = report.read_text(encoding='utf-8')
    assert 'TIE: A4_agf_st_hgan_mcbg - A3_static_concat' in text
    assert 'not run / missing metric' in text


def test_decision_report_includes_edge_gated_v2_comparison(tmp_path: Path):
    seed = tmp_path / 'seed_42' / 'analysis'
    seed.mkdir(parents=True)
    summary = seed / 'summary.csv'
    with summary.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'f1', 'precision', 'recall', 'edge_gate_mean'])
        w.writeheader()
        w.writerow({'experiment': 'B0_baseline_gcn', 'f1': '0.90', 'precision': '0.9', 'recall': '0.9'})
        w.writerow({'experiment': 'B2_mcbg_sthgan_no_gate', 'f1': '0.92', 'precision': '0.9', 'recall': '0.9'})
        w.writerow({'experiment': 'B3_edge_gated_sthgan_mcbg', 'f1': '0.94', 'precision': '0.9', 'recall': '0.9', 'edge_gate_mean': '0.51'})
    rows = load_rows(tmp_path)
    report = write_report(rows, tmp_path / 'report.md', eps=0.001, include_missing=True)
    text = report.read_text(encoding='utf-8')
    assert 'v2 edge gate: B3 vs B2' in text
    assert 'PASS: B3_edge_gated_sthgan_mcbg - B2_mcbg_sthgan_no_gate' in text
    assert 'B3 edge_gate_mean' in text


def test_v2_only_report_does_not_emit_a4_missing_lines(tmp_path: Path):
    seed = tmp_path / 'seed_42' / 'analysis'
    seed.mkdir(parents=True)
    summary = seed / 'summary.csv'
    with summary.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'f1', 'precision', 'recall'])
        w.writeheader()
        w.writerow({'experiment': 'B0_baseline_gcn', 'f1': '0.86', 'precision': '0.78', 'recall': '0.95'})
        w.writerow({'experiment': 'B1_mcbg_semantic_only', 'f1': '0.99', 'precision': '1.0', 'recall': '0.98'})
        w.writerow({'experiment': 'B2_mcbg_sthgan_no_gate', 'f1': '0.988', 'precision': '1.0', 'recall': '0.97'})
        w.writerow({'experiment': 'B3_edge_gated_sthgan_mcbg', 'f1': '0.9875', 'precision': '1.0', 'recall': '0.97'})
    rows = load_rows(tmp_path)
    report = write_report(rows, tmp_path / 'report.md', eps=0.001, include_missing=True)
    text = report.read_text(encoding='utf-8')
    assert 'A4 vs A0-GCN baseline' not in text
    assert 'not run / missing metric: A4_agf_st_hgan_mcbg' not in text
    assert 'v2 H1 is supported' in text
