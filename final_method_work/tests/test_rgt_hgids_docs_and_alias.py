from pathlib import Path


def test_rgt_hgids_docs_exist_and_name_is_consistent():
    docs = [
        Path('docs/rgt_hgids_method_overview.md'),
        Path('docs/rgt_hgids_paper_section_draft.md'),
        Path('docs/rgt_hgids_experiment_protocol.md'),
        Path('docs/rgt_hgids_references.md'),
        Path('docs/paper_assets/rgt_hgids_mermaid_flow.mmd'),
    ]
    for p in docs:
        assert p.exists(), p
        text = p.read_text(encoding='utf-8')
        assert 'RGT-HGIDS' in text or p.suffix == '.mmd'


def test_rgt_hgids_alias_scripts_delegate_to_existing_runners():
    rigorous = Path('scripts/run_rgt_hgids_rigorous.sh').read_text(encoding='utf-8')
    quick = Path('scripts/run_rgt_hgids_quick.sh').read_text(encoding='utf-8')
    assert 'run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh' in rigorous
    assert 'run_e1_rgd_bigru_tbb_rr_theia_cadets.sh' in quick
    assert 'target_boundary' in rigorous
    assert 'rgd_bigru' in rigorous
