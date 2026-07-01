from pathlib import Path


def test_tbb_scripts_exist_and_use_expected_modes():
    for path in [
        Path("scripts/run_cadets_tbb_rr_verdict.sh"),
        Path("scripts/run_theia_tbb_rr_verdict.sh"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert "run_mode \"off\"" in text
        assert "run_mode \"prefix_tree\"" in text
        assert "run_mode \"target_boundary\"" in text
        assert "winnowing_anchor" not in text
        assert "TBB_RR_TARGET_COMPRESSION" in text


def test_tbb_collectors_use_target_boundary_fallback():
    for path in [
        Path("scripts/collect_tbb_rr_analysis_bundle.sh"),
        Path("scripts/collect_theia_tbb_rr_analysis_bundle.sh"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert "target_boundary" in text
        assert "winnowing_anchor" not in text
        assert "summary_tbb_rr" in text or "summary_theia_tbb_rr" in text
