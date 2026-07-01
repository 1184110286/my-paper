from pathlib import Path


def test_rgd_bigru_one_key_script_has_expected_controls():
    text = Path("scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets.sh").read_text(encoding="utf-8")
    assert "REDUNDANCY_MODE=\"target_boundary\"" in text
    assert "rgd_bigru_mcbg" in text
    assert "RUN_DATASETS=\"${RUN_DATASETS:-cadets theia}\"" in text
    assert "E1_RGD_BIGRU_TBB_RR_SUMMARY.csv" in text
    assert "analysis_bundle" in text
    assert "metrics = payload.get(\"metrics\", payload)" in text
    assert "label=rgd_bigru" in text


def test_cadets_v3_runner_exports_rgd_config_values():
    text = Path("scripts/run_cadets_v3_ea_verdict.sh").read_text(encoding="utf-8")
    assert "RGD_KERNEL_SIZE" in text
    assert "rgd_dilations" in text
    assert "rgd_residual_scale_init" in text
    assert "rgd_depthwise_separable" in text
