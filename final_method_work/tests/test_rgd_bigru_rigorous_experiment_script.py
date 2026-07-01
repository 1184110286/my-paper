from pathlib import Path


def test_rgd_bigru_rigorous_script_has_strict_controls_and_bundle():
    text = Path("scripts/run_e1_rgd_bigru_tbb_rr_theia_cadets_rigorous.sh").read_text(encoding="utf-8")
    assert "EXPERIMENT_LEVEL=\"${EXPERIMENT_LEVEL:-rigorous}\"" in text
    assert "CADETS_EA_PRESET:=calib12m" in text
    assert "SEEDS:=42 43 44 45 46" in text
    assert "EPOCHS:=15" in text
    assert "REDUNDANCY_MODE=\"target_boundary\"" in text
    assert "RUN_B0=0 RUN_B1=0 RUN_E0=0 RUN_E1=1" in text
    assert "rgd_bigru_mcbg" in text
    assert "E1_RGD_BIGRU_TBB_RR_RIGOROUS_PAIRED_DELTAS.csv" in text
    assert "E1_RGD_BIGRU_TBB_RR_RIGOROUS_PAIRED_AGG.csv" in text
    assert "bootstrap_ci" in text
    assert "exact_two_sided_sign_p" in text
    assert "key_files_flat" in text
    assert "analysis_bundle" in text


def test_seed_helper_supports_deterministic_mode():
    text = Path("malsnif/utils/seed.py").read_text(encoding="utf-8")
    assert "MALSNIF_DETERMINISTIC" in text
    assert "torch.backends.cudnn.deterministic = True" in text
    assert "torch.use_deterministic_algorithms(True, warn_only=True)" in text
