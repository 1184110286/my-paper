from pathlib import Path
import subprocess


def test_theia_wra_runner_is_three_way_and_uses_theia_defaults():
    script = Path("scripts/run_theia_wra_rr_verdict.sh").read_text(encoding="utf-8")
    assert 'data/raw/darpa_tc/theia/e3/cdm' in script
    assert 'data/raw/darpa_tc/theia/e3/labels' in script
    assert 'ta1-theia-e3-official*.json*' in script
    assert 'run_mode "off" "off"' in script
    assert 'run_mode "prefix_tree" "prefix_tree"' in script
    assert 'run_mode "winnowing_anchor" "winnowing_anchor"' in script
    assert 'run_mode "first_last_boundary"' not in script
    assert 'run_mode "family_run_cap"' not in script
    assert 'collect_theia_wra_rr_analysis_bundle.sh' in script
    assert 'check_theia_data_layout.sh' in script
    assert 'single_folder_analysis_bundle' in script
    assert '[send me]' in script

    assert 'CADETS_CACHE_ROOT="$cache_root"' in script
    assert 'THEIA_CACHE_ROOT' not in script or 'unset THEIA_CACHE_ROOT' in script
    assert 'CACHE_MODE_VALIDATION.json' in script
    assert 'node_event_reduction_ratio' in script


def test_theia_collect_bundle_flattens_child_seed_outputs(tmp_path):
    run_root = tmp_path / "runs" / "theia_wra_rr_demo"
    exp = run_root / "experiment"
    for mode in ["off", "prefix_tree", "winnowing_anchor"]:
        seed_dir = exp / mode / "analysis_bundle" / "seed_42" / "experiments" / "E1_eha_only"
        seed_dir.mkdir(parents=True)
        (seed_dir / "metrics_test_compact.json").write_text(
            '{"metrics":{"f1":1.0,"precision":1.0,"recall":1.0,"mcc":1.0,"average_precision":1.0,"roc_auc":1.0,"tp":1,"fp":0,"tn":1,"fn":0,"threshold":0.5,"best_f1":1.0,"best_f1_threshold":0.5,"num_samples":2}}',
            encoding="utf-8",
        )
    (exp / "summary_theia_wra_rr.csv").write_text("label,f1\n", encoding="utf-8")
    subprocess.run(["bash", "scripts/collect_theia_wra_rr_analysis_bundle.sh", str(run_root)], check=True)
    collected = run_root / "analysis_bundle" / "collected"
    assert collected.exists()
    assert (collected / "experiment__summary_theia_wra_rr.csv").exists()
    assert any("mode-winnowing_anchor__seed_42" in p.name and p.name.endswith("metrics_test_compact.json") for p in collected.iterdir())
    assert (collected / "MANIFEST.txt").exists()
    assert not any("checkpoints" in str(p) or "graph_cache" in str(p) for p in collected.rglob("*"))
