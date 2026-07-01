from pathlib import Path


def test_cadets_v3_runner_defaults_and_cache():
    text = Path('scripts/run_cadets_v3_ea_verdict.sh').read_text(encoding='utf-8')
    assert 'DATASET_NAME="${DATASET_NAME:-cadets_e3_v3_ea_verdict}"' in text
    assert 'RAW_DIR="${RAW_DIR:-data/raw/darpa_tc/cadets/e3/cdm}"' in text
    assert 'LABEL_DIR="${LABEL_DIR:-data/raw/darpa_tc/cadets/e3/labels}"' in text
    assert 'RAW_GLOB="${RAW_GLOB:-ta1-cadets-e3-official*.json*}"' in text
    assert 'CADETS_EA_PRESET="${CADETS_EA_PRESET:-calib5m}"' in text
    assert 'MAX_EVENTS="${MAX_EVENTS:-5000000}"' in text
    assert 'runs/_cache/cadets_e3_${CADETS_EA_PRESET}_events${MAX_EVENTS}_win${WINDOW_EVENTS}' in text
    assert 'bash scripts/check_cadets_data_layout.sh' in text
    assert 'GRAPH_SIMPLIFY_MODE="${GRAPH_SIMPLIFY_MODE:-leaf}"' in text
    assert "'graph_simplify_mode': os.environ.get('GRAPH_SIMPLIFY_MODE', 'leaf')" in text
    assert "'graph_simplify_risk_threshold': float(os.environ.get('GRAPH_SIMPLIFY_RISK_THRESHOLD', '0.62'))" in text


def test_cadets_v3_runner_default_mechanism_set():
    text = Path('scripts/run_cadets_v3_ea_verdict.sh').read_text(encoding='utf-8')
    assert 'RUN_E0="${RUN_E0:-1}"' in text
    assert 'RUN_E1="${RUN_E1:-1}"' in text
    assert 'RUN_E3="${RUN_E3:-1}"' in text
    assert 'RUN_E5="${RUN_E5:-1}"' in text
    assert 'RUN_E7="${RUN_E7:-1}"' in text
    assert 'RUN_E2="${RUN_E2:-0}"' in text
    assert 'RUN_E4="${RUN_E4:-0}"' in text
    assert 'RUN_E6="${RUN_E6:-0}"' in text
    assert 'RUN_ALL_EA' in text


def test_cadets_layout_checker_documents_expected_files():
    text = Path('scripts/check_cadets_data_layout.sh').read_text(encoding='utf-8')
    assert 'ta1-cadets-e3-official.json' in text
    assert 'ta1-cadets-e3-official-1.json.4' in text
    assert 'ta1-cadets-e3-official-2.json.1' in text
    assert 'cadets.json' in text
    assert 'cadets.txt' in text
