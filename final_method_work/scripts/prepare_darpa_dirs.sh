#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for d in \
  data/raw/darpa_tc/cadets/e3/cdm \
  data/raw/darpa_tc/cadets/e3/labels \
  data/raw/darpa_tc/fivedirections/e3/cdm \
  data/raw/darpa_tc/fivedirections/e3/labels \
  data/raw/darpa_tc/theia/e3/cdm \
  data/raw/darpa_tc/theia/e3/labels \
  data/raw/darpa_tc/trace/e3/cdm \
  data/raw/darpa_tc/trace/e3/labels \
  data/raw/darpa_tc/labels; do
  mkdir -p "$d"
done

cat > data/raw/darpa_tc/cadets/e3/cdm/README.files.md <<'TXT'
CADETS-E3 CDM file placement. Put the official DARPA TC CADETS E3 CDM JSON shards here without renaming.
Default matcher used by scripts/run_cadets_v3_ea_verdict.sh:
  ta1-cadets-e3-official*.json*
Common shard names look like:
  ta1-cadets-e3-official.json
  ta1-cadets-e3-official.json.1
  ta1-cadets-e3-official.json.2
  ta1-cadets-e3-official-1.json
  ta1-cadets-e3-official-1.json.1
  ta1-cadets-e3-official-1.json.2
  ta1-cadets-e3-official-1.json.3
  ta1-cadets-e3-official-1.json.4
  ta1-cadets-e3-official-2.json
  ta1-cadets-e3-official-2.json.1
Compressed equivalents such as .json.gz/.json.1.gz are accepted by the parser.
Run scripts/check_cadets_data_layout.sh to generate a manifest of exact files found.
TXT

cat > data/raw/darpa_tc/cadets/e3/labels/README.labels.md <<'TXT'
将 DARPA/论文标注整理为以下任意文件即可：
- malicious_uuids.txt：每行一个恶意 subject/object/event UUID。
- malicious_paths.txt：每行一个恶意路径/命令/IP 片段；正则请写成 re:<pattern>。
- malicious_time_ranges.csv：start_ns,end_ns,uuid,path,event_type,description。
- malicious_events.csv：可包含 event_uuid,subject_uuid,object_uuid,path,event_type,start_ns,end_ns 等列。
TXT

cat > data/raw/darpa_tc/theia/e3/labels/README.labels.md <<'TXT'
THEIA-E3 label placement.  Put at least one effective label source here:
- theia.json: original/permissive JSON ground-truth file for THEIA.
- malicious_uuids.txt: one malicious CDM UUID per line.
- malicious_paths.txt: one malicious path/command/IP substring per line; regex uses re:<pattern>.
- malicious_event_types.txt: one event type per line, e.g. EVENT_WRITE.
- malicious_time_ranges.csv: start_ns,end_ns[,uuid,path,event_type,description].
- malicious_events.csv: flexible CSV with event_uuid/subject_uuid/object_uuid/path/event_type/start_ns/end_ns.
TXT
cat > data/raw/darpa_tc/theia/e3/cdm/README.files.md <<'TXT'
THEIA-E3 CDM file placement.  Put the official DARPA TC THEIA E3 CDM JSON shards here without renaming.
Default matcher used by scripts/run_theia_v2_edge_gate_verdict.sh:
  ta1-theia-e3-official*.json*
Common shard names look like:
  ta1-theia-e3-official.json
  ta1-theia-e3-official.json.1
  ta1-theia-e3-official.json.2
  ta1-theia-e3-official-1.json
  ta1-theia-e3-official-1.json.1
  ta1-theia-e3-official-2.json
Compressed equivalents such as .json.gz/.json.1.gz are accepted by the parser.
Run scripts/check_theia_data_layout.sh to generate a manifest of the exact files found on your machine.
TXT
echo "DARPA raw-data directories created under data/raw/darpa_tc/."
