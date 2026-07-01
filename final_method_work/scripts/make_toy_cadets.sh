#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/raw/toy_cadets"
mkdir -p "$OUT"
cat > "$OUT/toy_cadets.csv" <<'CSV'
Time,SrcId,SrcType,DstId,DstType,EdgeType,Tag
1,/usr/sbin/nginx,PROCESS,/var/log/nginx/access.log,FILE,ReadFile,0
2,/usr/sbin/nginx,PROCESS,/etc/nginx/nginx.conf,FILE,ReadFile,0
3,/usr/sbin/nginx,PROCESS,10.0.0.10:80,NETWORK,TCP Send,0
4,/bin/sh,PROCESS,/tmp/update.sh,FILE,ReadFile,0
5,/bin/sh,PROCESS,/usr/bin/id,PROCESS,Process Create,0
6,/usr/sbin/cron,PROCESS,/var/spool/cron/root,FILE,ReadFile,0
7,/usr/sbin/nginx,PROCESS,/var/www/index.html,FILE,ReadFile,0
8,/usr/sbin/nginx,PROCESS,10.0.0.10:80,NETWORK,TCP Receive,0
9,/usr/sbin/nginx,PROCESS,/var/log/nginx/access.log,FILE,WriteFile,0
10,/usr/sbin/nginx,PROCESS,/etc/hosts,FILE,ReadFile,0
11,/usr/sbin/nginx,PROCESS,10.0.0.10:80,NETWORK,TCP Send,0
12,/usr/sbin/nginx,PROCESS,/var/www/logo.png,FILE,ReadFile,0
13,/usr/local/bin/drakon,PROCESS,/tmp/clean,FILE,WriteFile,1
14,/usr/local/bin/drakon,PROCESS,/bin/chmod,PROCESS,Process Create,1
15,/usr/local/bin/drakon,PROCESS,161.116.88.72:4444,NETWORK,TCP Connect,1
16,/tmp/clean,PROCESS,/var/log/xdev,FILE,WriteFile,1
17,/tmp/clean,PROCESS,161.116.88.72:4444,NETWORK,TCP Receive,1
18,/tmp/clean,PROCESS,/usr/bin/profile,FILE,WriteFile,1
19,/usr/sbin/nginx,PROCESS,/var/log/nginx/error.log,FILE,WriteFile,0
20,/usr/sbin/sshd,PROCESS,/etc/ssh/sshd_config,FILE,ReadFile,0
21,/usr/sbin/sshd,PROCESS,10.0.0.20:22,NETWORK,TCP Receive,0
22,/usr/sbin/sshd,PROCESS,/var/log/auth.log,FILE,WriteFile,0
23,/usr/bin/profile,PROCESS,/usr/bin/mail,PROCESS,Process Create,1
24,/usr/bin/profile,PROCESS,10.0.0.21:22,NETWORK,TCP Connect,1
25,/usr/bin/profile,PROCESS,10.0.0.22:22,NETWORK,TCP Connect,1
26,/usr/bin/profile,PROCESS,10.0.0.23:22,NETWORK,TCP Connect,1
27,/usr/bin/mail,PROCESS,/var/mail/root,FILE,WriteFile,1
28,/usr/sbin/nginx,PROCESS,/var/www/index.html,FILE,ReadFile,0
29,/usr/sbin/nginx,PROCESS,10.0.0.10:80,NETWORK,TCP Send,0
30,/usr/sbin/cron,PROCESS,/usr/bin/updatedb,PROCESS,Process Create,0
31,/usr/bin/updatedb,PROCESS,/var/lib/mlocate/mlocate.db,FILE,WriteFile,0
32,/usr/bin/updatedb,PROCESS,/usr,FILE,ReadFile,0
33,/usr/local/bin/drakon,PROCESS,/etc/passwd,FILE,ReadFile,1
34,/usr/local/bin/drakon,PROCESS,/etc/shadow,FILE,ReadFile,1
35,/usr/local/bin/drakon,PROCESS,161.116.88.72:4444,NETWORK,TCP Send,1
36,/usr/sbin/nginx,PROCESS,/var/log/nginx/access.log,FILE,WriteFile,0
CSV
echo "Toy cadets-like data written to $OUT/toy_cadets.csv"
