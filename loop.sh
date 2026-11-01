#!/usr/bin/env bash
# Every 10 min: rebuild index.html and push, until machine D's queue ends (then one last push).
cd "$(dirname "$0")"
PY=/c/Users/ggnat/anaconda3/envs/sfr-ag/python.exe
D_LOG=/c/Users/ggnat/Documents/Claude/2026-10-31/galaxy-sfr-automl/results/logs/run_D_console.log
while true; do
  "$PY" -X utf8 build.py >/dev/null 2>&1
  git add -A index.html data >/dev/null 2>&1
  git diff --cached --quiet || { git commit -q -m "Progress update $(date '+%Y-%m-%d %H:%M')" && git push -q origin main; }
  grep -qE "^J17 (finished|FAILED|already done)" "$D_LOG" 2>/dev/null && break
  sleep 600
done
echo "progress loop finished $(date)"
