#!/usr/bin/env bash
# Restart the image lab on port 6008: stop the process listening on 6008, start start.sh in the background.
# Running jobs are interrupted (the service marks them failed on startup).
set -u
cd "$(dirname "$0")"
# anchored pattern: must not match the shell that runs this script
PID=$(pgrep -f '^python -m uvicorn app:app .*--port 6008' | head -1)
if [ -n "$PID" ]; then
  kill "$PID"
  for i in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
fi
LOG=../logs/student_lab_6008_$(date +%Y%m%d_%H%M%S).log
(setsid nohup bash start.sh > "$LOG" 2>&1 < /dev/null &)
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:6008/api/health > /dev/null && { echo "started, log $LOG"; exit 0; }; sleep 1; done
echo "not up after 60 s, see $LOG"; exit 1
