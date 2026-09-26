#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/Qwen-Image-2.1/lab/env.sh
cd "$LAB/student-lab"
exec python -m uvicorn app:app --host 127.0.0.1 --port 6008 --workers 1
