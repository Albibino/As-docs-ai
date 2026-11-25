#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHONPATH="$root/src" PYTHONDONTWRITEBYTECODE=1 python3 -P "$root/deploy/render_aws_oauth.py"
