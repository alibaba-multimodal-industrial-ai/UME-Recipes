#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
exec