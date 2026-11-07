#!/usr/bin/env bash
# Activate zerowam first. No Conda activation/module/GPU assumptions here.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
exec python "$SCRIPT_DIR/download.py" "$@"
