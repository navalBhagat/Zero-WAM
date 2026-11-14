#!/usr/bin/env bash
# Dry-run unless --submit is passed; all extra arguments go to the suite launcher.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
exec python3 "$SCRIPT_DIR/robotwin_suite.py" textcheck "$@"
