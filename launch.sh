#!/usr/bin/env bash
set -euo pipefail
simplegen_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$simplegen_dir"
if [[ ! -x .venv/bin/python ]]; then
  simplegen_python=""
  for candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3,11))'; then
      simplegen_python="$candidate"
      break
    fi
  done
  if [[ -z "$simplegen_python" ]]; then
    echo "simpleGen needs Python 3.11 or newer. Install it, then launch again."
    exit 1
  fi
  "$simplegen_python" -m venv .venv
  .venv/bin/python -m pip install -e .
fi
exec .venv/bin/python -m simplegen "$@"
