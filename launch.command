#!/usr/bin/env bash
set -euo pipefail
simplegen_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "$simplegen_dir/launch.sh" "$@"
