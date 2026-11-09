#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/run_dit_fid.sh" clear "$@"
