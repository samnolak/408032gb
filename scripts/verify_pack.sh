#!/usr/bin/env bash
# Verification pack (ROADMAP R6.1): thin wrapper around tools/verify/pack.py; every argument is passed through.
#   scripts/verify_pack.sh --runs RUNS_DIR --out PACK_DIR [--criteria FILE] [--hardlink]
#   scripts/verify_pack.sh --check PACK_DIR
# Exit codes: build 0 = PASS, 1 = FAIL, 3 = NOT RUN, 2 = usage/IO error; --check 0 = intact, 1 = mismatch, 2 = no manifest.
# Docs: tools/verify/README.md
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$here/../tools/verify/pack.py" "$@"
