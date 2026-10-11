#!/usr/bin/env bash
# Read-only check of the six GLM-5.3-Flash UD-Q4_K_XL GGUF shards on a host (epyc-ai): sizes, sha256 against the N15
# pins, GGUF header facts (architecture string, expert_group_count, chat template sha256). Writes only into --out
# (new or empty, outside --dir); no downloads, no network. All logic and the pins: glm_gguf_check.py next to this file.
# Usage: tools/hostcheck/glm_gguf_check.sh --dir <dir with the shards> --out <new dir> [--jobs N] [--no-hash]
# Example on epyc-ai:
#   tools/hostcheck/glm_gguf_check.sh \
#     --dir /mnt/data/home/grishberg/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL --out "$HOME/glm-gguf-check-$(date -u +%Y%m%dT%H%M%SZ)"
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# -I: isolated mode, nothing is imported from the current directory or PYTHON* variables
exec python3 -I "$HERE/glm_gguf_check.py" "$@"
