#!/usr/bin/env bash
# strata-glm: ярусный путь (VRAM + pinned-RAM + prefetch) на Linux. Замерено 03.10 на GPU2:
# decode 19.65 tok/s, prompt 15.23 tok/s (промпт 40 токенов, per-token) на тестовом паке.
# Подробная инструкция: ../docs/strata-tiered-run.md
#
# Память: ярус ~142 GiB pinned (загрузка ~175 s при тёплом page cache) + ~18 GiB VRAM.
# Нужно свободной RAM >= ~166 GiB. Если «cannot reserve the host tier» — RAMGIB=120.
#
# ВНИМАНИЕ: движок читает только pack-формат (dense.txt + experts.bin), GGUF не открывает.
set -uo pipefail
GPU=${GPU:-2}
PACK=${PACK:-/mnt/data/apps/glm-synthetic/pack}
PROFILE=${PROFILE:-/mnt/data/apps/glm-synthetic/profile.bin}
TOKENS=${TOKENS:-/mnt/data/apps/glm-synthetic/pack/tokens.txt}
BIN=${BIN:-/mnt/data/apps/strata-glm-build/strata-glm}
MAXNEW=${MAXNEW:-32}
RAMGIB=${RAMGIB:-}
ARGS=(--pack "$PACK" --tokens "$TOKENS" --chunk 0 --max-new "$MAXNEW"
      --skip-disk 0 --skip-ram 0 --dense-fp4 all --profile "$PROFILE")
[ -z "$RAMGIB" ] || ARGS+=(--ram-gib "$RAMGIB")
exec env CUDA_VISIBLE_DEVICES="$GPU" GLM_TIMING=1 "$BIN" "${ARGS[@]}"
