#!/usr/bin/env bash
# Functional test of the Linux DiskReader (patch 0003) on real files, CPU only.
#   tests/strata_glm/run_diskreader_test.sh <strata-glm checkout with patches applied>
set -euo pipefail
SRC=${1:?strata-glm checkout}; HERE=$(cd "$(dirname "$0")" && pwd); W=$(mktemp -d)
python3 - "$SRC/src/glm/glm_main.cpp" "$W/diskreader.inc" <<'PY'
import sys
s = open(sys.argv[1]).read()
a = s.index("class DiskReader {"); b = s.index("\n};\n", a) + 4
open(sys.argv[2], "w").write(s[a:b])
PY
sed "s#/tmp/drtest#$W#g" "$HERE/test_diskreader.cpp" > "$W/test.cpp"
g++ -std=c++17 -O1 -Wall -Werror -pthread -I"$W" "$W/test.cpp" -o "$W/test"
"$W/test"
