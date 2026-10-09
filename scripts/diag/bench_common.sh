# shellcheck shell=bash
# Shared helpers for scripts/diag/bench_*.sh (sourced, not executed).
# run_logged writes the same section format as collect.sh probes, so summarize.py and a
# human reader see the literal command line, UTC start, exit code and full output.

bench_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
bench_ts() { date -u +%Y%m%dT%H%M%SZ; }
bench_host() { printf '%s' "$(uname -n 2>/dev/null || echo unknown-host)" | tr -c 'A-Za-z0-9._-' '_'; }

# run_logged FILE CMD [ARGS...]
# Runs CMD with stdin from /dev/null, stores everything in FILE, echoes stdout/stderr
# to the console afterwards, returns CMD's exit code.
run_logged() {
  local file=$1 start t0 t1 rc name
  shift
  name=$(basename "$file" .txt)
  start=$(bench_utc)
  t0=$(date +%s)
  echo ">>> $(printf '%q ' "$@")"
  "$@" </dev/null >"$file.stdout.part" 2>"$file.stderr.part"
  rc=$?
  t1=$(date +%s)
  {
    echo "===== PROBE ====="
    echo "name: $name"
    echo "started_utc: $start"
    echo "cwd: $(pwd)"
    echo "===== COMMAND ====="
    printf '%q ' "$@" | sed 's/ $//'
    echo
    echo "===== STDOUT ====="
    cat "$file.stdout.part"
    [ -s "$file.stdout.part" ] && [ -n "$(tail -c 1 "$file.stdout.part")" ] && echo
    echo "===== STDERR ====="
    cat "$file.stderr.part"
    [ -s "$file.stderr.part" ] && [ -n "$(tail -c 1 "$file.stderr.part")" ] && echo
    echo "===== RESULT ====="
    echo "exit_code: $rc"
    echo "duration_s: $((t1 - t0))"
    if [ "$rc" -eq 0 ]; then echo "status: ok"; else echo "status: failed"; fi
  } > "$file"
  cat "$file.stdout.part"
  cat "$file.stderr.part" >&2
  rm -f "$file.stdout.part" "$file.stderr.part"
  echo "<<< exit $rc ($((t1 - t0)) s), log: $file"
  return "$rc"
}

# find_nvcc [EXPLICIT] -> prints the path of nvcc or nothing
find_nvcc() {
  local c
  for c in "${1:-}" "$(command -v nvcc 2>/dev/null)" /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc /opt/cuda/bin/nvcc; do
    [ -n "$c" ] && [ -x "$c" ] && { printf '%s\n' "$c"; return 0; }
  done
  return 1
}
