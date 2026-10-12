#!/usr/bin/env bash
# cold-start bound runs (commit with the `cold` column); same format as jobs.sh
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad/traces-analysis
SE="python3 scripts/traces/cachesim_sensitivity.py"
for me in qwen3-coder-30b-a3b:128 glm-4.7-flash:64; do
  m=${me%%:*}; E=${me##*:}; T=$S/$m/traces
  printf '%s\t%s\n' "$m/se_cold_last_full" "$SE $T --n-experts $E --capacity 0.607,0.75,1.0 --policies lru_global,lru_device"
  printf '%s\t%s\n' "$m/se_cold_folds_genonly" "$SE $T --n-experts $E --capacity 0.607,1.0 --policies static,lru,hybrid,lru_global --fold all --test-view gen_only"
  for N in 30 64; do
    printf '%s\t%s\n' "$m/se_cold_folds_genonly_gencap$N" "$SE $T --n-experts $E --capacity 0.607,1.0 --policies static,lru,hybrid,lru_global --fold all --test-view gen_only --gen-cap $N"
  done
done
