#!/usr/bin/env bash
# emits one job per line: <model>/<name>\t<command>   (run from /home/user/wt-research)
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad/traces-analysis
CS="python3 tools/g0/cachesim.py"; SE="python3 scripts/traces/cachesim_sensitivity.py"
BASE="--capacity 0.1,0.25,0.5,0.75 --holdout-every 4"
FINE="--capacity 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.546,0.55,0.6,0.607,0.65,0.7,0.746,0.75,0.8,0.85,0.9 --holdout-every 4"
TGT="--capacity 0.25,0.5,0.546,0.607,0.746"
for me in qwen3-coder-30b-a3b:128 glm-4.7-flash:64; do
  m=${me%%:*}; E=${me##*:}; T=$S/$m/traces
  printf '%s\t%s\n' "$m/cs_fine" "$CS $T --n-experts $E $FINE -o $S/out/$m/cs_fine.json"
  printf '%s\t%s\n' "$m/cs_warm" "$CS $T --n-experts $E $BASE --warm"
  printf '%s\t%s\n' "$m/cs_session" "$CS $T --n-experts $E $BASE --session"
  printf '%s\t%s\n' "$m/cs_warm_session" "$CS $T --n-experts $E $BASE --warm --session"
  printf '%s\t%s\n' "$m/cs_prompt_updates" "$CS $T --n-experts $E $BASE --prompt-updates"
  for D in 1 2 3 4 8; do
    printf '%s\t%s\n' "$m/cs_devices$D" "$CS $T --n-experts $E --capacity 0.25,0.5,0.607,0.75 --holdout-every 4 --policies lru_global,lru_device --devices $D"
    printf '%s\t%s\n' "$m/cs_devices${D}_prompt_updates" "$CS $T --n-experts $E --capacity 0.25,0.5,0.607,0.75 --holdout-every 4 --policies lru_global,lru_device --devices $D --prompt-updates"
  done
  printf '%s\t%s\n' "$m/se_folds_full" "$SE $T --n-experts $E $TGT --fold all -o $S/out/$m/se_folds_full.json"
  printf '%s\t%s\n' "$m/se_folds_genonly" "$SE $T --n-experts $E $TGT --fold all --test-view gen_only -o $S/out/$m/se_folds_genonly.json"
  printf '%s\t%s\n' "$m/se_folds_genonly_warm" "$SE $T --n-experts $E $TGT --fold all --test-view gen_only --warm"
  printf '%s\t%s\n' "$m/se_folds_full_warm" "$SE $T --n-experts $E $TGT --fold all --warm"
  printf '%s\t%s\n' "$m/se_pertrace" "$SE $T --n-experts $E --capacity 0.5,0.607 --per-trace"
  printf '%s\t%s\n' "$m/se_folds_traingen" "$SE $T --n-experts $E $TGT --fold all --policies static,hybrid --train-part gen"
  printf '%s\t%s\n' "$m/se_folds_trainprompt" "$SE $T --n-experts $E $TGT --fold all --policies static,hybrid --train-part prompt"
  for N in 30 64; do
    printf '%s\t%s\n' "$m/se_folds_full_gencap$N" "$SE $T --n-experts $E --capacity 0.5,0.607 --fold all --gen-cap $N"
    printf '%s\t%s\n' "$m/se_folds_genonly_gencap$N" "$SE $T --n-experts $E --capacity 0.5,0.607 --fold all --test-view gen_only --gen-cap $N"
  done
done
