#!/usr/bin/env bash
# tier_cost.py (GLM-5.3-Flash preset, n_tok 1, 262,144 ctx) with prediction recall/precision: old assumptions vs the
# pregate values measured on the GLM-4.7-Flash proxy (data/traces/real/glm-4.7-flash/predict.txt, gen rows).
# precision of an m-candidate set = recall * k / m. W = d x 0.44 ms (R8's one-layer lead at 50 tok/s).
cd /home/user/wt-research || exit 1
run() {  # label, then tier_cost --set arguments
  local label=$1; shift
  python3 tools/ceilings/tier_cost.py --model glm "$@" --json | python3 -c '
import json, sys
r = json.load(sys.stdin)["rows"][0]
print("%-46s %6.1f tok/s  step %6.2f ms  copies/layer %.2f  t_cpu %.3f ms  t_ssd %.3f ms" % (sys.argv[1], r["tok_s"], r["step_ms"], r["copies"], r["t_cpu"] * 1e3, r["t_ssd"] * 1e3))' "$label"
}
PRED=(
 "none|"
 "old R8: p .85 pi .50 d=1|--set p_recall=0.85 --set pi_prec=0.5 --set W=0.44e-3"
 "old 4b: p .95 pi .90 W 1.3 ms|--set p_recall=0.95 --set pi_prec=0.9 --set W=1.3e-3"
 "proxy pregate d=1 m=k: p .814 pi .814|--set p_recall=0.814 --set pi_prec=0.814 --set W=0.44e-3"
 "proxy pregate d=1 m=2k: p .945 pi .473|--set p_recall=0.945 --set pi_prec=0.473 --set W=0.44e-3"
 "proxy pregate d=2 m=k: p .742 pi .742|--set p_recall=0.742 --set pi_prec=0.742 --set W=0.88e-3"
 "proxy pregate d=2 m=2k: p .895 pi .448|--set p_recall=0.895 --set pi_prec=0.448 --set W=0.88e-3"
 "Strata probe, off-VRAM set: p .30 pi .08 d=1|--set p_recall=0.30 --set pi_prec=0.08 --set W=0.44e-3"
)
for hw in "default (32 thr x 8 GB/s)|" "32 thr x 3.6 GB/s|--set b_core=3.6e9 --set n_cpu=32" "46 thr x 3.6 GB/s|--set b_core=3.6e9 --set n_cpu=46"; do
  hwl=${hw%%|*}; hwa=${hw#*|}
  for hv in ${HV:-0.61 0.75}; do
    hr=$(python3 -c "print(round(1-$hv, 6))")
    echo "## CPU experts: $hwl; h_v $hv h_r $hr; 8 ch DRAM; GPU prefetch (prefetch_opt)"
    for p in "${PRED[@]}"; do
      # shellcheck disable=SC2086
      run "${p%%|*}" $hwa --set h_v=$hv --set h_r=$hr ${p#*|}
    done
  done
done
echo "## NVMe -> RAM staging only (prefetch_to=ram): h_v 0.61, h_r 0.34, 5% of routed experts on NVMe, defaults"
for p in "${PRED[@]}"; do
  # shellcheck disable=SC2086
  run "${p%%|*}" --set h_v=0.61 --set h_r=0.34 --set prefetch_to=ram ${p#*|}
done
