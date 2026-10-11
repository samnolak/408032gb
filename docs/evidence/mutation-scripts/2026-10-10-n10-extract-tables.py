"""N10 helper: prints the per-layer table, the summaries, the non-zero discrete-choice rows and the four largest rows of the
first table of each model from a tools/tinygen/run_parity_f16fa.sh (or tests-only) log. Usage: <script> <log>"""
import re, sys
lines = open(sys.argv[1]).read().splitlines()
for model in ("qwen4exp", "glm5-next"):
    print(f"--- {model}")
    # per-layer table
    i = next(k for k, l in enumerate(lines) if l.startswith(f"{model}: worst error per layer"))
    j = i
    while j < len(lines) and lines[j].strip():
        print(lines[j]); j += 1
    # summaries
    i = next(k for k, l in enumerate(lines) if l.startswith(f"{model}: summaries"))
    print(lines[i])
    j = i + 1
    while j < len(lines) and re.match(r"^(pf64|ub16|ub10|ub18|decode|rollback) ", lines[j]):
        if "tie replays = 0" not in lines[j]:
            print(lines[j])
        j += 1
    # discrete
    i = next(k for k, l in enumerate(lines) if l.startswith(f"{model}: rows where"))
    print(lines[i])
    j, zero = i + 1, 0
    while j < len(lines) and re.match(r"^(pf64|ub16|ub10|ub18|decode|rollback) +.* +\d+$", lines[j]):
        if lines[j].split()[-1] != "0":
            print(lines[j])
        else:
            zero += 1
        j += 1
    print(f"(and {zero} rows with 0, not shown)")
    # top aware errors of the first table
    i = next(k for k, l in enumerate(lines) if l.startswith(f"{model}, llama.cpp"))
    rows = []
    j = i + 2
    while j < len(lines) and lines[j].strip():
        p = lines[j].split()
        try:
            rows.append((float(p[-2]), lines[j]))
        except ValueError:
            pass
        j += 1
    print(f"largest errors against the precision-aware oracle (first table, {len(rows)} rows):")
    for v, l in sorted(rows, reverse=True)[:4]:
        print(l)
