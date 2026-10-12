"""N16: compare the dumps of tools/tinygen/run_parity_mimo2_decode.sh at the two pins, file by file (same run, graph and
tensor name), and the logits / h rows. usage: 2026-10-11-n16-pindiff.py <work at ec7630a>/n16 <work at 79e2e74>/n16
(and again with .../n16/mxfp4 for the quantized runs)."""
import filecmp, pathlib, sys
a, b = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for run in sorted(p.name for p in a.iterdir() if p.is_dir()):
    for sub in ([run] + ([f"{run}/mtp"] if (a / run / "mtp").is_dir() else [])):
        fa = {p.relative_to(a / sub) for p in (a / sub).glob("g*/*.bin")}
        fb = {p.relative_to(b / sub) for p in (b / sub).glob("g*/*.bin")}
        both = sorted(fa & fb)
        diff = [p for p in both if not filecmp.cmp(a / sub / p, b / sub / p, shallow=False)]
        extra = [str(p) for p in sorted(fa ^ fb)]
        names = sorted({p.name.split("__")[0] for p in (fa ^ fb)})
        lo = [f for f in ("logits.bin", "h_nextn.bin", "h_in.bin") if (a / sub / f).exists()]
        lod = [f for f in lo if not filecmp.cmp(a / sub / f, b / sub / f, shallow=False)]
        print(f"{sub:16s} files in both {len(both):6d}, differing {len(diff):4d}; only at one pin {len(extra):4d} {names}; "
              f"{', '.join(lo)} differing: {lod or 'none'}")
        if diff:
            print("    e.g.", [str(p) for p in diff[:5]])
