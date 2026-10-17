#!/usr/bin/env bash
# Run the whole thymidylate synthase cluster pipeline, checkpointed.
# Safe to re-run: every stage skips work already in results/ts_cluster_run.json.
#
#   bash run_nuc.sh                 # everything
#   bash run_nuc.sh 0,1,2           # selected stages
set -uo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python
NC=$(nproc)
LOG=results/nuc_$(date +%Y%m%d_%H%M).log
mkdir -p results
STAGES="${1:-}"

run() {            # run <threads> <stages...>
    local t=$1; shift
    echo "=== $(date +%H:%M:%S)  stages $*  (OMP_NUM_THREADS=$t)" | tee -a "$LOG"
    OMP_NUM_THREADS=$t $PY -u scripts/ts_cluster_nuc.py --stages "$@" 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
}

if [ -n "$STAGES" ]; then
    run 1 "$STAGES"
    exit $?
fi

# xTB stages: single-threaded is fastest for a cluster this size
run 1 0,1,2,3,4 || { echo "xTB stages failed; see $LOG" | tee -a "$LOG"; exit 1; }

# DFT: use the cores. def2-SVP first, then def2-TZVP on the same geometries.
run "$NC" dft --basis def2-svp  || echo "def2-SVP DFT failed, continuing" | tee -a "$LOG"
OMP_NUM_THREADS=$NC $PY -u scripts/ts_cluster_nuc.py --stages dft --basis def2-tzvp 2>&1 | tee -a "$LOG"

# active-space circuits
run "$NC" active || echo "active-space stage failed" | tee -a "$LOG"

echo
echo "Finished. Send me results/ts_cluster_run.json (and $LOG if anything failed)."
$PY - <<'PY'
import json, pathlib
p = pathlib.Path('results/ts_cluster_run.json')
if p.exists():
    r = json.loads(p.read_text())
    print('stages present:', [k for k in r if k != 'prep'])
    a = r.get('analysis')
    if a:
        print(f"barrier {a['dE_barrier']:.2f} kcal/mol, reaction {a['dE_rxn']:.2f}")
        for k, v in a['kie'].items():
            print(f"  KIE {k}: semiclassical {v['semiclassical']:.2f}  "
                  f"Wigner {v['wigner']:.2f}  Eckart {v['eckart']:.2f}")
        print('  experiment: H/T = 6.91 +- 0.05 at 293 K')
PY
