#!/usr/bin/env bash
# Bootstrap chemoqbit in a fresh Debian/Ubuntu LXC container.
#
#   bash setup_nuc.sh                      # clone from GitHub
#   bash setup_nuc.sh /path/to/chemoqbit   # use an existing copy (unzipped)
#
# Two traps this works around:
#   * Debian patches setuptools so geomeTRIC's legacy build fails with
#     "AttributeError: install_layout". A venv with fresh setuptools fixes it.
#   * tblite (GFN2-xTB) is SLOWER multi-threaded on systems this size:
#     0.13 s per gradient at OMP_NUM_THREADS=1 against 1.75 s at 2.
#     The run script therefore pins xTB stages to one thread.
set -euo pipefail

SRC="${1:-}"
REPO="${CHEMOQBIT_REPO:-https://github.com/quantumproteinsai/chemoqbit.git}"
DEST="${CHEMOQBIT_DIR:-$HOME/chemoqbit}"

echo "== system packages"
if command -v apt-get >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq python3 python3-venv python3-dev build-essential \
        git curl ca-certificates tmux
fi

echo "== source"
if [ -n "$SRC" ]; then
    [ "$(readlink -f "$SRC")" = "$(readlink -f "$DEST")" ] || cp -r "$SRC" "$DEST"
else
    [ -d "$DEST/.git" ] && git -C "$DEST" pull --ff-only || git clone "$REPO" "$DEST"
fi
cd "$DEST"

echo "== virtual environment"
python3 -m venv --system-site-packages .venv
./.venv/bin/pip install -q -U pip setuptools wheel        # the geomeTRIC fix
./.venv/bin/pip install -q numpy scipy pyscf geometric rdkit tblite pytest \
    pyscf-dispersion fastapi uvicorn httpx

echo "== versions"
./.venv/bin/python - <<'PY'
import numpy, scipy, pyscf, geometric, rdkit, tblite
print(f'  numpy {numpy.__version__}  scipy {scipy.__version__}  pyscf {pyscf.__version__}')
print(f'  geometric {geometric.__version__}  rdkit {rdkit.__version__}')
PY

echo "== tests"
./.venv/bin/python -m pytest -q tests

echo "== thread timing (xTB is often fastest single-threaded)"
for t in 1 2 4; do
  OMP_NUM_THREADS=$t ./.venv/bin/python - <<PY 2>/dev/null
import time, numpy as np
from chemoqbit.enzyme.build import build
from chemoqbit.enzyme.xtb import energy_grad
s, X, k, q = build('anti'); energy_grad(s, X, q)
t0 = time.time(); [energy_grad(s, X + 0.001 * i, q) for i in range(5)]
print(f'  OMP_NUM_THREADS=$t: {(time.time() - t0) / 5:.3f} s per xTB gradient')
PY
done

cat <<EOF

Ready in $DEST

Next (see NUC.md for what each stage does):

  cd $DEST
  tmux new -s chemoqbit          # so it survives a disconnect
  bash run_nuc.sh                # all stages, checkpointed

or stage by stage:

  OMP_NUM_THREADS=1 ./.venv/bin/python scripts/ts_cluster_nuc.py --stages 0,1,2,3,4
  OMP_NUM_THREADS=\$(nproc) ./.venv/bin/python scripts/ts_cluster_nuc.py --stages dft
EOF
