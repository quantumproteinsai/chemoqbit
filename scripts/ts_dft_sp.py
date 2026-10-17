"""DFT//xTB energy profile for the TS hydride transfer (reduced model).

B3LYP-D3(BJ)/6-31G(d), C-PCM eps = 4, single points on selected points of
the GFN2-xTB xi-scan (xi = d(Cd-H) - d(H-C7)), plus the xTB reactant, TS and
product complexes. Checkpointed per point -> results/ts_dft_sp.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pyscf import gto, dft, lib

lib.num_threads(2)
src = json.loads((ROOT / 'results' / 'ts_kie.json').read_text())
OUT = ROOT / 'results' / 'ts_dft_sp.json'
res = json.loads(OUT.read_text()) if OUT.exists() else {}

jobs = [('reactant', src['reactant']['xyz']), ('xts', src['ts']['xyz']), ('product', src['product']['xyz'])]
for p in src['xi_scan']:
    if round(p['xi'], 2) in (-1.6, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8, 1.0, 1.6):
        jobs.append((f"xi{p['xi']:+.2f}", p['xyz']))

for name, xyz in jobs:
    if name in res:
        continue
    t0 = time.time()
    atom = '\n'.join(xyz.strip().split('\n')[2:])
    mol = gto.M(atom=atom, basis='6-31g*', charge=1, verbose=0)
    mf = dft.RKS(mol); mf.xc = 'b3lyp'; mf.disp = 'd3bj'
    mf = mf.PCM(); mf.with_solvent.eps = 4.0
    mf.conv_tol = 1e-8
    mf.kernel()
    res[name] = dict(E=float(mf.e_tot), converged=bool(mf.converged), seconds=time.time() - t0)
    OUT.write_text(json.dumps(res, indent=1))
    print(f'{name:9s} E={mf.e_tot:.6f} conv={mf.converged} ({time.time() - t0:.0f}s)', flush=True)
