"""Build the 1TLS active-site cluster, relax only the hydrogens, report geometry.

-> results/ts_cluster.json   (xyz, frozen heavy-atom mask, key atom indices)
   results/ts_cluster.xyz
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rdkit import Chem
from chemoqbit.enzyme.cluster import build, add_hydrogens, to_xyz
from chemoqbit.enzyme.xtb import energy_grad

CHARGE = 1
sym0, X0, label, _ = build(ROOT / 'data/1TLS.pdb')
mh = add_hydrogens(sym0, X0, label, charge=CHARGE)
conf = mh.GetConformer()
sym = [a.GetSymbol() for a in mh.GetAtoms()]
X = np.array([list(conf.GetAtomPosition(i)) for i in range(mh.GetNumAtoms())])
heavy = np.array([s != 'H' for s in sym])
n_heavy = int(heavy.sum())
# labels: the first len(label) atoms are the heavy ones, in build() order
lab = label + [f'H{i}' for i in range(len(sym) - len(label))]
idx = {l: i for i, l in enumerate(label)}
print(f'{len(sym)} atoms ({n_heavy} heavy), charge {CHARGE}')

# --- relax hydrogens only -----------------------------------------------
free = np.flatnonzero(~heavy)
x0 = X[free].ravel()


def f(v):
    Y = X.copy(); Y[free] = v.reshape(-1, 3)
    E, G = energy_grad(sym, Y, CHARGE)
    return E, G[free].ravel()


r = minimize(f, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=3000, gtol=1e-4))
X[free] = r.x.reshape(-1, 3)
print(f'H relaxation: E = {r.fun:.6f} Eh, {r.nit} iterations')

d = lambda a, b: float(np.linalg.norm(X[a] - X[b]))
Cd, C11, C5u, C6u, S = (idx['folate:C6'], idx['methylene:C11'], idx['nucleotide:C5'],
                        idx['nucleotide:C6'], idx['cys146:SG'])
# the transferring hydride: the H on folate C6 nearest the acceptor C11
cand = [i for i in free if d(i, Cd) < 1.25]
Hd = min(cand, key=lambda i: d(i, C11))
key = dict(Cd=Cd, Hd=int(Hd), C11=C11, C5u=C5u, C6u=C6u, S=S,
           Cg=idx['arg166:CZ'], N5=idx['folate:N5'])
print(f'donor-acceptor C6...C11 {d(Cd, C11):.2f} A   C-H {d(Cd, Hd):.2f}   '
      f'H...C11 {d(Hd, C11):.2f}   C6u-S {d(C6u, S):.2f}   C5=C11 {d(C5u, C11):.2f}')
print('other H on donor C:', [round(d(i, C11), 2) for i in cand if i != Hd])

out = dict(symbols=sym, coords=X.tolist(), charge=CHARGE, labels=lab,
           frozen_heavy=[bool(v) for v in heavy], key=key, n_heavy=n_heavy,
           source='PDB 1TLS chain A, hydrides/caps added, H positions relaxed at GFN2-xTB')
(ROOT / 'results' / 'ts_cluster.json').write_text(json.dumps(out, indent=1))
(ROOT / 'results' / 'ts_cluster.xyz').write_text(
    f'{len(sym)}\nTS active-site cluster from 1TLS (charge +1)\n' +
    '\n'.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in zip(sym, X)) + '\n')
print('written: results/ts_cluster.json, results/ts_cluster.xyz')
