"""GFN2-xTB relaxed scan of the TS hydride transfer, both attack faces.

Coordinate: d(H...C7), the forming bond, from 2.8 to 1.1 A. Everything else
relaxes, so the C6-S cleavage is free to couple to the hydride transfer.
-> results/ts_scan.json (energies, key distances, xyz for every point)
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit.enzyme.build import build, to_xyz
from chemoqbit.enzyme.xtb import optimise

OUT = ROOT / 'results' / 'ts_scan.json'
res = json.loads(OUT.read_text()) if OUT.exists() else {}
grid = np.round(np.arange(2.8, 1.05, -0.1), 2)

for face in ('anti', 'syn'):
    if face in res and len(res[face]['points']) == len(grid):
        continue
    sym, X, key, q = build(face)
    d = lambda Y, a, b: float(np.linalg.norm(Y[key[a]] - Y[key[b]]))
    pts = []
    for tgt in grid:
        X, E = optimise(sym, X, q, constraints=[(key['Hd'], key['C7'], float(tgt))])
        pts.append(dict(d_HC7=float(tgt), E=float(E), d_CdH=d(X, 'Cd', 'Hd'),
                        d_C6S=d(X, 'C6', 'S'), d_CgS=d(X, 'Cg', 'S'),
                        xyz=to_xyz(sym, X, f'{face} d(H-C7)={tgt}')))
        print(f"{face} d(H-C7)={tgt:.2f} E={(E - pts[0]['E']) * 627.51:7.2f} kcal/mol "
              f"Cd-H={pts[-1]['d_CdH']:.2f} C6-S={pts[-1]['d_C6S']:.2f} Cg..S={pts[-1]['d_CgS']:.2f}",
              flush=True)
    res[face] = dict(key=key, charge=q, points=pts)
    OUT.write_text(json.dumps(res, indent=1))
