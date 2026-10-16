"""Case study 2: 5-fluorouracil lactam -> 4-lactim (N3-H -> O4 proton shift).

The O4 lactim is the minor tautomer implicated in 5-FU mispairing; the
direct gas-phase 1,3 shift is the textbook reference path.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rdkit import Chem
from rdkit.Chem import AllChem
from chemoqbit.workflow import run_case

m = Chem.AddHs(Chem.MolFromSmiles('O=C1NC=C(F)C(=O)N1'))
AllChem.EmbedMolecule(m, randomSeed=7)
AllChem.MMFFOptimizeMolecule(m)
X = m.GetConformer().GetPositions()
sym = [a.GetSymbol() for a in m.GetAtoms()]

# N3: the N bonded to two carbonyl carbons; O4: carbonyl O on the C bonded to C5(F)
def nbrs(i):
    return [n.GetIdx() for n in m.GetAtomWithIdx(i).GetNeighbors()]

def is_carbonyl_c(i):
    return sym[i] == 'C' and any(sym[j] == 'O' for j in nbrs(i))

N3 = next(i for i in range(len(sym)) if sym[i] == 'N'
          and sum(is_carbonyl_c(j) for j in nbrs(i)) == 2)
H3 = next(j for j in nbrs(N3) if sym[j] == 'H')
C4 = next(j for j in nbrs(N3) if is_carbonyl_c(j)
          and any(sym[k] == 'C' and any(sym[l] == 'F' for l in nbrs(k)) for k in nbrs(j)))
O4 = next(j for j in nbrs(C4) if sym[j] == 'O')

# bridging position: midpoint of N3/O4, pushed outward from C4 in the ring plane
mid = (X[N3] + X[O4]) / 2
out = mid - X[C4]; out /= np.linalg.norm(out)
h_ts = mid + 0.55 * out

def with_h(pos):
    Y = X.copy(); Y[H3] = pos
    return [(s, Y[i]) for i, s in enumerate(sym)]

def toward(a, b, d):
    v = b - a; return a + d * v / np.linalg.norm(v)

def frame(c, o, n):
    ex = (o - c) / np.linalg.norm(o - c)
    v = n - c; ey = v - v.dot(ex) * ex; ey /= np.linalg.norm(ey)
    return ex, ey, np.cross(ex, ey)


def mapped_ts_h():
    """Bridging H placed as in the converged formamide TS, expressed in the
    local C-O-N frame and transferred to C4-O4-N3 of 5-FU."""
    import json
    f = json.loads((ROOT / 'results/formamide_geom.json').read_text())['ts']['atoms']
    c, o, n, hb = (np.array(f[i][1]) for i in (0, 1, 2, 3))
    loc = [(hb - c).dot(e) for e in frame(c, o, n)]
    ex, ey, ez = frame(X[C4], X[O4], X[N3])
    return X[C4] + loc[0] * ex + loc[1] * ey + loc[2] * ez


import os
if os.environ.get('TS_GUESS') == 'mapped':
    h_ts = mapped_ts_h()

guesses = {
    'reactant': with_h(X[H3]),
    'ts': with_h(h_ts),
    'product': with_h(toward(X[O4], h_ts, 0.97)),
}

if __name__ == '__main__':
    run_case('fivefu', guesses, ROOT / 'results',
             log=lambda s: print(s, flush=True))
