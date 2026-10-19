"""Functional sensitivity of the hydride-transfer barrier on the embedded
ONIOM fragment (same 18-atom region and xTB point charges as stage_oniom).

Question it answers: is B3LYP's fragment barrier (16.5 kcal/mol at def2-SVP)
low compared with CCSD(T) (28.3) because B3LYP underestimates this barrier, or
are the wavefunction energies at the xTB geometry the outlier? Functionals that
are reliable for barrier heights landing near CCSD(T) means the former.

    python scripts/ts_functional_check.py [basis]      # default def2-tzvp
Writes results/ts_functional_check.json (resumable).
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import ts_cluster_nuc as S
from pyscf import gto, dft, qmmm

BASIS = sys.argv[1] if len(sys.argv) > 1 else 'def2-tzvp'
# exact exchange fraction in brackets; the last three are recommended for barriers
XC = ['b3lyp-d3bj',      # 20 %
      'pbe0-d3bj',       # 25 %
      'm062x-d3zero',    # 54 %, meta-GGA, fitted to barrier heights among others
      'wb97x-d3bj',      # range-separated, 16-100 %
      'bhandhlyp']       # 50 %
OUT = ROOT / 'results' / 'ts_functional_check.json'
K = S.HARTREE_KCAL
NAMES = ('reactant', 'ts', 'product')

r = json.loads((ROOT / 'results' / 'ts_cluster_run.json').read_text())
sym = r['prep']['symbols']
atoms, cuts = S.active_region(r)
frag = {}
for n in NAMES:
    xyz = S._fragment(sym, np.array(r[n]['coords']), atoms, cuts)
    Xe, qe, _ = S._environment(r, n, atoms, cuts)
    frag[n] = ('\n'.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in xyz), Xe, qe)

res = json.loads(OUT.read_text()) if OUT.exists() else {}
for xc in XC:
    tag = f'{xc}/{BASIS}'
    res.setdefault(tag, {})
    for n in NAMES:
        if n in res[tag]:
            continue
        t0 = time.time()
        atom, Xe, qe = frag[n]
        mol = gto.M(atom=atom, basis=BASIS, charge=0, unit='Angstrom', verbose=0)
        try:
            mf = qmmm.mm_charge(dft.RKS(mol, xc=xc), Xe, qe, unit='Angstrom')
            mf.conv_tol = 1e-9
            e = float(mf.kernel())
            res[tag][n] = dict(E=e, converged=bool(mf.converged), seconds=time.time() - t0)
        except Exception as ex:                       # e.g. no dispersion parameters for this xc
            res[tag][n] = dict(error=f'{type(ex).__name__}: {ex}')
        OUT.write_text(json.dumps(res, indent=1))
        print(f'{tag} {n}: {res[tag][n]}', flush=True)
    E = {n: res[tag][n].get('E') for n in NAMES}
    if all(v is not None for v in E.values()):
        print(f"{tag}: barrier {(E['ts'] - E['reactant']) * K:.2f}, "
              f"reaction {(E['product'] - E['reactant']) * K:+.2f} kcal/mol", flush=True)
