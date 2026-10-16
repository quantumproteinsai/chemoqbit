"""Composite energies: CCSD(T) + quantum static-correlation correction.

    E = E_CCSD(T)[full]  +  ( E_VQE[active]  -  E_CCSD(T)[active] )

Both bracketed terms use the SAME active-space Hamiltonian as the circuit
(rebuilt deterministically with ``build_problem``), so the bracket measures
only what the circuit captures beyond CCSD(T) inside the active space: the
static / multireference correlation. Checkpointed per geometry.

usage: python scripts/composite.py formamide fivefu
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pyscf import gto, scf, cc, ao2mo
from chemoqbit.chem import build_problem


def ccsd_t_in_space(p):
    n = p.h1.shape[0]
    mol = gto.M(verbose=0)
    mol.nelectron = p.nelec
    mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *a: p.h1
    mf.get_ovlp = lambda *a: np.eye(n)
    mf.energy_nuc = lambda *a: p.ecore
    mf._eri = ao2mo.restore(8, p.h2, n)
    mf.kernel(dm0=None)
    mycc = cc.CCSD(mf).run(conv_tol=1e-10)
    et = mycc.ccsd_t() if p.nelec > 2 else 0.0
    return float(mycc.e_tot + et), float(mf.e_tot)


def main(tags):
    for tag in tags:
        f = ROOT / 'results' / f'{tag}.json'
        out_f = ROOT / 'results' / f'{tag}_composite.json'
        data = json.loads(f.read_text())
        comp = json.loads(out_f.read_text()) if out_f.exists() else {}
        for g in ('reactant', 'ts', 'product'):
            atom = '; '.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in data[g]['atoms'])
            for sp in ('4e4o', '6e6o'):
                key = f'{g}/{sp}'
                if key in comp:
                    print(f'{tag} {key}: checkpointed', flush=True)
                    continue
                t0 = time.time()
                ne, no = int(sp[0]), int(sp[2])
                p = build_problem(atom, 'cc-pvdz', no, ne)
                stored = data[g]['spaces'][f'{sp}_adapt']['e_casci']
                assert abs(p.e_casci - stored) < 1e-7, \
                    f'rebuilt active space differs from the VQE run ({p.e_casci} vs {stored})'
                e_cct, e_hf_act = ccsd_t_in_space(p)
                comp[key] = dict(e_ccsd_t_active=e_cct, e_hf_active=e_hf_act, e_casci=p.e_casci)
                out_f.write_text(json.dumps(comp, indent=1))
                print(f'{tag} {key}: CASCI={p.e_casci:.8f} CCSD(T)act={e_cct:.8f} '
                      f'static={(p.e_casci - e_cct) * 1e3:+.3f} mHa ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
