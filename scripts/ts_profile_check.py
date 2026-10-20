"""Barrier along the xTB path with a functional that is reliable for barriers.

The functional check showed B3LYP underestimating the embedded-fragment barrier
by ~10 kcal/mol; the remaining question is the geometry. Every DFT/ONIOM energy
so far is a single point at the GFN2-xTB saddle point, which need not be where
a better method puts the top of the barrier. This evaluates, at the relaxed
reactant, TS and product and at scan points around the TS:

  frag_ee   wB97X-D3(BJ)/def2-TZVP on the 18-atom fragment in the xTB point charges
  oniom_me  E_xTB(cluster) + E_DFT(fragment) - E_xTB(fragment): ONIOM(DFT:xTB)
            with mechanical embedding, so the environment's strain and its
            interaction with the fragment enter at the xTB level without being
            counted twice

and reports where each profile peaks and how high, relative to the reactant.

    python scripts/ts_profile_check.py        # resumable: results/ts_profile_check.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
XC, BASIS = 'wb97x-d3bj', 'def2-tzvp'
OUT = ROOT / 'results' / 'ts_profile_check.json'
# dense around the TS (xi = +0.1), coarser towards the ends
XI = [-1.4, -1.2, -1.0, -0.8] + [round(-0.6 + 0.1 * i, 1) for i in range(15)] + [1.0, 1.2, 1.4]


def points(r):
    pts = [('reactant', None, r['reactant'])]
    by_xi = {p['xi']: p for p in r['scan']}
    pts += [(f'xi{x:+.1f}', x, by_xi[x]) for x in XI if x in by_xi]
    pts += [('ts', None, r['ts']), ('product', None, r['product'])]
    return pts


def main():
    import ts_cluster_nuc as S
    from chemoqbit.enzyme.xtb import energy_grad, charges
    from pyscf import gto, dft, qmmm
    K = S.HARTREE_KCAL
    r = json.loads((ROOT / 'results' / 'ts_cluster_run.json').read_text())
    sym = r['prep']['symbols']
    atoms, cuts = S.active_region(r)
    drop = set(atoms) | {b for _, b in cuts}
    env = [i for i in range(len(sym)) if i not in drop]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}

    for name, xi, p in points(r):
        if name in res and 'frag_ee' in res[name] and 'frag_gas' in res[name]:
            continue
        t0 = time.time()
        X = np.array(p['coords'])
        xyz = S._fragment(sym, X, atoms, cuts)
        fsym = [s for s, _ in xyz]; fX = np.array([x for _, x in xyz])
        atom = '\n'.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in xyz)
        q = charges(sym, X, r['prep']['charge'])
        e = res.setdefault(name, {})
        e.update(xi=xi, e_xtb_cluster=float(energy_grad(sym, X, r['prep']['charge'])[0]),
                 # gas phase, like the DFT fragment: solvation enters via the cluster term only
                 e_xtb_frag=float(energy_grad(fsym, fX, 0, solvent=None)[0]))
        for kind in ('frag_ee', 'frag_gas'):
            if kind in e:
                continue
            mol = gto.M(atom=atom, basis=BASIS, charge=0, unit='Angstrom', verbose=0)
            mf = dft.RKS(mol, xc=XC)
            if kind == 'frag_ee':
                mf = qmmm.mm_charge(mf, X[env], q[env], unit='Angstrom')
            mf.conv_tol = 1e-9
            e[kind] = float(mf.kernel())
            e[f'{kind}_converged'] = bool(mf.converged)
            OUT.write_text(json.dumps(res, indent=1))
        print(f"{name}: done ({time.time() - t0:.0f}s)", flush=True)

    # profiles relative to the relaxed reactant
    R = res['reactant']
    prof = {}
    for name, xi, _ in points(r):
        e = res[name]
        prof[name] = dict(
            xi=xi,
            xtb=(e['e_xtb_cluster'] - R['e_xtb_cluster']) * K,
            frag_ee=(e['frag_ee'] - R['frag_ee']) * K,
            oniom_me=((e['e_xtb_cluster'] + e['frag_gas'] - e['e_xtb_frag'])
                      - (R['e_xtb_cluster'] + R['frag_gas'] - R['e_xtb_frag'])) * K)
    res['_profile'] = prof
    OUT.write_text(json.dumps(res, indent=1))
    print(f'\n{"point":>9} {"xi":>6} {"xTB":>7} {"frag EE":>8} {"ONIOM-ME":>9}   (kcal/mol vs reactant)')
    for name, v in prof.items():
        xi = '' if v['xi'] is None else f"{v['xi']:+.1f}"
        print(f"{name:>9} {xi:>6} {v['xtb']:7.2f} {v['frag_ee']:8.2f} {v['oniom_me']:9.2f}")
    scan = {n: v for n, v in prof.items() if v['xi'] is not None}
    for k in ('xtb', 'frag_ee', 'oniom_me'):
        top = max(scan, key=lambda n: scan[n][k])
        print(f"{k}: scan maximum {scan[top][k]:.2f} at xi {scan[top]['xi']:+.1f}; at the xTB TS {prof['ts'][k]:.2f}")


if __name__ == '__main__':
    main()
