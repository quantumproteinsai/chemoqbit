"""Convergence of the ONIOM barrier with the size of the high-level region, at
the ONIOM(wB97X-D3:xTB)-optimised reactant, transition state and product.

E = E_xTB(cluster) + E_DFT(region) - E_xTB(region), mechanical embedding,
wB97X-D3(BJ)/def2-SVP, for regions grown residue by residue so that only link
atoms on the substrates' own cut bonds are needed:

  fragment-18         the 18-atom region used in the optimisation
  substrates          nucleotide ring, methylene and pterin in full
  substrates+contacts + Asn177, water 327, His147 and Cys146
  all                 the whole 96-atom cluster (DFT, gas phase, + xTB ALPB solvation)

    python scripts/ts_qmsize.py [model]    # model: default (o4h, his147p as ts_oniom_opt --variant)
Resumable: results/ts_qmsize[.model].json
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
XC, BASIS = 'wb97x-d3bj', 'def2-svp'
NAMES = ('reactant', 'ts', 'product')
REGIONS = {'substrates': {'nucleotide', 'methylene', 'folate'},
           'substrates+contacts': {'nucleotide', 'methylene', 'folate', 'asn177', 'wat327', 'his147', 'cys146'},
           'all': None}
# residue formal charges per model: Arg166 always; the variants' extra proton
FORMAL = {'default': {'arg166': 1}, 'his147p': {'arg166': 1, 'his147': 1},
          'o4h': {'arg166': 1, 'nucleotide': 1}}


def residues(sym, lab, X, k):
    import ts_cluster_nuc as S
    B = S._bonds(sym, X)
    res = [l.split(':')[0] if ':' in l else None for l in lab]
    for i in range(len(sym)):
        if res[i] is None:
            nb = [b if a == i else a for a, b in B if i in (a, b)]
            res[i] = res[nb[0]] if nb else None
    res[k['Hd']] = 'methylene'               # the transferring hydride: unbonded at the TS by the distance test
    return res, B


def main(model='default'):
    import ts_cluster_nuc as S
    from chemoqbit.enzyme.oniom import fragment
    from chemoqbit.enzyme.xtb import energy_grad
    from pyscf import gto, dft
    suf = '' if model == 'default' else f'.{model}'
    out = ROOT / 'results' / f'ts_qmsize{suf}.json'
    r = json.loads((ROOT / 'results' / f'ts_cluster_run{suf}.json').read_text())
    o = json.loads((ROOT / 'results' / f'ts_oniom_opt{suf}.json').read_text())
    p = r['prep']; sym, lab, k, q = p['symbols'], p['labels'], p['key'], p['charge']
    res, B = residues(sym, lab, np.array(o['ts']['X']), k)
    regions = {'fragment-18': S.active_region(r)}
    for name, rs in REGIONS.items():
        at = list(range(len(sym))) if rs is None else [i for i in range(len(sym)) if res[i] in rs]
        cuts = [(a, b) if a in at else (b, a) for a, b in B if (a in at) != (b in at) and k['Hd'] not in (a, b)]
        regions[name] = (at, cuts)
    # a region's charge = the formal charges of the residues it holds
    charge = {name: sum(c for rs_, c in FORMAL[model].items() if any(res[i] == rs_ for i in at))
              for name, (at, _) in regions.items()}
    assert charge['all'] == q, (charge, q)
    data = json.loads(out.read_text()) if out.exists() else {}
    for name, (at, cuts) in regions.items():
        for n in NAMES:
            key = f'{name}/{n}'
            if key in data:
                continue
            t0 = time.time()
            X = np.array(o[n]['X'])
            fs, fX = fragment(sym, X, at, cuts)
            mol = gto.M(atom=[(s, x) for s, x in zip(fs, fX)], basis=BASIS, charge=charge[name], unit='Angstrom', verbose=0)
            mf = dft.RKS(mol, xc=XC).density_fit() if len(fs) > 40 else dft.RKS(mol, xc=XC)
            mf.conv_tol = 1e-9
            e_hi = float(mf.kernel())
            e_cl = float(energy_grad(sym, X, q)[0]); e_lf = float(energy_grad(fs, fX, charge[name], solvent=None)[0])
            data[key] = dict(atoms=len(fs), charge=charge[name], E_high=e_hi, E_xtb_cluster=e_cl, E_xtb_region=e_lf,
                             E=e_cl + e_hi - e_lf, converged=bool(mf.converged), seconds=time.time() - t0)
            tmp = out.with_suffix('.json.tmp'); tmp.write_text(json.dumps(data, indent=1)); tmp.replace(out)
            print(f'{key}: {len(fs)} atoms, E={data[key]["E"]:.6f} ({time.time() - t0:.0f}s)', flush=True)
        E = {n: data[f'{name}/{n}']['E'] for n in NAMES}
        print(f"qmsize {name}: barrier {(E['ts'] - E['reactant']) * 627.509474:.2f}, "
              f"reaction {(E['product'] - E['reactant']) * 627.509474:+.2f} kcal/mol", flush=True)
    data['_done'] = True
    tmp = out.with_suffix('.json.tmp'); tmp.write_text(json.dumps(data, indent=1)); tmp.replace(out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'default')
