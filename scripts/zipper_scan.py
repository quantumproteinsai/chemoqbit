"""What holds an amyloid steric zipper together, residue by residue.

Two beta-sheets of tau's PHF6 motif (306-VQIVYK-311) mate through a dry
interface (PDB 2ON9, 1.51 A; Sawaya et al., Nature 2007). The two chains of the
asymmetric unit are the two sheets, touching only through Val1, Ile3 and Tyr5 --
one face of the strand.

  E_int  = E(both sheets) - E(sheet A) - E(sheet B), at one geometry
  ddE    = E_int(residue truncated to alanine in both sheets) - E_int(wild type)

Truncation is done in the relaxed geometry, so every other atom stays put and
the scan is a clean difference. Hydrogens are relaxed once with the heavy atoms
held at their crystal positions.

This segment is used instead of a full-length fibril because 1.51 A supports a
quantum-chemical argument about side chains and 3.4-4.0 A cryo-EM does not. The
price is that it answers what makes this sequence zip, not what a familial
mutation does.

    python scripts/zipper_scan.py [stage ...]      # wt, scan, dft
Resumable: results/zipper.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit.zipper import build, protonate
from chemoqbit.enzyme.frozen import optimise_frozen
from chemoqbit.enzyme.xtb import energy_grad

PDB = ROOT / 'data' / 'rcsb_2on9' / '2ON9.pdb'
OUT = ROOT / 'results' / 'zipper.json'
K = 627.509474
SOLVENT = 'water'
NEIGHBOURS = 0            # one strand per sheet; 1 adds a stacked strand either side
XC, BASIS = 'wb97x-d3bj', 'def2-svp'
RESIDUES = [1, 2, 3, 4, 5, 6]
NAMES = {1: 'Val306', 2: 'Gln307', 3: 'Ile308', 4: 'Val309', 5: 'Tyr310', 6: 'Lys311'}


def save(d):
    tmp = OUT.with_suffix('.json.tmp'); tmp.write_text(json.dumps(d, indent=1)); tmp.replace(OUT)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def pieces(to_alanine=()):
    c = build(str(PDB), neighbours=NEIGHBOURS, to_alanine=to_alanine)
    sym, X, lab, q, sheet = protonate(c)
    heavy = [i for i, s in enumerate(sym) if s != 'H']
    X, _ = optimise_frozen(sym, X, charge=q, frozen=heavy, solvent=SOLVENT,
                           maxiter=300, gtol=5e-4)
    return sym, np.array(X), q, sheet, lab


def interaction(sym, X, q, sheet):
    """E(both) - E(sheet 0) - E(sheet 1); each sheet carries half the charge."""
    E = lambda s, x, qq: float(energy_grad(s, x, qq, solvent=SOLVENT)[0])
    half = q // 2
    assert 2 * half == q, f'charge {q} does not split evenly between the sheets'
    e_all = E(sym, X, q)
    out = {}
    for s_ in (0, 1):
        i = np.flatnonzero(sheet == s_)
        out[s_] = E([sym[k] for k in i], X[i], half)
    return (e_all - out[0] - out[1]) * K, dict(E_both=e_all, E_sheet0=out[0], E_sheet1=out[1])


def stage_wt(res):
    if 'wt' in res:
        return
    t0 = time.time()
    sym, X, q, sheet, lab = pieces()
    ei, parts = interaction(sym, X, q, sheet)
    res['wt'] = dict(atoms=len(sym), charge=q, neighbours=NEIGHBOURS, E_int=ei,
                     seconds=time.time() - t0, **parts)
    save(res)
    print(f'wild type: {len(sym)} atoms, charge {q:+d}, E_int {ei:+.2f} kcal/mol '
          f'({time.time() - t0:.0f}s)', flush=True)


def stage_scan(res):
    res.setdefault('alanine', {})
    for n in RESIDUES:
        key = NAMES[n]
        if key in res['alanine']:
            continue
        t0 = time.time()
        try:
            sym, X, q, sheet, lab = pieces(to_alanine=[n])
            ei, parts = interaction(sym, X, q, sheet)
            res['alanine'][key] = dict(E_int=ei, ddE=ei - res['wt']['E_int'], charge=q,
                                       atoms=len(sym), seconds=time.time() - t0, **parts)
            print(f"  {key} -> Ala  E_int {ei:+8.2f}  ddE {ei - res['wt']['E_int']:+7.2f} kcal/mol "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            res['alanine'][key] = dict(failed=f'{type(e).__name__}: {e}')
            print(f'  {key} failed: {type(e).__name__}: {e}', flush=True)
        save(res)
    rows = sorted(((k, v['ddE']) for k, v in res['alanine'].items() if 'ddE' in v),
                  key=lambda r: r[1], reverse=True)
    print('side chains whose loss costs most: ' + ', '.join(f'{k} {v:+.1f}' for k, v in rows),
          flush=True)


def stage_dft(res):
    """A DFT check of the wild-type interface energy."""
    if 'dft' in res:
        return
    from pyscf import gto, dft
    sym, X, q, sheet, lab = pieces()
    out = {}
    half = q // 2
    for tag, idx, qq in (('both', np.arange(len(sym)), q),
                         ('sheet0', np.flatnonzero(sheet == 0), half),
                         ('sheet1', np.flatnonzero(sheet == 1), half)):
        t0 = time.time()
        mol = gto.M(atom=[(sym[i], X[i]) for i in idx], basis=BASIS, charge=int(qq),
                    unit='Angstrom', verbose=0)
        mf = dft.RKS(mol, xc=XC).density_fit()
        mf.conv_tol = 1e-8
        out[tag] = dict(E=float(mf.kernel()), converged=bool(mf.converged), atoms=len(idx),
                        seconds=time.time() - t0)
        res['dft'] = out; save(res)
        print(f"  dft {tag}: {len(idx)} atoms, E {out[tag]['E']:.6f} ({time.time() - t0:.0f}s)",
              flush=True)
    out['E_int'] = (out['both']['E'] - out['sheet0']['E'] - out['sheet1']['E']) * K
    res['dft'] = out; save(res)
    print(f"DFT E_int {out['E_int']:+.2f} kcal/mol (gas phase)", flush=True)


STAGES = dict(wt=stage_wt, scan=stage_scan, dft=stage_dft)


def main(names):
    res = load()
    for n in names:
        print(f'=== {n}', flush=True)
        STAGES[n](res)
        res['_stages_done'] = sorted(set(res.get('_stages_done', [])) | {n})
        save(res)


if __name__ == '__main__':
    main(sys.argv[1:] or ['wt', 'scan', 'dft'])
