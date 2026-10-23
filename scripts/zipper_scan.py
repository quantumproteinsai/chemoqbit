"""What holds an amyloid steric zipper together, residue by residue.

Three segments, each a microcrystal structure good enough for a side-chain
argument -- unlike the cryo-EM fibrils of full-length Abeta or tau, which carry
the familial mutations but sit at 3.4-4.0 A:

  tau       tau PHF6, 306-VQIVYK-311 (2ON9, 1.51 A; Sawaya et al., Nature 2007)
  asyn47    alpha-synuclein 47-56, GVVHGVTTVA (4ZNN, 1.41 A; Rodriguez et al.,
            Nature 2015) -- the A53T FAMILIAL PARKINSON mutant as crystallised.
            Truncating Thr53 to alanine restores the wild-type sequence, so the
            pathogenic mutation is a difference between two models that differ
            by one side chain, at a geometry good enough to mean something.
  nacore    alpha-synuclein 68-78, GAVVTGVTAVA (4RIL, 1.43 A), the toxic core

The mating sheet is the partner that buries the most area with the largest
side-chain share (chemoqbit.zipper.mating_sheet), which is how Sawaya et al.
define a zipper: facing side chains interdigitating in a dry interface.
Counting close contacts, as a first attempt did, scores an edge-on packing
contact just as highly and picked the wrong partner.

  E_int  = E(both sheets) - E(sheet A) - E(sheet B), at one geometry
  ddE    = E_int(residue truncated to alanine in both sheets) - E_int(wild type)

Truncation happens in the relaxed geometry, so every other atom stays put.
Hydrogens are relaxed once with the heavy atoms at their crystal positions.

    python scripts/zipper_scan.py <system> [stage ...]     # wt, scan, dft
Resumable: results/zipper_<system>.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit.zipper import build, protonate
from chemoqbit.enzyme.frozen import optimise_frozen
from chemoqbit import sanity
from chemoqbit.enzyme.xtb import energy_grad

K = 627.509474
SOLVENT = 'water'
NEIGHBOURS = 0            # one strand per sheet; 1 adds a stacked strand either side
XC, BASIS = 'wb97x-d3bj', 'def2-svp'
SYSTEMS = {
    'tau':    dict(pdb='rcsb_2on9/2ON9.pdb', mate='B', note='tau PHF6 306-VQIVYK-311'),
    'asyn47': dict(pdb='rcsb_4znn/4ZNN.pdb', mate=None,
                   note='alpha-synuclein 47-56, the A53T familial Parkinson mutant; '
                        'truncating Thr53 to Ala restores the wild-type sequence'),
    'nacore': dict(pdb='rcsb_4ril/4RIL.pdb', mate=None, note='alpha-synuclein NACore 68-78'),
}
S = {}
PDB = None
OUT = None


def save(d):
    tmp = OUT.with_suffix('.json.tmp'); tmp.write_text(json.dumps(d, indent=1)); tmp.replace(OUT)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def pieces(to_alanine=()):
    c = build(str(PDB), mate=S['mate'], neighbours=NEIGHBOURS, to_alanine=to_alanine)
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


def residues(lab):
    """(number, label) of each residue of the first sheet, in order."""
    out = {}
    for l in lab:
        strand, res, _nm = l.split(':')
        n = int(''.join(c for c in res if c.isdigit()))
        out[n] = res
    return sorted(out.items())


def stage_wt(res):
    if 'wt' in res:
        return
    t0 = time.time()
    from chemoqbit.zipper import mating_sheet
    if S['mate'] is None:
        _R, _t, desc = mating_sheet(str(PDB))
        res['interface'] = desc
        print(f"  interface: symmetry op {desc['op']} cell {desc['cell']}, buries "
              f"{desc['buried']:.1f} A^2, side-chain fraction {desc['side_fraction']:.2f}, "
              f"interfacial waters {desc['interfacial_waters']}", flush=True)
    sym, X, q, sheet, lab = pieces()
    ei, parts = interaction(sym, X, q, sheet)
    res['wt'] = dict(atoms=len(sym), charge=q, neighbours=NEIGHBOURS, E_int=ei,
                     residues=[f'{r}{n}' for n, r in residues(lab)],
                     seconds=time.time() - t0, **parts)
    res['geometry'] = dict(labels=lab)
    save(res)
    sanity.record(res, 'interaction_energy', ei, label='zipper wild type', fatal=True)
    print(f'wild type: {len(sym)} atoms, charge {q:+d}, E_int {ei:+.2f} kcal/mol '
          f'({time.time() - t0:.0f}s)', flush=True)


def stage_scan(res):
    res.setdefault('alanine', {})
    nums = [n for n, r in residues(res['geometry']['labels'])]
    names = dict(residues(res['geometry']['labels']))
    for n in nums:
        key = f'{names[n]}{n}'
        if key in res['alanine'] or names[n] in ('GLY', 'ALA'):
            if names[n] in ('GLY', 'ALA'):
                res['alanine'].setdefault(key, dict(skipped='no side chain beyond CB'))
            continue
        t0 = time.time()
        try:
            sym, X, q, sheet, lab = pieces(to_alanine=[n])
            ei, parts = interaction(sym, X, q, sheet)
            res['alanine'][key] = dict(E_int=ei, ddE=ei - res['wt']['E_int'], charge=q,
                                       atoms=len(sym), seconds=time.time() - t0, **parts)
            sanity.record(res, 'alanine_ddE', ei - res['wt']['E_int'], label=f'{key} -> Ala')
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


def main(system, names):
    global S, PDB, OUT
    S = SYSTEMS[system]
    PDB = ROOT / 'data' / S['pdb']
    OUT = ROOT / 'results' / f'zipper_{system}.json'
    print(f"system {system}: {S['note']}", flush=True)
    res = load()
    for n in names:
        print(f'=== {n}', flush=True)
        STAGES[n](res)
        res['_stages_done'] = sorted(set(res.get('_stages_done', [])) | {n})
        save(res)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:] or ['wt', 'scan', 'dft'])
