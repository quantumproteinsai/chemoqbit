"""What holds lumacaftor in its CFTR pocket, residue by residue.

Lumacaftor and tezacaftor bind a hydrophobic pocket in CFTR's first
transmembrane domain and stabilise it, rescuing dF508 allosterically; mutating
pocket residues made dF508-CFTR insensitive to both drugs (Fiedorczuk & Chen,
Cell 2022, doi 10.1016/j.cell.2021.12.009; PDB 7SVD, 2.70 A).

The computed observable is a difference throughout:

  E_int  = E(complex) - E(pocket) - E(ligand), at one geometry, so internal
           energies cancel and only the contact survives
  ddE    = E_int(residue truncated to alanine) - E_int(wild type), the
           contribution of each side chain

Heavy atoms stay at their deposited positions and only hydrogens are relaxed:
at 2.70 A the side-chain positions are not accurate enough to re-optimise
freely, and an alanine scan wants a common geometry anyway.

    python scripts/cftr_pocket.py [stage ...]      # wt, scan, dft
Resumable: results/cftr_pocket.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit.pocket import cut, protonate
from chemoqbit.enzyme.frozen import optimise_frozen
from chemoqbit.enzyme.xtb import energy_grad

PDB = ROOT / 'data' / 'rcsb_7svd' / '7SVD.pdb'
OUT = ROOT / 'results' / 'cftr_pocket.json'
LIG, CUTOFF, SOLVENT = 'VX8', 4.5, 'ether'     # ALPB: a membrane-like low-polarity medium
K = 627.509474
LIG_CHARGE = 0                                  # lumacaftor's benzoic acid drawn neutral
XC, BASIS = 'wb97x-d3bj', 'def2-svp'


def save(d):
    tmp = OUT.with_suffix('.json.tmp'); tmp.write_text(json.dumps(d, indent=1)); tmp.replace(OUT)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def pieces(to_alanine=()):
    """(symbols, X, charge, ligand mask) with hydrogens added and relaxed."""
    c = cut(str(PDB), LIG, cutoff=CUTOFF, to_alanine=to_alanine)
    sym, X, lab, q, mol = protonate(c, ligand_charge=LIG_CHARGE)
    # which atoms belong to the ligand: the first n_ligand heavy atoms and their hydrogens
    lig = np.zeros(len(sym), bool)
    lig[:c['n_ligand']] = True
    for a in mol.GetAtoms():
        if a.GetSymbol() == 'H':
            nb = a.GetNeighbors()
            if nb and lig[nb[0].GetIdx()]:
                lig[a.GetIdx()] = True
    heavy = [i for i, s in enumerate(sym) if s != 'H']
    X, _ = optimise_frozen(sym, X, charge=q, frozen=heavy, solvent=SOLVENT, maxiter=600, gtol=5e-4)
    return sym, np.array(X), q, lig, c


def interaction(sym, X, q, lig, q_lig=LIG_CHARGE):
    """E(complex) - E(pocket) - E(ligand), all at the same geometry (kcal/mol)."""
    E = lambda s, x, qq: float(energy_grad(s, x, qq, solvent=SOLVENT)[0])
    e_all = E(sym, X, q)
    li = np.flatnonzero(lig); pi = np.flatnonzero(~lig)
    e_lig = E([sym[i] for i in li], X[li], q_lig)
    e_poc = E([sym[i] for i in pi], X[pi], q - q_lig)
    return (e_all - e_lig - e_poc) * K, dict(E_complex=e_all, E_ligand=e_lig, E_pocket=e_poc)


def stage_wt(res):
    if 'wt' in res:
        return
    t0 = time.time()
    sym, X, q, lig, c = pieces()
    ei, parts = interaction(sym, X, q, lig)
    res['wt'] = dict(residues=c['residues'], atoms=len(sym), charge=q,
                     n_ligand_atoms=int(lig.sum()), E_int=ei, seconds=time.time() - t0, **parts)
    res['geometry'] = dict(symbols=sym, X=X.tolist(), lig=lig.tolist())
    save(res)
    print(f"wild type: {len(sym)} atoms, charge {q:+d}, {len(c['residues'])} residues, "
          f"E_int {ei:+.2f} kcal/mol ({time.time() - t0:.0f}s)", flush=True)
    print('  residues: ' + ', '.join(c['residues']), flush=True)


def stage_scan(res):
    res.setdefault('alanine', {})
    nums = [int(''.join(ch for ch in r if ch.isdigit())) for r in res['wt']['residues']]
    names = dict(zip(nums, res['wt']['residues']))
    for n in nums:
        if names[n] in res['alanine']:
            continue
        if names[n].startswith(('GLY', 'ALA')):
            res['alanine'][names[n]] = dict(skipped='no side chain beyond CB')
            save(res); continue
        t0 = time.time()
        try:
            sym, X, q, lig, c = pieces(to_alanine=[n])
            ei, parts = interaction(sym, X, q, lig)
            res['alanine'][names[n]] = dict(E_int=ei, ddE=ei - res['wt']['E_int'], charge=q,
                                            atoms=len(sym), seconds=time.time() - t0, **parts)
            print(f"  {names[n]:8s} -> Ala  E_int {ei:+8.2f}  ddE {ei - res['wt']['E_int']:+7.2f} kcal/mol "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            res['alanine'][names[n]] = dict(failed=f'{type(e).__name__}: {e}')
            print(f'  {names[n]:8s} failed: {e}', flush=True)
        save(res)
    rows = [(k, v['ddE']) for k, v in res['alanine'].items() if 'ddE' in v]
    rows.sort(key=lambda r: r[1], reverse=True)
    print('most important side chains (largest loss on truncation): '
          + ', '.join(f'{k} {v:+.1f}' for k, v in rows[:6]), flush=True)


def stage_dft(res):
    """One DFT check of the wild-type interaction energy."""
    if 'dft' in res:
        return
    from pyscf import gto, dft
    g = res['geometry']; sym = g['symbols']; X = np.array(g['X']); lig = np.array(g['lig'])
    q = res['wt']['charge']
    out = {}
    for tag, idx, qq in (('complex', np.arange(len(sym)), q),
                         ('ligand', np.flatnonzero(lig), LIG_CHARGE),
                         ('pocket', np.flatnonzero(~lig), q - LIG_CHARGE)):
        t0 = time.time()
        mol = gto.M(atom=[(sym[i], X[i]) for i in idx], basis=BASIS, charge=int(qq),
                    unit='Angstrom', verbose=0)
        mf = dft.RKS(mol, xc=XC).density_fit()
        mf.conv_tol = 1e-8
        out[tag] = dict(E=float(mf.kernel()), converged=bool(mf.converged), atoms=len(idx),
                        seconds=time.time() - t0)
        res['dft'] = out; save(res)
        print(f"  dft {tag}: {len(idx)} atoms, E {out[tag]['E']:.6f} ({time.time() - t0:.0f}s)", flush=True)
    ei = (out['complex']['E'] - out['ligand']['E'] - out['pocket']['E']) * K
    out['E_int'] = ei
    res['dft'] = out; save(res)
    print(f'DFT E_int {ei:+.2f} kcal/mol (gas phase, so not comparable to the ALPB value directly)',
          flush=True)


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
