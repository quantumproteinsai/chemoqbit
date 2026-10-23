"""What holds a drug in its pocket, residue by residue -- for two diseases.

  cftr   lumacaftor in CFTR's first transmembrane domain (cystic fibrosis).
         The drug stabilises TMD1 and rescues dF508 allosterically; mutating
         pocket residues made dF508-CFTR insensitive to it (Fiedorczuk & Chen,
         Cell 2022, doi 10.1016/j.cell.2021.12.009; PDB 7SVD, 2.70 A).
  ache   donepezil in human acetylcholinesterase (Alzheimer's disease), an
         approved drug whose pocket is the aromatic gorge (Cheung et al.,
         J. Med. Chem. 2012, 55, 10282, doi 10.1021/jm300871x; PDB 4EY7, 2.35 A).

The computed observable is a difference throughout:

  E_int  = E(complex) - E(pocket) - E(ligand), at one geometry, so internal
           energies cancel and only the contact survives
  ddE    = E_int(residue truncated to alanine) - E_int(wild type), the
           contribution of each side chain

Heavy atoms stay at their deposited positions and only hydrogens are relaxed:
at 2.3-2.7 A the side-chain positions are not accurate enough to re-optimise
freely, and an alanine scan wants a common geometry anyway.

    python scripts/pocket_scan.py <system> [stage ...]     # stages: wt, scan, dft
Resumable: results/pocket_<system>.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit.pocket import cut, protonate, mutate_to_alanine
from chemoqbit import sanity

from chemoqbit.enzyme.frozen import optimise_frozen
from chemoqbit.enzyme.xtb import energy_grad

K = 627.509474
XC, BASIS = 'wb97x-d3bj', 'def2-svp'
SYSTEMS = {
    # ALPB 'ether' stands in for a lipid-facing site, 'water' for a solvent-filled gorge
    'cftr': dict(pdb='rcsb_7svd/7SVD.pdb', lig='VX8', chain='A', cutoff=4.5, solvent='ether',
                 lig_charge=0, sites=None),
    'ache': dict(pdb='rcsb_4ey7/4EY7.pdb', lig='E20', chain='A', cutoff=4.5, solvent='water',
                 lig_charge=1, sites={'lig:N14': 1}),   # donepezil's piperidinium
}
S = {}          # the chosen system, filled by main()
OUT = None      # results/pocket_<system>.json, set by main()


def save(d):
    tmp = OUT.with_suffix('.json.tmp'); tmp.write_text(json.dumps(d, indent=1)); tmp.replace(OUT)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def pieces(to_alanine=()):
    """(symbols, X, charge, ligand mask, cut, hydrogens-of-each-heavy-atom) with
    hydrogens added and relaxed."""
    c = cut(str(ROOT / 'data' / S['pdb']), S['lig'], chain=S['chain'], cutoff=S['cutoff'],
            to_alanine=to_alanine)
    sym, X, lab, q, mol = protonate(c, ligand_charge=S['lig_charge'], sites=S['sites'])
    # which atoms belong to the ligand: the first n_ligand heavy atoms and their hydrogens
    lig = np.zeros(len(sym), bool)
    lig[:c['n_ligand']] = True
    for a in mol.GetAtoms():
        if a.GetSymbol() == 'H':
            nb = a.GetNeighbors()
            if nb and lig[nb[0].GetIdx()]:
                lig[a.GetIdx()] = True
    hofs = {i: [] for i, s_ in enumerate(sym) if s_ != 'H'}
    for a in mol.GetAtoms():
        if a.GetSymbol() == 'H':
            nb = a.GetNeighbors()
            if nb and nb[0].GetIdx() in hofs:
                hofs[nb[0].GetIdx()].append(a.GetIdx())
    heavy = list(hofs)
    X, _ = optimise_frozen(sym, X, charge=q, frozen=heavy, solvent=S['solvent'],
                           maxiter=300, gtol=5e-4)
    return sym, np.array(X), q, lig, c, hofs


def interaction(sym, X, q, lig):
    """E(complex) - E(pocket) - E(ligand), all at the same geometry (kcal/mol)."""
    q_lig = S['lig_charge']
    E = lambda s, x, qq: float(energy_grad(s, x, qq, solvent=S['solvent'])[0])
    e_all = E(sym, X, q)
    li = np.flatnonzero(lig); pi = np.flatnonzero(~lig)
    e_lig = E([sym[i] for i in li], X[li], q_lig)
    e_poc = E([sym[i] for i in pi], X[pi], q - q_lig)
    return (e_all - e_lig - e_poc) * K, dict(E_complex=e_all, E_ligand=e_lig, E_pocket=e_poc)


def stage_wt(res):
    if 'wt' in res:
        return
    t0 = time.time()
    sym, X, q, lig, c, hofs = pieces()
    ei, parts = interaction(sym, X, q, lig)
    res['wt'] = dict(residues=c['residues'], atoms=len(sym), charge=q,
                     n_ligand_atoms=int(lig.sum()), E_int=ei, seconds=time.time() - t0, **parts)
    res['geometry'] = dict(symbols=sym, X=X.tolist(), lig=lig.tolist(), labels=c['labels'],
                           hofs={str(k): v for k, v in hofs.items()})
    save(res)
    sanity.record(res, 'interaction_energy', ei, label='wild type', fatal=True)
    print(f"wild type: {len(sym)} atoms, charge {q:+d}, {len(c['residues'])} residues, "
          f"E_int {ei:+.2f} kcal/mol ({time.time() - t0:.0f}s)", flush=True)
    print('  residues: ' + ', '.join(c['residues']), flush=True)


def stage_scan(res):
    res.setdefault('alanine', {})
    g = res['geometry']
    sym, X = g['symbols'], np.array(g['X'])
    lig, lab = np.array(g['lig']), g['labels']
    hofs = {int(k): v for k, v in g['hofs'].items()}
    q = res['wt']['charge']
    for resid in res['wt']['residues']:
        if resid in res['alanine']:
            continue
        if resid.startswith(('GLY', 'ALA')):
            res['alanine'][resid] = dict(skipped='no side chain beyond CB')
            save(res); continue
        t0 = time.time()
        try:
            s2, X2, lig2, dq = mutate_to_alanine(sym, X, lab, lig, hofs, resid)
            ei, parts = interaction(s2, X2, q - dq, lig2)
            res['alanine'][resid] = dict(E_int=ei, ddE=ei - res['wt']['E_int'], charge=q - dq,
                                         atoms=len(s2), seconds=time.time() - t0, **parts)
            sanity.record(res, 'alanine_ddE', ei - res['wt']['E_int'], label=f'{resid} -> Ala')
            print(f"  {resid:8s} -> Ala  E_int {ei:+8.2f}  ddE {ei - res['wt']['E_int']:+7.2f} kcal/mol "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            res['alanine'][resid] = dict(failed=f'{type(e).__name__}: {e}')
            print(f'  {resid:8s} failed: {type(e).__name__}: {e}', flush=True)
        save(res)
    rows = [(k, v['ddE']) for k, v in res['alanine'].items() if 'ddE' in v]
    rows.sort(key=lambda r: r[1], reverse=True)
    print('side chains whose loss costs most: ' + ', '.join(f'{k} {v:+.1f}' for k, v in rows[:6]),
          flush=True)


def stage_dft(res):
    """One DFT check of the wild-type interaction energy."""
    if 'dft' in res:
        return
    from pyscf import gto, dft
    g = res['geometry']; sym = g['symbols']; X = np.array(g['X']); lig = np.array(g['lig'])
    q = res['wt']['charge']
    out = {}
    for tag, idx, qq in (('complex', np.arange(len(sym)), q),
                         ('ligand', np.flatnonzero(lig), S['lig_charge']),
                         ('pocket', np.flatnonzero(~lig), q - S['lig_charge'])):
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


def main(system, names):
    global S, OUT
    S = SYSTEMS[system]
    OUT = ROOT / 'results' / f'pocket_{system}.json'
    res = load()
    for n in names:
        print(f'=== {n}', flush=True)
        STAGES[n](res)
        res['_stages_done'] = sorted(set(res.get('_stages_done', [])) | {n})
        save(res)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:] or ['wt', 'scan', 'dft'])
