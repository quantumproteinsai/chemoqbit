"""Misfolding and its rescue on the diamond lattice -> results/fold_misfold.json

For each peptide, exhaustively over all self-avoiding conformations:
  1. the wild-type native state (ground-state contact map) and the reference
     temperature T_ref at which the wild type is 80 % native;
  2. every single mutant: dG_fold, ddG = dG(mutant) - dG(wild type), and
     whether a non-native fold becomes the ground state (misfolding);
  3. for the most destabilising mutants, every second-site mutation, and the
     suppressors that bring the native population back to the wild type's;
  4. chaperone ligands that bind 1, 2 or 3 residues of a native pocket, at
     activity x = 1 and binding energy eps per contact: native population vs
     eps for the wild type and those mutants, and the eps needed for rescue.

    python scripts/fold_misfold.py
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit.fold import misfold as M
from chemoqbit.fold.lattice import load_mj

# (label, sequence, residue number of the first residue, familial disease mutations)
PEPTIDES = [('chignolin', 'GYDPETGTWG', 1, []),
            ('tau 300-310', 'VPGGGSVQIVY', 300, ['P301L', 'P301S']),
            ('Abeta 21-30', 'AEDVGSNKGA', 21, ['E22G', 'E22Q', 'E22K', 'D23N'])]
P_REF = 0.5            # wild-type native population defining T_ref (the midpoint)
N_WORST = 8            # destabilised mutants carried into the rescue screens
EPS = np.round(np.arange(0, 6.01, 0.25), 2)
OUT = ROOT / 'results' / 'fold_misfold.json'
AA = load_mj()[0]


def name(seq, p, a, first=1):
    return f'{seq[p]}{p + first}{a}'


def study(label, seq, first, disease):
    t0 = time.time()
    ens = M.Ensemble(len(seq))
    key, E0 = M.native_key(ens, seq)
    assert key is not None, 'wild-type ground states have different contact maps'
    nat = ens.native_mask(key)
    T = brentq(lambda t: M.population(E0, nat, t)[0] - P_REF, 0.05, 5.0)
    pn0, g0 = M.population(E0, nat, T)
    out = dict(sequence=seq, N=len(seq), qubits=ens.nq, conformations=int(len(ens.index)),
               native_contacts=[list(c) for c in key], native_conformations=int(nat.sum()),
               E_native=float(E0[nat][0]), T_ref=T, P_native=pn0, dG=g0)

    # single mutants
    singles = []
    for p in range(len(seq)):
        for a in AA:
            if a == seq[p]:
                continue
            E = ens.energies(M.mutate(seq, p, a))
            pn, g = M.population(E, nat, T)
            ok, _ = M.ground_is_native(E, nat)
            gs = int(np.argmin(E))
            singles.append(dict(mutant=name(seq, p, a, first), pos=p, aa=a, P_native=pn, ddG=g - g0, ground_native=ok,
                                misfold_contacts=None if ok else [list(c) for c in ens.contact_key(gs)],
                                gap=float(E[~nat].min() - E[nat].min())))
    singles.sort(key=lambda r: -r['ddG'])
    out['singles'] = singles
    out['n_misfolding'] = sum(not r['ground_native'] for r in singles)
    worst = singles[:N_WORST]

    # second-site suppressors
    sup = {}
    for w in worst:
        s1 = M.mutate(seq, w['pos'], w['aa'])
        rows = []
        for p in range(len(seq)):
            if p == w['pos']:
                continue
            for a in AA:
                if a == s1[p]:
                    continue
                E = ens.energies(M.mutate(s1, p, a))
                pn, g = M.population(E, nat, T)
                rows.append(dict(second=name(s1, p, a, first), P_native=pn, ddG=g - g0,
                                 ground_native=M.ground_is_native(E, nat)[0]))
        rows.sort(key=lambda r: r['ddG'])
        rescued = [r for r in rows if r['P_native'] >= pn0]
        sup[w['mutant']] = dict(tested=len(rows), rescued=len(rescued), best=rows[:5],
                                rescued_positions=sorted({int(r['second'][1:-1]) for r in rescued}))
    out['suppressors'] = sup
    rank = {r['mutant']: i + 1 for i, r in enumerate(singles)}
    out['disease'] = [dict(mutant=m, rank=rank[m], of=len(singles),
                           **{k: next(r for r in singles if r['mutant'] == m)[k] for k in ('ddG', 'P_native', 'ground_native')})
                      for m in disease]

    # chaperone ligands
    lig = {}
    for beads in (1, 2):
        pk, n = M.best_pocket(ens, nat, beads)
        if pk is None:
            continue
        size = len(pk)
        entry = dict(pocket=[int(i) for i in pk], pocket_residues=[f'{seq[i]}{i + first}' for i in pk],
                     binding_conformations=int(np.sum(n == size)), selectivity=float(nat.sum() / np.sum(n == size)),
                     curves={})
        for tag, s in [('wild type', seq)] + [(w['mutant'], M.mutate(seq, w['pos'], w['aa'])) for w in worst]:
            E = ens.energies(s)
            pns = [M.population(E, nat, T, M.ligand_shift(n, e, 1.0, T))[0] for e in EPS]
            resc = next((float(e) for e, v in zip(EPS, pns) if v >= pn0), None)
            entry['curves'][tag] = dict(P_native=pns, eps_rescue=resc)
        lig[str(beads)] = entry
    # deletions
    ens_d = M.Ensemble(len(seq) - 1)
    dels = []
    for p in range(len(seq)):
        m = M.keeps_fold(ens_d, key, p)
        row = dict(deletion=f'd{seq[p]}{p + first}', pos=p,
                   surviving_contacts=[list(c) for c in M.surviving_contacts(key, p)])
        if m is None:
            row.update(realisable=False)
        else:
            Ed = ens_d.energies(M.delete(seq, p))
            pn, g = M.population(Ed, m, T)
            row.update(realisable=True, conformations=int(m.sum()), P_fold=pn, dG=g,
                       ddG=g - g0, ground_keeps_fold=M.ground_is_native(Ed, m)[0])
        dels.append(row)
    out['deletions'] = dels

    out['ligand'] = dict(eps=EPS.tolist(), activity=1.0, by_beads=lig)
    out['seconds'] = time.time() - t0
    return out


def main():
    res = {}
    for label, seq, first, disease in PEPTIDES:
        r = res[label] = study(label, seq, first, disease)
        print(f"{label}: T_ref {r['T_ref']:.3f}, {r['n_misfolding']}/{len(r['singles'])} singles misfold; "
              f"worst {[s['mutant'] for s in r['singles'][:N_WORST]]}")
        for m, s in r['suppressors'].items():
            print(f"  {m}: {s['rescued']}/{s['tested']} suppressors, best {s['best'][0]['second']} "
                  f"(ddG {s['best'][0]['ddG']:+.2f})")
        nr = [d['deletion'] for d in r['deletions'] if not d['realisable']]
        print(f"  deletions: surviving native contacts unrealisable for {len(nr)}/{len(r['deletions'])} ({', '.join(nr)})")
        for d in r['disease']:
            print(f"  disease {d['mutant']}: ddG {d['ddG']:+.2f}, rank {d['rank']}/{d['of']}")
        for size, l in r['ligand']['by_beads'].items():
            print(f"  ligand {size}-bead {l['pocket_residues']}: binds {l['binding_conformations']} conf.; rescue eps "
                  + ', '.join(f"{k}:{v['eps_rescue']}" for k, v in l['curves'].items()))
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
