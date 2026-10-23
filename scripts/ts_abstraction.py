"""C5 proton abstraction in thymidylate synthase: dUMP against FdUMP.

Why 5-fluorouracil traps the enzyme, as a calculation rather than an assumption.
In the covalent ternary complex the C5 proton is removed and the Cys146 thiolate
eliminated in a concerted step (Kholodar et al., Biochemistry 60, 1243 (2021)).
FdUMP carries fluorine at C5 and so has no proton to remove; the standard account
is that the complex is therefore trapped, which this script tests by computing
both routes for both substrates.

Stages, each resumable from results/ts_abstraction.json:

  prep      build the 103-atom cluster from PDB 1TLS (unedited covalent complex,
            plus Glu58) for c5 = H and F, relax with GFN2-xTB
  acceptor  dUMP: 1D proton-transfer scan to each candidate acceptor, C6-S free,
            to find which acceptor the proton actually reaches
  surface   dUMP: 2D scan over the proton coordinate and the C6-S distance, the
            concerted route, with the minimax path across the grid
  cleave    both: C6-S cleavage alone, with C5 keeping its substituent -- the
            only route open to FdUMP
  tsopt     P-RFO transition state and partial Hessian for the preferred route
  oniom     ONIOM(wB97X-D3(BJ)/def2-SVP:GFN2-xTB) energies at the stationary points

    python scripts/ts_abstraction.py [stage ...]      # default: all, in order
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from chemoqbit.enzyme import abstraction as A
from chemoqbit.enzyme.frozen import optimise_frozen
from chemoqbit.enzyme.xtb import energy_grad

OUT = ROOT / 'results' / 'ts_abstraction.json'
K = 627.509474
XI = np.round(np.arange(-1.4, 1.41, 0.2), 2)        # d(C5-H) - d(H-acceptor)
RCS = np.round(np.arange(1.95, 3.56, 0.2), 2)       # d(C6-S)
KRES = 20.0


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(d):
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(OUT)


def model(c5):
    """(symbols, X, labels, frozen indices, index of the C5 hydrogen or None)"""
    sym, X0, lab, fr = A.build(c5=c5)
    molH = A.add_hydrogens(sym, X0, lab)
    conf = molH.GetConformer()
    P = np.array([list(conf.GetAtomPosition(i)) for i in range(molH.GetNumAtoms())])
    S = [a.GetSymbol() for a in molH.GetAtoms()]
    return S, P, lab, [i for i, f in enumerate(fr) if f], A.c5_hydrogen(molH, lab)


def geom(X, lab, h):
    i = lambda k: lab.index(k)
    d = lambda a, b: float(np.linalg.norm(X[a] - X[b]))
    g = {'C6-SG': d(i('nucleotide:C6'), i('cys146:SG')),
         'C5-C11': d(i('nucleotide:C5'), i('methylene:C11')),
         'C11-N5': d(i('methylene:C11'), i('folate:N5')),
         'C5-C6': d(i('nucleotide:C5'), i('nucleotide:C6'))}
    if h is not None:
        g['C5-H'] = d(i('nucleotide:C5'), h)
        for name, key in A.ACCEPTORS.items():
            g[f'H-{name}'] = d(h, i(key))
    return g


def stage_prep(res):
    res.setdefault('prep', {})
    for c5 in ('H', 'F'):
        if c5 in res['prep']:
            continue
        S, P, lab, frozen, h = model(c5)
        t0 = time.time()
        X, E = optimise_frozen(S, P, charge=0, frozen=frozen)
        res['prep'][c5] = dict(symbols=S, labels=lab, frozen=frozen, hydrogen=h, X=X.tolist(),
                               E=float(E), geometry=geom(X, lab, h), seconds=time.time() - t0)
        save(res)
        print(f"prep {c5}: E {E:.6f}  " + '  '.join(f'{k} {v:.2f}' for k, v in res['prep'][c5]['geometry'].items()),
              flush=True)


def _relax(p, X, cons):
    return optimise_frozen(p['symbols'], X, charge=0, frozen=p['frozen'], constraints=cons, k=KRES)


def stage_acceptor(res):
    p = res['prep']['H']; lab, h = p['labels'], p['hydrogen']
    i5 = lab.index('nucleotide:C5')
    res.setdefault('acceptor', {})
    for name, key in A.ACCEPTORS.items():
        if name in res['acceptor']:
            continue
        ja = lab.index(key)
        X = np.array(p['X']); rows = []
        for xi in XI:
            X, E = _relax(p, X, [(i5, h, ja, h, float(xi))])
            rows.append(dict(xi=float(xi), E=float(E), rel=float((E - p['E']) * K), X=X.tolist(),
                             **geom(X, lab, h)))
            print(f"  acceptor {name} xi {xi:+.1f}  {(E - p['E']) * K:+8.2f} kcal/mol  "
                  f"C5-H {rows[-1]['C5-H']:.2f} C6-SG {rows[-1]['C6-SG']:.2f}", flush=True)
        res['acceptor'][name] = dict(atom=key, scan=rows,
                                     barrier=float(max(r['rel'] for r in rows)),
                                     final=rows[-1]['rel'])
        save(res)
    best = min(res['acceptor'], key=lambda k: res['acceptor'][k]['barrier'])
    res['best_acceptor'] = best
    save(res)
    print(f"best acceptor: {best} (barrier {res['acceptor'][best]['barrier']:.2f} kcal/mol)", flush=True)


def stage_surface(res):
    p = res['prep']['H']; lab, h = p['labels'], p['hydrogen']
    i5, i6 = lab.index('nucleotide:C5'), lab.index('nucleotide:C6')
    isg = lab.index('cys146:SG')
    ja = lab.index(A.ACCEPTORS[res['best_acceptor']])
    s = res.setdefault('surface', dict(xi=XI.tolist(), rcs=RCS.tolist(), acceptor=res['best_acceptor'], points={}))
    scan1d = {r['xi']: r['X'] for r in res['acceptor'][res['best_acceptor']]['scan']}
    for xi in XI:
        X = np.array(scan1d[float(xi)])        # each column starts from the 1D scan
        for rcs in RCS:
            kk = f'{xi}_{rcs}'
            if kk in s['points']:
                X = np.array(s['points'][kk]['X']); continue
            t0 = time.time()
            X, E = _relax(p, X, [(i5, h, ja, h, float(xi)), (i6, isg, float(rcs))])
            s['points'][kk] = dict(E=float(E), rel=float((E - p['E']) * K), X=X.tolist(),
                                   **geom(X, lab, h))
            save(res)
            print(f"  surface xi {xi:+.1f} rCS {rcs:.2f}  {(E - p['E']) * K:+8.2f} kcal/mol ({time.time() - t0:.0f}s)",
                  flush=True)
    save(res)


def stage_cleave(res):
    """C6-S cleavage with the C5 substituent kept: the only route open to FdUMP."""
    res.setdefault('cleave', {})
    for c5 in ('H', 'F'):
        if c5 in res['cleave']:
            continue
        p = res['prep'][c5]; lab, h = p['labels'], p['hydrogen']
        i6, isg = lab.index('nucleotide:C6'), lab.index('cys146:SG')
        X = np.array(p['X']); rows = []
        for rcs in RCS:
            X, E = _relax(p, X, [(i6, isg, float(rcs))])
            rows.append(dict(rcs=float(rcs), E=float(E), rel=float((E - p['E']) * K), **geom(X, lab, h)))
            print(f"  cleave {c5} rCS {rcs:.2f}  {(E - p['E']) * K:+8.2f} kcal/mol", flush=True)
        res['cleave'][c5] = dict(scan=rows, barrier=float(max(r['rel'] for r in rows)), final=rows[-1]['rel'])
        save(res)


STAGES = dict(prep=stage_prep, acceptor=stage_acceptor, surface=stage_surface, cleave=stage_cleave)


def main(names):
    res = load()
    for n in names:
        print(f'=== {n}', flush=True)
        STAGES[n](res)
    res['_stages_done'] = sorted(set(res.get('_stages_done', [])) | set(names))
    save(res)


if __name__ == '__main__':
    main(sys.argv[1:] or ['prep', 'acceptor', 'cleave', 'surface'])
