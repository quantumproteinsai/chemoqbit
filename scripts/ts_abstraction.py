"""C5 proton abstraction in thymidylate synthase: dUMP against FdUMP.

Why 5-fluorouracil traps the enzyme, as a calculation rather than an assumption.
In the covalent ternary complex the C5 proton is removed and the Cys146 thiolate
eliminated in a concerted step (Kholodar et al., Biochemistry 60, 1243 (2021)).
FdUMP carries fluorine at C5 and so has no proton to remove; the standard account
is that the complex is therefore trapped, which this script tests by computing
both routes for both substrates.

No candidate base is within reach of the proton in any crystal structure. In
1TLS the nearest acceptors to C5 are Tyr94 OH (4.14 A) and water 387 (4.55 A),
and in the 2021 intermediate-analog structures 7JX1/7JXF (superposed onto 1TLS
here to 0.3 A RMSD over 26 CA) the nearest water overlays water 387, at 4.07 A.
The base must therefore approach before the proton can move, which is the
objection Ghosh et al. (Phys. Chem. Chem. Phys. 17, 30867 (2015)) raise against
any single residue acting as the general base. The coordinates are accordingly
the approach r = d(C5...A) and the transfer xi = d(C5-H) - d(H...A).

Stages, each resumable from results/ts_abstraction.json:

  prep      build the 103-atom cluster from PDB 1TLS (unedited covalent complex,
            plus Glu58) for c5 = H and F, relax with GFN2-xTB
  acceptor  dUMP: how cheaply each candidate base can approach C5
  cleave    both: C6-S cleavage alone, with C5 keeping its substituent -- the
            only route open to FdUMP
  band      dUMP: 2D band over the approach and the proton transfer

    python scripts/ts_abstraction.py [stage ...]      # default: all, in order
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from chemoqbit.enzyme import abstraction as A
from chemoqbit.enzyme.frozen import optimise_frozen

OUT = ROOT / 'results' / 'ts_abstraction.json'
K = 627.509474
APPROACH = np.round(np.arange(2.5, 4.81, 0.2), 2)   # d(C5...acceptor)
XIR = np.round(np.arange(-2.6, 0.61, 0.2), 2)       # d(C5-H) - d(H...acceptor)
RCS = np.round(np.arange(1.95, 3.56, 0.2), 2)       # d(C6-S)
KRES = 20.0
# the band keeps the proton bonded to C5 or to the acceptor: for a collinear
# C5-H...A, D_LO <= d(C5-H) <= D_HI. Outside it a restrained xi merely tears the
# proton off into the cavity, which is what an absolute xi range did.
D_LO, D_HI = 0.95, 1.70


def band(r):
    """the xi values reachable at approach r without tearing the proton off"""
    return [float(x) for x in XIR if 2 * D_LO - r <= x <= 2 * D_HI - r]


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
            g[f'C5-{name}'] = d(i('nucleotide:C5'), i(key))
    return g


def _relax(p, X, cons):
    return optimise_frozen(p['symbols'], X, charge=0, frozen=p['frozen'], constraints=cons, k=KRES)


def _relax_safe(p, X, cons):
    """(X, E, ok): an xTB SCF failure marks the point instead of ending the run."""
    try:
        Xn, E = _relax(p, X, cons)
        return Xn, float(E), True
    except Exception as e:
        print(f'    point failed: {type(e).__name__}: {e}', flush=True)
        return np.array(X), None, False


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
        print(f"prep {c5}: E {E:.6f}  "
              + '  '.join(f'{k} {v:.2f}' for k, v in res['prep'][c5]['geometry'].items()), flush=True)


def stage_acceptor(res):
    """How cheaply can each candidate base approach C5? The proton keeps its
    natural position; only d(C5...A) is restrained."""
    p = res['prep']['H']; lab, h = p['labels'], p['hydrogen']
    i5 = lab.index('nucleotide:C5')
    res.setdefault('acceptor', {})
    for name, key in A.ACCEPTORS.items():
        if name in res['acceptor']:
            continue
        ja = lab.index(key)
        X = np.array(p['X']); r0 = float(np.linalg.norm(X[i5] - X[ja]))
        rows = []
        for r in [v for v in APPROACH if v <= r0 + 1e-9][::-1]:
            X, E, ok = _relax_safe(p, X, [(i5, ja, float(r))])
            rows.append(dict(r=float(r), E=E, rel=None if E is None else float((E - p['E']) * K),
                             converged=ok, X=X.tolist() if ok else None, **geom(X, lab, h)))
            print(f"  approach {name} r {r:.2f}  "
                  + ('failed' if E is None else f"{(E - p['E']) * K:+8.2f} kcal/mol")
                  + f"  C5-H {rows[-1]['C5-H']:.2f} C6-SG {rows[-1]['C6-SG']:.2f}", flush=True)
        good = [x for x in rows if x['rel'] is not None]
        res['acceptor'][name] = dict(
            atom=key, r_reactant=r0, scan=rows,
            cost_close=(min(good, key=lambda x: x['r'])['rel'] if good else None),
            r_close=(min(good, key=lambda x: x['r'])['r'] if good else None),
            cost_min=min((x['rel'] for x in good), default=None))
        save(res)
    # rank on the stored scans, not on a field that an older run may have left
    # empty: the approach grid holds no 2.6 A point, so an exact lookup found none
    ranked = {}
    for k, v in res['acceptor'].items():
        good = [x for x in v['scan'] if x.get('rel') is not None]
        if good:
            c = min(good, key=lambda x: x['r'])
            ranked[k] = (c['rel'], c['r'])
    assert ranked, 'no acceptor produced a usable approach scan'
    best = min(ranked, key=lambda k: ranked[k][0])
    res['best_acceptor'] = best
    save(res)
    print('closest approach reached and its cost: '
          + ', '.join(f'{k} {r:.2f} A {c:+.2f}' for k, (c, r) in ranked.items())
          + f'  -> best {best}', flush=True)


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
            X, E, ok = _relax_safe(p, X, [(i6, isg, float(rcs))])
            rows.append(dict(rcs=float(rcs), E=E, rel=None if E is None else float((E - p['E']) * K),
                             converged=ok, **geom(X, lab, h)))
            print(f"  cleave {c5} rCS {rcs:.2f}  "
                  + ('failed' if E is None else f"{(E - p['E']) * K:+8.2f} kcal/mol"), flush=True)
        good = [x for x in rows if x['rel'] is not None]
        res['cleave'][c5] = dict(scan=rows, barrier=max((x['rel'] for x in good), default=None),
                                 final=good[-1]['rel'] if good else None)
        save(res)


def stage_band(res):
    """2D band over the approach and the proton transfer, for the best acceptor."""
    p = res['prep']['H']; lab, h = p['labels'], p['hydrogen']
    i5 = lab.index('nucleotide:C5')
    name = res['best_acceptor']
    ja = lab.index(A.ACCEPTORS[name])
    appr = {x['r']: x['X'] for x in res['acceptor'][name]['scan'] if x['X']}
    s = res.setdefault('band', dict(acceptor=name, xi=XIR.tolist(), points={}))
    for r in sorted(appr, reverse=True):
        X = np.array(appr[r])                      # each row starts from the approach scan
        for xi in band(r):
            kk = f'{r}_{xi}'
            if kk in s['points']:
                if s['points'][kk].get('X'):
                    X = np.array(s['points'][kk]['X'])
                continue
            t0 = time.time()
            X, E, ok = _relax_safe(p, X, [(i5, ja, float(r)), (i5, h, ja, h, float(xi))])
            s['points'][kk] = dict(E=E, rel=None if E is None else float((E - p['E']) * K),
                                   converged=ok, X=X.tolist() if ok else None, **geom(X, lab, h))
            save(res)
            print(f"  band r {r:.2f} xi {xi:+.1f}  "
                  + ('failed' if E is None else f"{(E - p['E']) * K:+8.2f} kcal/mol")
                  + f"  C5-H {s['points'][kk]['C5-H']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    good = [v['rel'] for v in s['points'].values() if v['rel'] is not None]
    s['min_barrier_on_band'] = min(good) if good else None
    save(res)


STAGES = dict(prep=stage_prep, acceptor=stage_acceptor, cleave=stage_cleave, band=stage_band)


def main(names):
    res = load()
    for n in names:
        print(f'=== {n}', flush=True)
        STAGES[n](res)
        res['_stages_done'] = sorted(set(res.get('_stages_done', [])) | {n})
        save(res)


if __name__ == '__main__':
    main(sys.argv[1:] or ['prep', 'acceptor', 'cleave', 'band'])
