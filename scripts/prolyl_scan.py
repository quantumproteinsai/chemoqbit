"""cis/trans isomerisation of a prolyl amide, and what phosphorylation does to it.

The conformational switch behind cis-pThr231-Pro tau. For each capped dipeptide
Ac-Xaa-Pro-NHMe the amide torsion omega is scanned right round, so both senses
of rotation are covered, with GFN2-xTB (ALPB water); the stationary points are
then re-optimised and given DFT single points. The observable that matters is
the DIFFERENCE phosphorylated minus unphosphorylated, for the barrier and for
the cis/trans equilibrium; the absolute barrier is also meaningful here, since
the system is small enough for a reliable level of theory and experiment puts
prolyl barriers at 14-24 kcal/mol.

Stages, resumable from results/prolyl.json:

  scan    relaxed omega scan, every model, GFN2-xTB/ALPB water
  points  trans and cis minima and the two saddles, refined
  dft     wB97X-D3(BJ)/def2-TZVP with C-PCM(water) at those geometries

    python scripts/prolyl_scan.py [stage ...]
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit.prolyl import MODELS, build, torsion
from chemoqbit.enzyme.frozen import optimise_frozen

OUT = ROOT / 'results' / 'prolyl.json'
K = 627.509474
STEP = 15
OMEGA = list(range(-180, 180, STEP))         # the full circle: both rotation senses
SOLVENT = 'water'
KTOR = 0.05                                  # Eh/rad^2, soft enough not to distort the rest
XC, BASIS = 'wb97x-d3bj', 'def2-tzvp'


def save(d):
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(OUT)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


MAXITER, GTOL = 500, 5e-4      # the soft torsion restraint never reaches a tighter
                               # gradient, so cap the iterations instead of grinding


def relax(sym, X, q, cons, k=KTOR, maxiter=MAXITER):
    try:
        Xn, E = optimise_frozen(sym, X, charge=q, constraints=cons, k=k, solvent=SOLVENT,
                                maxiter=maxiter, gtol=GTOL)
        return np.array(Xn), float(E), True
    except Exception as e:
        print(f'    failed: {type(e).__name__}: {e}', flush=True)
        return np.array(X), None, False


def stage_scan(res):
    res.setdefault('scan', {})
    for name in MODELS:
        if name in res['scan']:
            continue
        sym, X0, om, q = build(name)
        rows = {}
        X = X0.copy()
        # start from trans and go round, so each point starts from its neighbour
        order = sorted(OMEGA, key=lambda w: abs(((w - 180) + 180) % 360 - 180))
        for w in order:
            t0 = time.time()
            Xn, E, ok = relax(sym, X, q, [('tor', *om, float(w))])
            if ok:
                X = Xn
            rows[str(w)] = dict(omega_target=w, omega=torsion(X, om) if ok else None,
                                E=E, X=X.tolist() if ok else None, seconds=time.time() - t0)
            print(f'  scan {name:6s} omega {w:+4d} -> {rows[str(w)]["omega"] if ok else float("nan"):+7.1f}  '
                  + ('failed' if E is None else f'E {E:.6f}') + f'  ({time.time() - t0:.0f}s)', flush=True)
        good = {int(k_): v for k_, v in rows.items() if v['E'] is not None}
        e0 = min(v['E'] for v in good.values())
        res['scan'][name] = dict(charge=q, omega_atoms=list(om), symbols=sym,
                                 points=rows, e_min=e0,
                                 profile={k_: (v['E'] - e0) * K for k_, v in sorted(good.items())})
        save(res)
        prof = res['scan'][name]['profile']
        tr = min((k_ for k_ in prof if abs(abs(k_) - 180) <= 30), key=lambda k_: prof[k_], default=None)
        ci = min((k_ for k_ in prof if abs(k_) <= 30), key=lambda k_: prof[k_], default=None)
        print(f'  {name}: trans {tr} ({prof.get(tr, float("nan")):.2f}), cis {ci} '
              f'({prof.get(ci, float("nan")):.2f}) kcal/mol relative to the lowest point', flush=True)


def _stationary(prof):
    """(trans, cis, saddle_pos, saddle_neg) omega keys from the scan profile"""
    ks = sorted(prof)
    tr = min((k for k in ks if abs(abs(k) - 180) <= 40), key=lambda k: prof[k], default=None)
    ci = min((k for k in ks if abs(k) <= 40), key=lambda k: prof[k], default=None)
    sp = max((k for k in ks if 20 < k < 160), key=lambda k: prof[k], default=None)
    sn = max((k for k in ks if -160 < k < -20), key=lambda k: prof[k], default=None)
    return tr, ci, sp, sn


def stage_points(res):
    res.setdefault('points', {})
    for name, d in res['scan'].items():
        if name in res['points']:
            continue
        prof = {int(k): v for k, v in d['profile'].items()}
        tr, ci, sp, sn = _stationary(prof)
        sym, q, om = d['symbols'], d['charge'], tuple(d['omega_atoms'])
        out = {}
        for tag, w in (('trans', tr), ('cis', ci), ('saddle+', sp), ('saddle-', sn)):
            if w is None:
                continue
            X = np.array(d['points'][str(w)]['X'])
            if tag in ('trans', 'cis'):           # free relaxation: a true minimum
                Xn, E, ok = relax(sym, X, q, [], k=0.0)
            else:                                 # hold the torsion at the saddle
                Xn, E, ok = relax(sym, X, q, [('tor', *om, float(w))], k=1.0)
            out[tag] = dict(omega_scan=w, omega=torsion(Xn, om) if ok else None, E=E,
                            X=Xn.tolist() if ok else None)
            print(f'  point {name:6s} {tag:8s} omega {out[tag]["omega"]}  '
                  + ('failed' if E is None else f'E {E:.6f}'), flush=True)
        res['points'][name] = out
        save(res)


def stage_dft(res):
    from pyscf import gto, dft
    res.setdefault('dft', {})
    for name, pts in res['points'].items():
        res['dft'].setdefault(name, {})
        sym, q = res['scan'][name]['symbols'], res['scan'][name]['charge']
        for tag, p in pts.items():
            if tag in res['dft'][name] or not p['X']:
                continue
            t0 = time.time()
            mol = gto.M(atom=[(s, x) for s, x in zip(sym, p['X'])], basis=BASIS, charge=q,
                        unit='Angstrom', verbose=0)
            mf = dft.RKS(mol, xc=XC).density_fit()
            mf = mf.PCM()
            mf.with_solvent.eps = 78.4
            mf.conv_tol = 1e-9
            E = float(mf.kernel())
            res['dft'][name][tag] = dict(E=E, converged=bool(mf.converged), seconds=time.time() - t0)
            save(res)
            print(f'  dft {name:6s} {tag:8s} E {E:.6f} ({time.time() - t0:.0f}s)', flush=True)
        d = res['dft'][name]
        if 'trans' in d and 'cis' in d:
            sad = [d[t]['E'] for t in ('saddle+', 'saddle-') if t in d]
            print(f"  {name}: DFT cis-trans {(d['cis']['E'] - d['trans']['E']) * K:+.2f}, "
                  f"barrier trans->cis {(min(sad) - d['trans']['E']) * K:.2f} kcal/mol" if sad else '', flush=True)


STAGES = dict(scan=stage_scan, points=stage_points, dft=stage_dft)


def main(names):
    res = load()
    for n in names:
        print(f'=== {n}', flush=True)
        STAGES[n](res)
        res['_stages_done'] = sorted(set(res.get('_stages_done', [])) | {n})
        save(res)


if __name__ == '__main__':
    main(sys.argv[1:] or ['scan', 'points', 'dft'])
