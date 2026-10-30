#!/usr/bin/env python
"""Refine the prolyl saddle guesses, because the 15 degree grid is not close enough.

Every saddle search in the `points` stage failed except thr, and results/prolyl.json
records `points` as done with an empty dict for all seven models -- a stage that ran,
produced nothing, and was then marked complete.

The cause is not mode selection and not the curvature threshold. prolyl_scan.py already
records why the torsion guide was abandoned (it overlaps 0.72 with a null-space residual
and 0.12 with the true mode, because rotating the proline is mostly a global rotation
the projector removes) and why -0.02 Eh/A^2 was the wrong bar (a heavy collective mode
shows only -3e-4 at a genuine transition state). Both of those are right.

What is left is the guess. The lowest mode AT the 15 degree scan maximum:

    thr    -1.55e-03   converged, 61i cm-1, one imaginary mode
    pser1  -8.10e-04   did not converge
    ser    -2.79e-04   did not converge
    ala    -6.08e-05   followed mode turned POSITIVE at +7.31e-05
    pthr1  -3.65e-16   no negative mode to follow at all

Only thr starts with curvature the optimiser can hold. On a flat barrier top a 15 degree
grid point can sit far enough off the ridge that the transition mode is not yet negative.
So: sweep finely around the maximum, take the highest point of that, and start P-RFO
there. The threshold, the projector and the mode choice are left exactly as they are --
if a saddle still cannot be found from a 2 degree guess, the honest conclusion is that
the model has no transition state at this level of theory, not that the search needs
further loosening.

    python scripts/prolyl_refine.py [model ...]        resumable
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chemoqbit.prolyl import MODELS, build, torsion
from chemoqbit.enzyme.frozen import optimise_frozen
from chemoqbit.enzyme import tsopt as TS
from chemoqbit.enzyme.xtb import energy_grad
from chemoqbit import sanity

RES = ROOT / 'results'
OUT = RES / 'prolyl_refine.json'
SRC = RES / 'prolyl.json'
SOLVENT = 'water'
KTOR, MAXITER, GTOL = 3.0, 500, 5e-4
WINDOW, STEP = 15, 3          # one full coarse step either side, so the peak is always
                              # bracketed. At +/-12 centred on the coarse maximum the
                              # pthr1 window sat on the EDGE of a flat top -- its 105 and
                              # 120 points differ by 0.05 kcal -- and scanned monotonically
                              # downhill for 10 kcal, which is why it found nothing.
HARTREE = 627.509474


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {'models': {}}


def save(d):
    tmp = OUT.with_suffix('.tmp')
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(OUT)                      # atomic, so an interrupted run resumes clean


def relax(sym, X, q, cons):
    try:
        Xn, E = optimise_frozen(sym, X, charge=q, constraints=cons, k=KTOR,
                                solvent=SOLVENT, maxiter=MAXITER, gtol=GTOL)
        return np.array(Xn), float(E), True
    except Exception as e:
        print(f'    relax failed: {type(e).__name__}: {e}', flush=True)
        return np.array(X), None, False


def saddle_from_scan(scan):
    """The omega key of the highest point in the positive-rotation branch."""
    pts = {float(k): v for k, v in scan.get('points', {}).items()
           if isinstance(v, dict) and v.get('E') is not None and v.get('X')}
    pos = {k: v for k, v in pts.items() if 20 < k < 160}
    neg = {k: v for k, v in pts.items() if -160 < k < -20}
    branch = pos or neg
    if not branch:
        return None, None
    k = max(branch, key=lambda k: branch[k]['E'])
    return k, branch[k]


def main(names):
    src = json.loads(SRC.read_text())
    out = load()
    for name in names:
        if name in out['models'] and out['models'][name].get('done'):
            print(f'{name}: already refined, skipping', flush=True)
            continue
        scan = src['scan'].get(name)
        if not scan:
            print(f'{name}: no scan in prolyl.json', flush=True); continue
        w0, row = saddle_from_scan(scan)
        if w0 is None:
            print(f'{name}: no saddle branch in the scan', flush=True); continue

        sym, _X0, om, q = build(name)        # (symbols, X, omega_atoms, charge)

        rec = out['models'].setdefault(name, {'coarse_omega': w0, 'fine': {}})
        X = np.array(row['X'])
        print(f'\n{name}: coarse maximum at omega {w0:+.0f}, refining '
              f'{w0-WINDOW:+.0f}..{w0+WINDOW:+.0f} in {STEP} degree steps', flush=True)

        best = (float(row['E']), float(w0), X)
        # outward in each direction, each point starting from the previous one, so the
        # geometry is carried rather than re-guessed
        order = ([int(w0) + d for d in range(STEP, WINDOW + 1, STEP)]
                 + [int(w0) - d for d in range(STEP, WINDOW + 1, STEP)])
        Xprev = {int(w0): X}
        for w in order:
            if str(w) in rec['fine'] and rec['fine'][str(w)].get('X'):
                Xprev[w] = np.array(rec['fine'][str(w)]['X']); continue
            step = STEP if w > w0 else -STEP
            start = Xprev.get(w - step, X)
            t0 = time.time()
            Xn, E, ok = relax(sym, start, q, [('tor', *om, float(w))])
            if ok:
                Xprev[w] = Xn
            rec['fine'][str(w)] = dict(omega_target=w,
                                       omega=(torsion(Xn, om) if ok else None),
                                       E=E, X=(Xn.tolist() if ok else None),
                                       seconds=round(time.time() - t0, 1))
            if ok and E is not None and E > best[0]:
                best = (E, torsion(Xn, om), Xn)
            print(f'  omega {w:+4d} -> ' + ('failed' if E is None else
                  f'{torsion(Xn, om):+7.1f}  E {E:.6f}  '
                  f'{(E - float(row["E"])) * HARTREE:+6.2f} kcal vs coarse')
                  + f'  ({time.time() - t0:.0f}s)', flush=True)
            save(out)

        rec['refined_omega'] = best[1]
        rec['gain_kcal'] = round((best[0] - float(row['E'])) * HARTREE, 3)
        print(f'  refined maximum omega {best[1]:+.1f}, '
              f'{rec["gain_kcal"]:+.2f} kcal above the coarse point', flush=True)

        # the whole point: is the transition mode negative at the refined geometry?
        gfn = lambda Y: energy_grad(sym, Y, q, solvent=SOLVENT)[1]
        fn = lambda Y: (lambda r: (float(r[0]), r[1]))(energy_grad(sym, Y, q, solvent=SOLVENT))
        Xb = np.array(best[2])
        H0 = TS.fd_hessian(Xb, gfn)
        Pr = TS._cart_projector(Xb)
        bb, VV = np.linalg.eigh(Pr @ H0 @ Pr)
        rec['lowest_mode_refined'] = float(bb.min())
        coarse_mode = rec.get('lowest_mode_coarse')
        print(f'  lowest mode at the refined maximum {bb.min():+.2e} Eh/A^2'
              + (f' (coarse {coarse_mode:+.2e})' if coarse_mode else ''), flush=True)

        try:
            guide = VV[:, int(np.argmin(bb))].reshape(-1, 3)
            t0 = time.time()
            Xn, E, info = TS.prfo_follow(Xb, fn, guide, grad_fn=gfn, trust=0.05,
                                         gtol=3e-4, maxiter=60, neg=1e-5,
                                         log=lambda m: None)
            masses = [TS.AMU[s_] for s_ in sym]
            nu = TS.frequencies(np.array(Xn), TS.fd_hessian(np.array(Xn), gfn), masses)
            nim = int((nu < -50).sum())
            rec['saddle'] = dict(E=float(E), omega=torsion(np.array(Xn), om),
                                 n_imaginary=nim, nu_lowest=float(nu.min()),
                                 X=np.array(Xn).tolist(), seconds=round(time.time() - t0, 1))
            print(f'  SADDLE  omega {rec["saddle"]["omega"]:+.1f}  E {E:.6f}  '
                  f'imaginary {nim}' + (f' ({abs(nu.min()):.0f}i cm-1)' if nim else ''),
                  flush=True)
        except Exception as e:
            rec['saddle'] = dict(failed=f'{type(e).__name__}: {e}')
            print(f'  SADDLE  failed: {type(e).__name__}: {e}', flush=True)

        rec['done'] = True
        save(out)
    print(f'\nwritten: {OUT}')


if __name__ == '__main__':
    sel = sys.argv[1:] or [m for m in MODELS]
    main(sel)
