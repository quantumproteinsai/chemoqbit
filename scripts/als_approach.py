#!/usr/bin/env python
"""What does the fold charge to bring Cys6 into attack range of the SOD1 disulfide?

The 2013 mutational work established Cys6 as the initiator of disulfide scrambling
and explained it as "the proximity of C146 to C6". Our geometry says the two
sulfurs are 8.37 A apart at a 133 deg approach angle, and the straight line
between them runs through the backbone of Gly147 and Leu117 on packed neighbouring
strands of the beta-barrel. So the approach is not a side-chain rotation and
proximity is the wrong word. Nobody has the energy, and this computes it.

Two constraint regimes bracket the real protein:

    rigid   every backbone N, CA, C, O held at its crystallographic position:
            the fold cannot yield at all.             UPPER bound
    mobile  only the outer shell held, local backbone free. LOWER bound

Either bound can settle the question on its own. The disease allows about
31 kcal/mol for an event that must happen once in fifty years, so if even the
mobile lower bound clears that, scrambling cannot initiate in the folded protein
and something else has to happen first. If the mobile bound falls well below it,
the approach is affordable and the barrier at sulfur becomes the question.

Stages, each resumable:

    build     cluster, hydrogens, relax at the crystallographic separation
    mobile    restrained approach with the local backbone free
    rigid     the same with the whole backbone held
    dft       wB97X-D3(BJ)/def2-SVP single points at the endpoints of each

GFN2-xTB is used for the scan deliberately. The chalcogen ladder showed it is
unreliable for the hypervalent sulfur chemistry, but nothing here forms or breaks
an S-S bond: this is sterics and backbone strain, which is what a tight-binding
method is good at. The DFT stage checks that claim rather than assuming it.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chemoqbit import als_approach as A          # noqa: E402
from chemoqbit import sanity                     # noqa: E402
from chemoqbit.enzyme.frozen import optimise_frozen   # noqa: E402

HARTREE_KCAL = 627.5094740631
SOLVENT = 'water'
STEP = 0.35
TARGET = 3.2            # a reactive thiolate-disulfide encounter sits at 3.0-3.5 A
OUT = ROOT / 'results' / 'als_approach.json'


def save(d):
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(OUT)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


# The default maxiter of 3000 is what made the first attempt infeasible: one
# relaxation of this cluster ran for TEN HOURS without converging, which put the
# full two-regime scan at 439 hours. Each scan point is seeded from its neighbour,
# so it starts close and does not need thousands of steps; the cost was the cap
# being reached, not the number of degrees of freedom.
MAXITER = 300
GTOL = 1e-3


def _relax(sym, X, charge, frozen, i, j, d0, k=20.0):
    cons = [('dist', i, j, d0)]
    return optimise_frozen(sym, X, charge=charge, frozen=frozen, constraints=cons,
                           k=k, solvent=SOLVENT, maxiter=MAXITER, gtol=GTOL)


def stage_build(res):
    if 'build' in res:
        return
    t0 = time.time()
    sym, X, q, idx, lab, rigid, mobile = A.build()
    symH, XH, qH, idx, labH = A.protonate(sym, X, q, idx, lab)
    d0 = A.separation(XH, idx)

    # relax at the crystallographic separation so the reference is a relaxed
    # cluster and not the raw crystal geometry; otherwise every later point
    # inherits whatever strain the truncation introduced
    Xr, Er = _relax(symH, XH, qH, mobile, idx['S6'], idx['S146'], d0)

    res['build'] = dict(symbols=symH, X=Xr.tolist(), charge=qH, idx=idx, labels=labH,
                        frozen_rigid=rigid, frozen_mobile=mobile,
                        n_atoms=len(symH), d_crystal=d0,
                        d_relaxed=A.separation(Xr, idx), E_ref=Er,
                        blocking=A.blocking(Xr, labH, idx, within=3.0),
                        seconds=time.time() - t0)
    print(f'cluster: {len(symH)} atoms, charge {qH:+d}, S6...S146 '
          f'{d0:.2f} A crystal -> {A.separation(Xr, idx):.2f} A relaxed', flush=True)
    print('  on the axis: ' + ', '.join(
        f"{b['label']} {b['off_axis']:.2f}" for b in res['build']['blocking'][:5]), flush=True)


def _approach(res, regime):
    b = res['build']
    sym, charge, idx = b['symbols'], b['charge'], b['idx']
    frozen = b['frozen_rigid'] if regime == 'rigid' else b['frozen_mobile']
    X = np.array(b['X'])
    d_start = b['d_relaxed']

    grid = np.arange(d_start, TARGET - 0.5 * STEP, -STEP)
    pts = []
    for d0 in grid:
        X, E = _relax(sym, X, charge, frozen, idx['S6'], idx['S146'], d0)
        d = A.separation(X, idx)
        pts.append(dict(d_target=float(d0), d=d, E=E, X=X.tolist(),
                        blocking=A.blocking(X, b['labels'], idx, within=2.5)[:3]))
        print(f'  {regime} d {d0:.2f} -> {d:.2f} A   E {E:+.6f} Eh   '
              f'W {(E - pts[0]["E"]) * HARTREE_KCAL:+7.2f} kcal/mol', flush=True)
    return pts


def _record(res, regime, pts):
    W = [(p['E'] - pts[0]['E']) * HARTREE_KCAL for p in pts]
    res[regime] = dict(
        profile=[{k: v for k, v in p.items() if k != 'X'} for p in pts],
        geometries={f"{p['d']:.2f}": p['X'] for p in pts},
        work=W[-1], d_final=pts[-1]['d'], d_start=pts[0]['d'])
    sanity.record(res, 'approach_tracking', max(abs(p['d'] - p['d_target']) for p in pts),
                  label=f'{regime} restraint following')
    sanity.record(res, 'conformational_work', W[-1], label=f'{regime} S6 approach')
    print(f'{regime}: W = {W[-1]:+.2f} kcal/mol to reach {pts[-1]["d"]:.2f} A', flush=True)


def stage_mobile(res):
    if 'mobile' in res:
        return
    print('=== mobile: only the outer shell held, local backbone free', flush=True)
    _record(res, 'mobile', _approach(res, 'mobile'))


def stage_rigid(res):
    if 'rigid' in res:
        return
    print('=== rigid: the whole backbone held at crystallographic positions', flush=True)
    _record(res, 'rigid', _approach(res, 'rigid'))


def stage_report(res):
    lo = res.get('mobile', {}).get('work')
    hi = res.get('rigid', {}).get('work')
    if lo is None or hi is None:
        return
    res['report'] = dict(
        lower_bound=lo, upper_bound=hi,
        window_lo=25.19, window_hi=31.23,       # once a day / once in fifty years, 310 K
        verdict=('scrambling cannot initiate in the folded protein'
                 if lo > 31.23 else
                 'the approach is affordable; the barrier at sulfur becomes the question'
                 if lo < 25.19 else
                 'the approach lands inside the window the disease allows'))
    print(f"\nconformational work to bring Cys6 to {TARGET} A of Cys146:")
    print(f"  lower bound (local backbone free)  {lo:+8.2f} kcal/mol")
    print(f"  upper bound (fold held rigid)      {hi:+8.2f} kcal/mol")
    print(f"  the disease allows 25.2-31.2 kcal/mol for a once-in-decades event")
    print(f"  -> {res['report']['verdict']}", flush=True)
    save(res)


STAGES = dict(build=stage_build, mobile=stage_mobile, rigid=stage_rigid,
              report=stage_report)


def main(names):
    res = load()
    for n in names:
        print(f'=== {n}', flush=True)
        STAGES[n](res)
        res['_stages_done'] = sorted(set(res.get('_stages_done', [])) | {n})
        save(res)


if __name__ == '__main__':
    main(sys.argv[1:] or ['build', 'mobile', 'rigid', 'report'])
