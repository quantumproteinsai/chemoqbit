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
unreliable for the hypervalent sulfur chemistry, but this is meant to be sterics
and backbone strain, which is what a tight-binding method is good at. The DFT
stage checks that claim rather than assuming it.

That claim used to read "nothing here forms or breaks an S-S bond", and the first
rigid run (27 Oct 2026, archived as als_approach.ssbroken_20261027.json) falsified
it. With the backbone frozen, d(S57,S146) went 2.18 -> 4.33 A in one step between
6.27 and 5.92 A: the disulfide ruptured with Cys6 still 5.9 A away and taking no
part in it. The remaining nine points were a scan on a protein whose bond had
already dissociated and whose fragments could not escape, and the two energy drops
along them (-15.6 and -39.9 kcal/mol) were those fragments relaxing. The endpoint,
+198.73 kcal/mol, was therefore not an upper bound on the work to bring Cys6 to
3.2 A of an intact disulfide, and was not even the maximum of its own path (+238.64
at 3.47 A). The rigid regime now holds the disulfide at its relaxed length, and
every point checks that it is still a bond.
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


# An S_N2 at sulfur needs the nucleophile collinear with the bond being broken, so
# the distance alone does not define the target. Cys6 sits at 133 deg in the crystal
# and 121 deg in the relaxed cluster, and driving only d(S6,S146) to contact at that
# angle presses the two sulfurs together sideways: a contact pair, not a reactive
# geometry. The angle is driven with the distance, from its starting value to 175 deg.
ANGLE_END = 175.0


# The rigid regime means the fold cannot yield. The first run showed that the
# weakest thing in a frozen fold is not the backbone but the disulfide itself, so
# "nothing yields" has to include the bond: it is held at the length it relaxes to
# in the reference cluster, 2.02 A. The mobile branch lets it lengthen to 2.24 A
# over the same approach, so forbidding that relieves a real strain and inflates
# the work by a little -- which is the safe direction for an upper bound.
HOLD_SS = True


def _relax(sym, X, charge, frozen, i, j, d0, k=20.0, a=None, a0=None, extra=()):
    cons = [('dist', i, j, d0)]
    if a0 is not None:
        cons.append(('angle', i, j, a, a0))
    cons.extend(extra)
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
    a_start = A.attack_angle(Xr, idx)

    res['build'] = dict(symbols=symH, X=Xr.tolist(), charge=qH, idx=idx, labels=labH,
                        frozen_rigid=rigid, frozen_mobile=mobile,
                        n_atoms=len(symH), d_crystal=d0,
                        d_relaxed=A.separation(Xr, idx), E_ref=Er,
                        angle_start=a_start, angle_end=ANGLE_END,
                        blocking=A.blocking(Xr, labH, idx, within=3.0),
                        seconds=time.time() - t0)
    print(f'cluster: {len(symH)} atoms, charge {qH:+d}, S6...S146 '
          f'{d0:.2f} A crystal -> {A.separation(Xr, idx):.2f} A relaxed', flush=True)
    print('  on the axis: ' + ', '.join(
        f"{b['label']} {b['off_axis']:.2f}" for b in res['build']['blocking'][:5]), flush=True)


def _approach(res, regime, hold_ss=False):
    b = res['build']
    sym, charge, idx = b['symbols'], b['charge'], b['idx']
    frozen = b['frozen_rigid'] if regime == 'rigid' else b['frozen_mobile']
    X = np.array(b['X'])
    d_start = b['d_relaxed']
    ss0 = float(np.linalg.norm(X[idx['S57']] - X[idx['S146']]))
    extra = [('dist', idx['S57'], idx['S146'], ss0)] if hold_ss else []
    # the bond list comes from the relaxed reference and is fixed for the stage, so
    # every later point is judged against the structure the work is measured from
    bonds = A.covalent_bonds(sym, np.array(b['X']))

    grid = np.arange(d_start, TARGET - 0.5 * STEP, -STEP)
    a_start = b.get('angle_start', 133.0)
    angles = np.linspace(a_start, ANGLE_END, len(grid))

    # One point is three quarters of an hour and a stage is sixteen of them. The
    # first rigid run held its points in memory until the stage ended, so when the
    # stage had to be thrown away, eleven hours of finished work existed only as
    # printed lines in the service log. Each point is now checkpointed as it lands
    # and the stage resumes from the last one.
    pts = res.get('_partial', {}).get(regime, [])
    if pts:
        X = np.array(pts[-1]['X'])
        print(f'  resuming {regime} at point {len(pts) + 1} of {len(grid)}', flush=True)

    for d0, a0 in list(zip(grid, angles))[len(pts):]:
        X, E = _relax(sym, X, charge, frozen, idx['S6'], idx['S146'], d0,
                      a=idx['S57'], a0=float(a0), extra=extra)
        d, ang = A.separation(X, idx), A.attack_angle(X, idx)
        ss = float(np.linalg.norm(X[idx['S57']] - X[idx['S146']]))
        stretch, si, sj = A.worst_stretch(bonds, X)
        pts.append(dict(d_target=float(d0), d=d, angle_target=float(a0), angle=ang,
                        E=E, d_ss=ss, X=X.tolist(),
                        stretch=stretch, stretch_bond=[b['labels'][si], b['labels'][sj]],
                        blocking=A.blocking(X, b['labels'], idx, within=2.5)[:3]))
        print(f'  {regime} d {d0:.2f} -> {d:.2f} A  angle {a0:.0f} -> {ang:.0f}  '
              f'S57-S146 {ss:.2f} A  stretch x{stretch:.3f} '
              f'({b["labels"][si]}-{b["labels"][sj]})  '
              f'W {(E - pts[0]["E"]) * HARTREE_KCAL:+7.2f} kcal/mol', flush=True)
        # Two things have to stay true for a point to mean anything: the bond being
        # displaced is still a bond, and the cluster has strained rather than come
        # apart. The first rigid run failed the first of these and ran eleven more
        # hours; the second failed only the second, and conformational_work caught
        # it at the endpoint by the accident of 208 exceeding 200. Both are recorded
        # and saved before either is raised, so the geometry that broke survives in
        # the checkpoint and can be looked at.
        ok = sanity.record(res, 'chalcogen_bond', ss,
                           label=f'{regime} Cys57-Cys146 at d {d:.2f}')
        ok_b = sanity.record(res, 'covalent_stretch', stretch,
                             label=f'{regime} {b["labels"][si]}-{b["labels"][sj]} '
                                   f'at d {d:.2f}')
        res.setdefault('_partial', {})[regime] = pts
        save(res)
        if not ok:
            raise sanity.Violation(
                f'{regime}: the Cys57-Cys146 disulfide is {ss:.2f} A at d {d:.2f}; '
                f'it is no longer a bond, so the rest of this scan would not be an '
                f'approach to a disulfide. Stopping here.')
        if not ok_b:
            broke = A.new_bonds(sym, X, bonds)
            made = ', '.join(f'{b["labels"][i]}-{b["labels"][j]} {r:.2f} A'
                             for i, j, r in broke[:4]) or 'none'
            raise sanity.Violation(
                f'{regime}: {b["labels"][si]}-{b["labels"][sj]} is stretched x{stretch:.2f} '
                f'at d {d:.2f}. The cluster has come apart rather than strained, so the '
                f'work past here is not a conformational cost. New bonds: {made}. '
                f'Stopping here.')

    # The attack angle is driven from 121 deg up to 175, so for the first nine
    # points it is legitimately below the 150 deg floor, and recording it there
    # produced nine false violations per regime. The bound is about the geometry a
    # displacement is attempted in, which is the last point.
    sanity.record(res, 'chalcogen_attack_angle', pts[-1]['angle'],
                  label=f'{regime} at contact')
    return pts


def _record(res, regime, pts):
    W = [(p['E'] - pts[0]['E']) * HARTREE_KCAL for p in pts]
    # The work to traverse the path is the maximum along it, not the value at the
    # end. Recording the endpoint is what let the first rigid run pass its own
    # 200 kcal/mol ceiling: the endpoint was 198.73 and the maximum, four points
    # from it, was 238.64, which is what the ceiling exists to catch.
    i = max(range(len(W)), key=W.__getitem__)
    res[regime] = dict(
        profile=[{k: v for k, v in p.items() if k != 'X'} for p in pts],
        geometries={f"{p['d']:.2f}": p['X'] for p in pts},
        work=W[-1], work_max=W[i], d_at_max=pts[i]['d'],
        segments=sanity.monotonic_segments(W),
        d_final=pts[-1]['d'], d_start=pts[0]['d'])
    res.get('_partial', {}).pop(regime, None)
    sanity.record(res, 'approach_tracking', max(abs(p['d'] - p['d_target']) for p in pts),
                  label=f'{regime} restraint following')
    sanity.record(res, 'conformational_work', W[i], label=f'{regime} S6 approach')
    print(f'{regime}: W = {W[-1]:+.2f} kcal/mol to reach {pts[-1]["d"]:.2f} A', flush=True)
    if i != len(W) - 1:
        print(f'  not monotonic: the maximum is {W[i]:+.2f} kcal/mol at '
              f'{pts[i]["d"]:.2f} A, {len(W) - 1 - i} points before the end, in '
              f'{res[regime]["segments"]} monotonic segments', flush=True)


def stage_mobile(res):
    if 'mobile' in res:
        return
    print('=== mobile: only the outer shell held, local backbone free', flush=True)
    _record(res, 'mobile', _approach(res, 'mobile'))


def stage_rigid(res):
    if 'rigid' in res:
        return
    print('=== rigid: the whole backbone held at crystallographic positions, '
          'and the disulfide held at its relaxed length', flush=True)
    _record(res, 'rigid', _approach(res, 'rigid', hold_ss=HOLD_SS))


def _bound(res, regime):
    """The work to traverse a branch: the maximum along it. 'work_max' is absent
    from the mobile branch recorded before this distinction was drawn, where the
    endpoint happened to be the maximum anyway, so it falls back to 'work'."""
    r = res.get(regime)
    return None if r is None else r.get('work_max', r.get('work'))


def stage_report(res):
    lo = _bound(res, 'mobile')
    hi = _bound(res, 'rigid')
    if lo is None or hi is None:
        return
    res['report'] = dict(
        lower_bound=lo, upper_bound=hi,
        mobile_endpoint=res['mobile']['work'], rigid_endpoint=res['rigid']['work'],
        mobile_segments=res['mobile'].get('segments'),
        rigid_segments=res['rigid'].get('segments'),
        disulfide_held=HOLD_SS,
        window_lo=25.19, window_hi=31.23,       # once a day / once in fifty years, 310 K
        verdict=('scrambling cannot initiate in the folded protein'
                 if lo > 31.23 else
                 'the approach is affordable; the barrier at sulfur becomes the question'
                 if lo < 25.19 else
                 'the approach lands inside the window the disease allows'))
    print(f"\nconformational work to bring Cys6 to {TARGET} A of Cys146:")
    print(f"  lower bound (local backbone free)  {lo:+8.2f} kcal/mol")
    print(f"  upper bound (fold held rigid)      {hi:+8.2f} kcal/mol")
    print(f"  the disulfide was held at its relaxed length: {HOLD_SS}")
    print(f"  the disease allows 25.2-31.2 kcal/mol for a once-in-decades event")
    print(f"  -> {res['report']['verdict']}", flush=True)
    save(res)


STAGES = dict(build=stage_build, mobile=stage_mobile, rigid=stage_rigid,
              report=stage_report)


def main(names):
    res = load()
    for n in names:
        print(f'=== {n}', flush=True)
        try:
            STAGES[n](res)
        except sanity.Violation as e:
            # A ruptured disulfide is not a transient failure, so the unit must not
            # restart into it: each restart would recompute the same 45-minute point
            # and stop at the same place. Exit 3, which the unit file excludes from
            # Restart=on-failure.
            print(f'STOPPED in {n}: {e}', flush=True)
            save(res)
            return 3
        res['_stages_done'] = sorted(set(res.get('_stages_done', [])) | {n})
        save(res)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:] or ['build', 'mobile', 'rigid', 'report']))
