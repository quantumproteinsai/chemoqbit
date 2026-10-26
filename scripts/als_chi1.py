#!/usr/bin/env python
"""Can a side-chain rotation open the Cys146 attack site, and what does it cost?

The backside attack point on Cys146 -- 3.0 A beyond it, collinear with the S-S
bond being broken -- is occupied in the crystal structure by Thr116 CG2 at 1.92 A
and Val118 CG2 at 2.14 A, against a sulfur-carbon van der Waals contact of about
3.5 A. Those are side chains, not backbone, and a chi1 rotation moves a
gamma-substituent several angstroms.

So before paying for the full conformational-work scan, ask the cheap question: is
there any rotamer pair that clears the site, and what does it cost? If a rotation
opens it for a couple of kcal/mol then the site is not really blocked and the
expensive calculation is answering the wrong question.

This is a RIGID rotation with single-point energies: the side chains beyond CB are
turned about the CA-CB axis and nothing relaxes. That makes the energies upper
bounds -- a clash that relaxation would partly relieve shows here at full cost --
and it makes the clearances exact, since the attack point depends only on Cys57
and Cys146, which do not move. Any rotamer that looks promising is worth relaxing
properly afterwards; one that is still blocked when nothing can push back is
blocked.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chemoqbit import als_approach as A          # noqa: E402
from chemoqbit.enzyme import xtb                 # noqa: E402

HARTREE_KCAL = 627.5094740631
STEP = 30.0                     # degrees; 12 x 12 = 144 rotamer pairs
REACTIVE = 3.0                  # A beyond S146, where the thiolate must arrive
CLEAR = 3.3                     # a sulfur-carbon contact; below this the site is blocked
OUT = ROOT / 'results' / 'als_chi1_scan.json'


def _bonded_beyond(X, start, exclude, cut=1.9):
    """Atoms reachable from `start` without passing through `exclude`."""
    seen, stack = {start}, [start]
    while stack:
        a = stack.pop()
        for b in range(len(X)):
            if b in seen or b == exclude:
                continue
            if np.linalg.norm(X[a] - X[b]) < cut:
                seen.add(b)
                stack.append(b)
    return sorted(seen)


def _rotate(X, idx_move, p0, axis, deg):
    """Rigid rotation of `idx_move` about the line through p0 along `axis`."""
    k = axis / np.linalg.norm(axis)
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    Y = np.array(X, float)
    v = Y[idx_move] - p0
    Y[idx_move] = (v * c + np.cross(k, v) * s
                   + np.outer(np.dot(v, k), k) * (1 - c)) + p0
    return Y


def main():
    t0 = time.time()
    sym, X0, q, idx, lab, _r, _m = A.build()
    symH, XH, qH, idx, labH = A.protonate(sym, X0, q, idx, lab)

    def find(name):
        return next(i for i, l in enumerate(labH) if l == name)

    groups = {}
    for res, g in (('THR116', 'CG2'), ('VAL118', 'CG1')):
        ca, cb = find(f'{res}:CA'), find(f'{res}:CB')
        groups[res] = dict(ca=ca, cb=cb, gamma=find(f'{res}:{g}'),
                           moving=[i for i in _bonded_beyond(XH, cb, ca) if i != cb])
        print('%s: rotating %d atoms about CA-CB' % (res, len(groups[res]['moving'])),
              flush=True)

    S57, S146 = idx['S57'], idx['S146']
    u = XH[S146] - XH[S57]
    u = u / np.linalg.norm(u)
    P = XH[S146] + REACTIVE * u          # fixed: neither sulfur moves

    fixed = {S57, S146}

    def clearance(Y):
        """Distance from the attack point to the nearest atom that is not a
        reacting sulfur. This is what has to exceed a van der Waals contact."""
        best, who = 1e9, None
        for i, (x, l) in enumerate(zip(Y, labH)):
            if i in fixed:
                continue
            d = float(np.linalg.norm(x - P))
            if d < best:
                best, who = d, l
        return best, who

    grid = np.arange(0.0, 360.0, STEP)
    e_ref = None
    pts = []
    for a in grid:
        Ya = _rotate(XH, groups['THR116']['moving'], XH[groups['THR116']['cb']],
                     XH[groups['THR116']['cb']] - XH[groups['THR116']['ca']], a)
        for b in grid:
            Y = _rotate(Ya, groups['VAL118']['moving'], Ya[groups['VAL118']['cb']],
                        Ya[groups['VAL118']['cb']] - Ya[groups['VAL118']['ca']], b)
            c, who = clearance(Y)
            e, _g = xtb.energy_grad(symH, Y, charge=qH, solvent='water')
            if a == 0.0 and b == 0.0:
                e_ref = e
            pts.append(dict(chi1_thr=float(a), chi1_val=float(b),
                            clearance=c, nearest=who,
                            dE=(e - e_ref) * HARTREE_KCAL if e_ref is not None else 0.0))
        print(f'  Thr116 chi1 {a:+6.1f}  done ({time.time() - t0:.0f}s)', flush=True)

    open_pts = [p for p in pts if p['clearance'] >= CLEAR]
    best = min(open_pts, key=lambda p: p['dE']) if open_pts else None
    wt = pts[0]

    res = dict(
        question='Can a chi1 rotation of Thr116 or Val118 clear the Cys146 attack site?',
        method=('Rigid rotation about CA-CB with GFN2-xTB/ALPB single points on the 259-atom '
                'cluster. Nothing relaxes, so the energies are UPPER bounds and the '
                'clearances are exact.'),
        attack_point=f'{REACTIVE} A beyond Cys146 SG, collinear with the Cys57-Cys146 bond',
        clear_threshold=CLEAR, step_deg=STEP, n_points=len(pts),
        crystal=dict(clearance=wt['clearance'], nearest=wt['nearest']),
        n_open=len(open_pts),
        best_open=best,
        conclusion=(
            f"No rotamer clears the site: the best clearance over {len(pts)} rotamer pairs is "
            f"{max(p['clearance'] for p in pts):.2f} A, still inside a van der Waals contact."
            if not open_pts else
            f"{len(open_pts)} of {len(pts)} rotamer pairs clear the site, the cheapest at "
            f"{best['dE']:+.2f} kcal/mol (Thr116 chi1 {best['chi1_thr']:+.0f}, Val118 "
            f"{best['chi1_val']:+.0f}), giving {best['clearance']:.2f} A."),
        profile=pts, seconds=time.time() - t0, date='2026-10-26')
    OUT.write_text(json.dumps(res, indent=1) + '\n')

    print()
    print(f"crystal clearance {wt['clearance']:.2f} A (nearest: {wt['nearest']})")
    print(f"best clearance over the scan: {max(p['clearance'] for p in pts):.2f} A")
    print(res['conclusion'])


if __name__ == '__main__':
    main()
