#!/usr/bin/env python
"""S_N2 at a chalcogen: the barrier for displacement at S versus at Se.

Why this and not the disulfide barrier. The accessibility result (0.00 A^2 for both
Cys57 and Cys146 sulfurs) and the attack geometry (Cys6 at 8.37 A / 133 deg, Cys111
at 18.36 A / 167.8 deg) together say the destructive reaction cannot start in the
folded protein, so its barrier is not a computable quantity without a destabilised
ensemble that no structure exists for. The protective direction does not have that
problem: ebselen catalyses formation of the same disulfide, and the product of that
reaction is the structure we know at 1.24 A.

Each system here is one displacement, driven along

    xi = d(X...Y) - d(Y-Z)

the same antisymmetric stretch used for the proton transfers, with the same
restraint and the same P-RFO saddle search. See chemoqbit/chalcogen.py for why
these six systems and not others.

Stages, each resumable and each checkpointed on completion:

    prep    build the encounter complex, relax it at fixed xi, relax the product
    scan    relaxed xTB scan along xi in both directions, merged into one profile
    saddle  DFT saddle from the scan maximum, finite-difference Hessian, frequencies
    dft     wB97X-D3(BJ)/def2-TZVP with C-PCM at reactant, saddle and product
    cc      CCSD(T)/def2-TZVP single points, and DF-MP2 for the basis correction

The scan runs both ways because a one-directional sweep hid a 30 kcal/mol error in
the prolyl work, and the profile is checked for a drop along a monotonic run because
that is how three bad coordinates were caught.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chemoqbit import chalcogen as CH          # noqa: E402
from chemoqbit import sanity                   # noqa: E402
from chemoqbit.enzyme import xtb               # noqa: E402
from chemoqbit.enzyme.frozen import optimise_frozen   # noqa: E402

HARTREE_KCAL = 627.5094740631
SOLVENT = 'water'          # the reaction is a thiolate in water, not in chloroform
STEP = 0.15                # A in xi; fine enough that the maximum is not stepped over
SPAN = 2.4                 # A either side of the symmetric point

OUT = None
LAB = None


def save(d):
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(OUT)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def _relax_at(sym, X, charge, idx, x0, k=20.0):
    """Relax everything with xi restrained to x0."""
    cons = [('relay', [(idx['nuc'], idx['attacked'], idx['leaving'])], x0)]
    Y, E = optimise_frozen(sym, X, charge=charge, frozen=(), constraints=cons,
                           k=k, solvent=SOLVENT)
    return Y, E


def stage_prep(res):
    if 'prep' in res:
        return
    t0 = time.time()
    sym, X, charge, idx = CH.build(LAB)

    # the encounter complex: hold xi where the build put it so the fragments neither
    # fly apart nor collapse into the product during the first relaxation
    x_start = CH.xi(X, idx)
    Xr, Er = _relax_at(sym, X, charge, idx, x_start)

    ang = CH.attack_angle(Xr, idx)
    sanity.record(res, 'chalcogen_attack_angle', ang, label=f'{LAB} reactant', fatal=False)

    # No constructed product. Mirroring the reactant through the symmetric point put
    # se_s's backward sweep 40 kcal/mol above its forward sweep at EVERY point, not at
    # one: the guess landed in another basin and the sweep never left it. The cause was
    # product_geometry running its connectivity walk on coordinates whose nucleophile
    # had already been shifted in, so the walk crossed the fragment boundary and
    # displaced atoms belonging to the other side. The product is instead taken from the
    # far end of the forward sweep, which arrives there by neighbour seeding -- the same
    # fix that stopped the 2D abstraction columns wrenching the proton off its donor.
    # By the definition of xi the product sits at -x_start, so the grid needs no guess.
    bl_r = CH.bond_lengths(Xr, idx)
    res['prep'] = dict(
        symbols=sym, charge=charge, idx={k: v for k, v in idx.items()},
        reactant=dict(X=Xr.tolist(), E=Er, xi=CH.xi(Xr, idx), angle=ang, **bl_r),
        product_xi=-x_start,
        seconds=time.time() - t0)
    print(f'{LAB}: reactant xi {CH.xi(Xr, idx):+.2f} A, forming {bl_r["forming"]:.2f}, '
          f'breaking {bl_r["breaking"]:.2f}, angle {ang:.1f} deg', flush=True)
    print(f'{LAB}: product end of the grid at xi {-x_start:+.2f}, reached by sweeping',
          flush=True)


def _sweep(sym, X0, charge, idx, grid):
    """Relaxed scan along the grid, each point seeded from the previous one. Seeding
    from the neighbour rather than restarting from the reactant is what stopped the
    2D abstraction columns from wrenching the proton off its donor."""
    X = np.array(X0, float)
    pts = []
    for x0 in grid:
        X, E = _relax_at(sym, X, charge, idx, x0)
        pts.append(dict(xi=x0, E=E, angle=CH.attack_angle(X, idx),
                        **CH.bond_lengths(X, idx), X=X.tolist()))
        print(f'  xi {x0:+.2f}  E {E:+.6f} Eh  angle {pts[-1]["angle"]:.0f}', flush=True)
    return pts


def stage_scan(res):
    if 'scan' in res:
        return
    t0 = time.time()
    p = res['prep']
    sym, charge, idx = p['symbols'], p['charge'], p['idx']
    Xr = np.array(p['reactant']['X'])

    x_r = p['reactant']['xi']
    x_p = p.get('product_xi', -x_r)
    lo, hi = min(x_r, x_p), max(x_r, x_p)
    grid = np.arange(lo, hi + 0.5 * STEP, STEP)

    # the forward sweep starts at the reactant, wherever on the grid that falls
    fwd_grid = grid if abs(grid[0] - x_r) < abs(grid[-1] - x_r) else grid[::-1]
    print(f'{LAB}: forward sweep, {len(grid)} points from {fwd_grid[0]:+.2f} '
          f'to {fwd_grid[-1]:+.2f}', flush=True)
    fwd = _sweep(sym, Xr, charge, idx, fwd_grid)

    # and the backward sweep starts from where the forward one ARRIVED, not from a
    # constructed product. That is the whole point: the far end is reached by relaxing
    # through every intermediate geometry, so it is in the right basin by construction.
    print(f'{LAB}: backward sweep from the far end of the forward sweep', flush=True)
    bwd = _sweep(sym, np.array(fwd[-1]['X']), charge, idx, fwd_grid[::-1])

    # keep the lower of the two passes at each xi: a relaxed scan can be trapped in
    # one direction and not the other, and the lower branch is the physical one
    by_xi = {}
    for pt in fwd + bwd:
        key = round(pt['xi'], 3)
        if key not in by_xi or pt['E'] < by_xi[key]['E']:
            by_xi[key] = pt
    prof = [by_xi[k] for k in sorted(by_xi)]

    E0 = min(pt['E'] for pt in prof)
    for pt in prof:
        pt['rel'] = (pt['E'] - E0) * HARTREE_KCAL

    rel = [pt['rel'] for pt in prof]

    # xi is POSITIVE at the reactant (nucleophile far, bond short) and negative at the
    # product, so prof[0] is the product end. Measuring the barrier from prof[0] is
    # harmless for an identity exchange, where the two ends are equal by symmetry, and
    # wrong for se_s and s_se, where they are not.
    r_end = int(np.argmin([abs(pt['xi'] - p['reactant']['xi']) for pt in prof]))
    p_end = int(np.argmin([abs(pt['xi'] - x_p) for pt in prof]))
    top = int(np.argmax(rel))
    barrier = rel[top] - rel[r_end]
    reaction_energy = rel[p_end] - rel[r_end]

    # Whether the SYMMETRIC arrangement is bound, measured against the lower of the two
    # ends. Taking the global minimum instead conflates a bound hypervalent species with
    # an exothermic reaction: s_se runs 29.8 kcal/mol downhill to its product because
    # selenolate is far more stable than thiolate, and the first version of this reported
    # that as a 29.8 kcal/mol well when the profile is in fact monotonic.
    sym = int(np.argmin([abs(pt['xi']) for pt in prof]))
    well_depth = max(0.0, min(rel[r_end], rel[p_end]) - rel[sym])

    # Per-direction energies, because a hysteresis failure is only diagnosable if the
    # two sweeps are kept: the merge takes the lower point at each xi and therefore
    # hides whichever sweep blew up. se_s disagreed by 46 kcal/mol and the merged
    # profile looked clean, with nothing left on disk to say where it went wrong.
    sweeps = dict(forward=[dict(xi=q['xi'], E=q['E']) for q in fwd],
                  backward=[dict(xi=q['xi'], E=q['E']) for q in bwd])
    worst = max(zip(fwd, bwd[::-1]), key=lambda fb: abs(fb[0]['E'] - fb[1]['E']))
    hysteresis = abs(worst[0]['E'] - worst[1]['E']) * HARTREE_KCAL

    res['scan'] = dict(
        profile=[{k: v for k, v in pt.items() if k != 'X'} for pt in prof],
        geometries={f"{pt['xi']:.2f}": pt['X'] for pt in prof},
        sweeps=sweeps,
        barrier_xtb=barrier, top_xi=prof[top]['xi'], well_depth=well_depth,
        well_xi=prof[sym]['xi'], reactant_xi=prof[r_end]['xi'],
        reaction_energy=reaction_energy,
        hysteresis=hysteresis, hysteresis_xi=worst[0]['xi'],
        seconds=time.time() - t0)

    sanity.record(res, 'scan_drop', sanity.largest_drop(rel), label=f'{LAB} profile')
    sanity.record(res, 'exchange_segments', sanity.monotonic_segments(rel),
                  label=f'{LAB} merged profile')
    sanity.record(res, 'chalcogen_barrier', barrier, label=f'{LAB} xTB')
    sanity.record(res, 'exchange_well', well_depth, label=f'{LAB} symmetric species')
    if LAB in ('ss', 'sese'):
        # the two ends are the same two species, so this is pure scan error
        sanity.record(res, 'identity_asymmetry', reaction_energy,
                      label=f'{LAB} identity exchange')
    sanity.record(res, 'scan_hysteresis', hysteresis, label=f'{LAB} fwd vs bwd')
    for pt in prof:
        sanity.record(res, 'chalcogen_attack_angle', pt['angle'],
                      label=f"{LAB} xi {pt['xi']:+.2f}")

    print(f'{LAB}: xTB barrier {barrier:+.2f} kcal/mol at xi {prof[top]["xi"]:+.2f}, '
          f'dE_rxn {reaction_energy:+.2f}, symmetric species {well_depth:+.2f} bound, '
          f'hysteresis {hysteresis:.2f} at xi {worst[0]["xi"]:+.2f}', flush=True)


def stage_saddle(res):
    """DFT saddle from the scan maximum, with one imaginary mode as the certificate."""
    if 'saddle' in res:
        return
    from chemoqbit.enzyme.tsopt import fd_hessian, frequencies, prfo_bofill
    from pyscf import dft, gto

    t0 = time.time()
    p, s = res['prep'], res['scan']
    sym, charge, idx = p['symbols'], p['charge'], p['idx']
    X0 = np.array(s['geometries'][f"{s['top_xi']:.2f}"])

    def energy_grad(X):
        mol = gto.M(atom=[(a, tuple(r)) for a, r in zip(sym, X)], basis='def2-svp',
                    charge=charge, verbose=0)
        mf = dft.RKS(mol).density_fit()
        mf.xc = 'wb97x_d3bj'
        mf = mf.PCM()
        mf.with_solvent.method = 'C-PCM'
        mf.with_solvent.eps = 78.3553
        e = mf.kernel()
        g = mf.nuc_grad_method().kernel()
        return float(e), np.array(g)

    # the guide is the lowest eigenvector of the Hessian, not a hand-built vector:
    # a guessed displacement overlapped 0.72 with a rotational null-space residual in
    # the prolyl work and followed the wrong mode
    H = fd_hessian(np.array(X0), lambda Y: energy_grad(Y), free=range(len(sym)), h=0.01)
    w, V = np.linalg.eigh(H)
    guide = V[:, 0]

    X, info = prfo_bofill(np.array(X0), energy_grad, H, guide, frozen=(),
                          trust=0.06, gtol=3e-4, maxiter=120, neg=1e-5)
    Hs = fd_hessian(X, lambda Y: energy_grad(Y), free=range(len(sym)), h=0.01)
    nu = frequencies(Hs, sym)
    imag = [v for v in nu if v < 0]

    res['saddle'] = dict(X=X.tolist(), E=info['E'], n_imag=len(imag),
                         imag_cm=[abs(v) for v in imag],
                         angle=CH.attack_angle(X, idx), **CH.bond_lengths(X, idx),
                         xi=CH.xi(X, idx), seconds=time.time() - t0)
    sanity.record(res, 'n_imaginary', len(imag), label=f'{LAB} saddle', fatal=False)
    if imag:
        sanity.record(res, 'imag_wavenumber', abs(imag[0]), label=f'{LAB} saddle')
    sanity.record(res, 'chalcogen_attack_angle', res['saddle']['angle'],
                  label=f'{LAB} saddle')
    print(f'{LAB}: saddle has {len(imag)} imaginary mode(s) '
          f'{[f"{abs(v):.0f}i" for v in imag]}, xi {CH.xi(X, idx):+.2f}', flush=True)


STAGES = dict(prep=stage_prep, scan=stage_scan, saddle=stage_saddle)


def main(label, names):
    global OUT, LAB
    LAB = label
    OUT = ROOT / 'results' / f'chalcogen_{label}.json'
    res = load()
    for n in names:
        print(f'=== {n}', flush=True)
        STAGES[n](res)
        res['_stages_done'] = sorted(set(res.get('_stages_done', [])) | {n})
        save(res)


if __name__ == '__main__':
    if sys.argv[1] == 'all':
        for lab in CH.SYSTEMS:
            main(lab, sys.argv[2:] or ['prep', 'scan'])
    else:
        main(sys.argv[1], sys.argv[2:] or ['prep', 'scan'])
