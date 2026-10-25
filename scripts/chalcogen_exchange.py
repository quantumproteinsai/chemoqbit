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


XC, DFT_BASIS = 'wb97x-d3bj', 'def2-svp'    # the hyphen form; the underscore one is not a
                                            # libxc name and fails to parse
# The two environments the question turns on. Thiolate-disulfide exchange is reported to
# pass through a bound trichalcogen anion in low dielectric and a plain saddle once
# solvated, so the topology is a property of the surroundings and not of the molecule.
# SOD1 decides which applies: Cys6, Cys57 and Cys146 are all at 0.00 A^2 accessible
# surface, so that reaction happens desolvated, in the eps ~ 4 column.
DIELECTRIC = dict(protein=4.0, water=78.3553)


def _dft_energy_grad(sym, charge, eps, basis=DFT_BASIS):
    from pyscf import dft, gto

    def eg(X):
        mol = gto.M(atom=[(a, tuple(r)) for a, r in zip(sym, X)], basis=basis,
                    charge=charge, verbose=0)
        mf = dft.RKS(mol).density_fit()
        mf.xc = XC
        mf = mf.PCM()
        mf.with_solvent.method = 'C-PCM'
        mf.with_solvent.eps = eps
        e = mf.kernel()
        g = mf.nuc_grad_method().kernel()
        return float(e), np.array(g)
    return eg


def _xi_penalty(idx, x0, k):
    """Harmonic restraint on xi and its gradient, matching the xTB 'relay' form."""
    def pen(Y):
        u1 = Y[idx['nuc']] - Y[idx['attacked']]
        d1 = np.linalg.norm(u1)
        u2 = Y[idx['attacked']] - Y[idx['leaving']]
        d2 = np.linalg.norm(u2)
        dev = (d1 - d2) - x0
        G = np.zeros_like(Y)
        ff = 2 * k * dev
        G[idx['nuc']] += ff * u1 / d1
        G[idx['attacked']] -= ff * u1 / d1
        G[idx['attacked']] -= ff * u2 / d2
        G[idx['leaving']] += ff * u2 / d2
        return k * dev ** 2, G
    return pen


def _minimise(X0, eg, pen=None, maxiter=80, gtol=3e-4):
    from scipy.optimize import minimize
    shape = np.array(X0).shape

    def f(v):
        Y = v.reshape(shape)
        e, g = eg(Y)
        if pen is not None:
            ep, gp = pen(Y)
            e, g = e + ep, g + gp
        return e, g.ravel()

    r = minimize(f, np.array(X0, float).ravel(), jac=True, method='L-BFGS-B',
                 options=dict(maxiter=maxiter, gtol=gtol, ftol=1e-12))
    return r.x.reshape(shape), float(r.fun), bool(r.success)


def stage_topology(res):
    """Is the symmetric trichalcogen arrangement a minimum or a saddle, and does the
    answer change with the dielectric?

    This replaces a single saddle search, which assumed the answer. The protocol is the
    same in both columns: hold xi at zero and relax everything else, so the symmetric
    structure is clean; release the restraint and minimise; then take a
    finite-difference Hessian and count imaginary modes at whatever it converged to.

        0 imaginary  ->  a bound intermediate, so the mechanism is addition-elimination
        1 imaginary  ->  the transition state, so the mechanism is concerted
        runs away    ->  xi = 0 is not a stationary point at all

    A bound intermediate is the therapeutically interesting answer, because an
    intermediate is a species that exists long enough to be stabilised or occupied and
    a saddle is not.
    """
    res.setdefault('topology', {})
    p, s = res['prep'], res['scan']
    sym, charge, idx = p['symbols'], p['charge'], p['idx']
    key = min(s['geometries'], key=lambda k: abs(float(k)))
    X_sym = np.array(s['geometries'][key])

    # A full finite-difference Hessian is 90 gradients, and one DFT gradient on this
    # system costs 77 s, so that is two hours per column before anything is learned.
    # The curvature along xi answers the same question for a fifth of the cost: fit a
    # parabola to five relaxed points either side of the symmetric arrangement and read
    # its sign. Positive curvature means xi = 0 is a minimum along the reaction
    # coordinate, hence a bound intermediate; negative means it is the barrier top.
    GRID = (-0.30, -0.15, 0.00, 0.15, 0.30)

    for name, eps in DIELECTRIC.items():
        if name in res['topology']:
            continue
        t0 = time.time()
        eg = _dft_energy_grad(sym, charge, eps)

        X = np.array(X_sym, float)
        pts = []
        for x0 in GRID:
            # seeded from the neighbour, so every point after the first needs few steps
            X, E, conv = _minimise(X, eg, pen=_xi_penalty(idx, x0, 20.0), maxiter=40)
            Ec, _ = eg(X)                      # energy without the restraint term
            pts.append(dict(xi_target=x0, xi=CH.xi(X, idx), E=Ec, converged=conv,
                            angle=CH.attack_angle(X, idx), **CH.bond_lengths(X, idx)))
            print(f'  {name} xi {x0:+.2f} -> {pts[-1]["xi"]:+.3f}  E {Ec:+.6f} Eh  '
                  f'angle {pts[-1]["angle"]:.0f}', flush=True)

        xs = np.array([q['xi'] for q in pts])
        es = (np.array([q['E'] for q in pts]) - min(q['E'] for q in pts)) * HARTREE_KCAL
        c2, c1, c0 = np.polyfit(xs, es, 2)      # kcal/mol per A^2

        verdict = ('bound intermediate' if c2 > 0 else 'barrier top')
        res['topology'][name] = dict(
            eps=eps, profile=pts, rel_kcal=es.tolist(),
            curvature=float(c2), slope=float(c1), verdict=verdict,
            seconds=time.time() - t0)
        save(res)

        for q in pts:
            sanity.record(res, 'chalcogen_attack_angle', q['angle'],
                          label=f'{LAB} {name} xi {q["xi"]:+.2f}')
        print(f'{LAB} eps={eps:.1f} ({name}): d2E/dxi2 = {c2:+.1f} kcal/mol/A^2 '
              f'-> {verdict}  ({time.time() - t0:.0f}s)', flush=True)

    a, b = res['topology'].get('protein'), res['topology'].get('water')
    if a and b:
        res['topology']['solvent_changes_topology'] = (a['verdict'] != b['verdict'])
        print(f"{LAB}: protein-interior '{a['verdict']}' vs water '{b['verdict']}' -- "
              f"dielectric {'DOES' if a['verdict'] != b['verdict'] else 'does not'} "
              f"change the mechanism", flush=True)


STAGES = dict(prep=stage_prep, scan=stage_scan, topology=stage_topology)


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
