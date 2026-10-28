#!/usr/bin/env python
"""Is the TDP-43 autoregulatory loop bistable, and if so what sets the threshold?

The biology this rests on is established. TDP-43 binds its own 3'UTR and represses
its own production, so nuclear TDP-43 holds itself at a setpoint by negative
feedback. The consequence that matters is what happens when it leaves: repression
lifts, the cell makes MORE TDP-43, and the new protein partitions to the cytoplasm
where the aggregate is. The negative feedback inverts into positive feedback. A
2025 preprint (biorxiv 2025.11.12.687969, now Sci. Rep.) describes exactly that
loop and reports "a threshold at which modestly increased cytoplasmic residency
becomes pathogenic" -- but the threshold there is conceptual. Nobody has written
the model.

If the loop is bistable then "does TDP-43 mislocalisation cause ALS or follow from
something upstream" is the wrong question. A bistable system has two stable states
separated by an unstable one; below the threshold any insult decays, above it the
system runs to collapse with no further input. The trigger then need not be large,
specific or identifiable -- only sufficient, once. What distinguishes a vulnerable
cell is not its exposure but its MARGIN.

Three variables, in hours:

    N   nuclear TDP-43, the functional pool that does the splicing
    C   cytoplasmic TDP-43, soluble
    A   aggregated TDP-43, insoluble

    dN/dt = kin(A).C - kout.N - dN.N
    dC/dt = p(N) + kout.N - kin(A).C - dC.C - agg(C,A)
    dA/dt = agg(C,A) - dA.A

with two nonlinearities, each standing for an established piece of biology:

    p(N)     = p0 / (1 + (N/K)^h)          autoregulation: nuclear TDP-43
                                           represses its own production
    kin(A)   = kin0 / (1 + (A/KA)^m)       aggregates impair nucleocytoplasmic
                                           transport, which is reported for
                                           C9orf72 dipeptides and for TDP-43
                                           inclusions themselves
    agg(C,A) = kn.C^2 + kg.C.A             primary nucleation plus templated
                                           growth; the C.A term is the
                                           prion-like self-templating that
                                           explains contiguous spread

The parameters are ILLUSTRATIVE, chosen for the right order of magnitude rather
than fitted: protein half-life about a day, shuttling on the scale of an hour,
roughly 85 % of TDP-43 nuclear at rest. The question asked here is not "what is
the threshold in a motor neuron" but "can this class of model be bistable at
all, and what moves the threshold". Those are different questions and only the
second is answerable without data this model does not have.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'tdp43_bistability.json'

BASE = dict(
    p0=0.60,      # max production rate, arbitrary concentration units per hour
    K=1.0,        # autoregulation setpoint, in units of nuclear TDP-43
    h=2.0,        # autoregulation cooperativity
    kin0=6.0,     # import rate, 1/h
    KA=0.40,      # aggregate level at which import is halved
    m=2.0,        # steepness of transport impairment
    kout=1.0,     # export rate, 1/h
    dN=0.05,      # nuclear turnover, 1/h  (about a 14 h half-life)
    dC=0.05,      # cytoplasmic turnover
    dA=0.020,     # aggregate clearance, slow: autophagy, not the proteasome
    kn=1.0e-4,    # primary nucleation: rare, as nucleation must be
    kg=0.040,     # templated growth, the autocatalytic term
)
# The condition for a healthy state to exist at all is kg.C_healthy < dA: templated
# growth must lose to clearance while the cytoplasmic pool is small. The first run
# of this model had kg.C = 0.090 x 0.111 = 0.0100 against dA = 0.0100, sitting
# exactly on the bifurcation, so there was no healthy state and everything
# collapsed. That is a parameter error and not a result, and it is recorded here
# because the same arithmetic is the whole mechanism: bistability needs clearance
# to win at low A and lose at high A, and what flips it is the cytoplasmic pool
# rising as aggregates block import.


def rhs(t, y, p):
    N, C, A = np.maximum(y, 0.0)
    prod = p['p0'] / (1.0 + (N / p['K']) ** p['h'])
    kin = p['kin0'] / (1.0 + (A / p['KA']) ** p['m'])
    agg = p['kn'] * C * C + p['kg'] * C * A
    return [kin * C - p['kout'] * N - p['dN'] * N,
            prod + p['kout'] * N - kin * C - p['dC'] * C - agg,
            agg - p['dA'] * A]


def settle(y0, p, t_end=20000.0):
    """Integrate to a steady state and return it."""
    s = solve_ivp(rhs, (0, t_end), y0, args=(p,), method='LSODA',
                  rtol=1e-9, atol=1e-12, dense_output=False)
    return s.y[:, -1]


def states(p):
    """The attractors reached from a healthy start and from a seeded start."""
    healthy = settle([1.0, 0.15, 0.0], p)
    seeded = settle([1.0, 0.15, 3.0], p)      # a large aggregate seed
    return healthy, seeded


def is_bistable(p, tol=1e-3):
    h, s = states(p)
    return abs(h[2] - s[2]) > tol * max(1.0, s[2]), h, s


def threshold(p, lo=0.0, hi=5.0):
    """Smallest aggregate seed that does NOT decay: the separatrix in A."""
    def lands_high(a0):
        y = settle([1.0, 0.15, a0], p, t_end=20000.0)
        return y[2]
    ok, h, s = is_bistable(p)
    if not ok:
        return None
    mid = 0.5 * (h[2] + s[2])
    f = lambda a0: lands_high(a0) - mid
    if f(lo) > 0 or f(hi) < 0:
        return None
    return brentq(f, lo, hi, xtol=1e-4)


def transient_test(p, push_hours, export_boost=14.0):
    """The experiment that discriminates: force TDP-43 out for a while, then stop.

    A graded system returns to health when the push is removed. A bistable one
    does not -- past the separatrix it keeps going on its own. This is what the
    export-biased construct in that preprint could be used to do."""
    pp = dict(p); pp['kout'] = p['kout'] * export_boost
    y = solve_ivp(rhs, (0, push_hours), [1.0, 0.15, 0.0], args=(pp,),
                  method='LSODA', rtol=1e-9, atol=1e-12).y[:, -1]
    end = settle(y, p)                 # push removed, normal parameters restored
    return dict(push_hours=push_hours, N_after_push=float(y[0]), A_after_push=float(y[2]),
                N_final=float(end[0]), A_final=float(end[2]))


def main():
    res = dict(model=__doc__.split('\n\n')[0], parameters=BASE)

    ok, healthy, seeded = is_bistable(BASE)
    res['bistable'] = bool(ok)
    res['healthy_state'] = dict(N=float(healthy[0]), C=float(healthy[1]), A=float(healthy[2]))
    res['collapsed_state'] = dict(N=float(seeded[0]), C=float(seeded[1]), A=float(seeded[2]))
    print('bistable: %s' % ok)
    print('  healthy   N %.3f  C %.3f  A %.3f' % tuple(healthy))
    print('  collapsed N %.3f  C %.3f  A %.3f' % tuple(seeded))
    if ok:
        print('  nuclear TDP-43 retained in the collapsed state: %.0f %% of healthy'
              % (100 * seeded[0] / healthy[0]))

    thr = threshold(BASE)
    res['separatrix_A'] = thr
    if thr is not None:
        print('  threshold seed A* = %.3f  (healthy A = %.3f)' % (thr, healthy[2]))

    # the discriminating experiment, at several push durations
    print('\ntransient push then release:')
    res['transient'] = []
    for hrs in (6, 12, 24, 48, 96, 168):
        t = transient_test(BASE, hrs)
        verdict = 'RECOVERS' if t['A_final'] < 0.5 * res['collapsed_state']['A'] else 'collapses'
        t['verdict'] = verdict
        res['transient'].append(t)
        print('  %4d h push -> A %.3f at release -> final A %.3f   %s'
              % (hrs, t['A_after_push'], t['A_final'], verdict))

    # what moves the threshold: the margin
    print('\nwhat sets the margin (threshold seed A*):')
    res['margin_scan'] = {}
    for name, key, factors in (
            ('aggregate clearance', 'dA', (0.6, 0.8, 1.0, 1.3, 1.8)),
            ('transport impairment KA', 'KA', (0.7, 0.85, 1.0, 1.2, 1.5)),
            ('templated growth', 'kg', (0.7, 0.85, 1.0, 1.2, 1.5)),
            ('autoregulation strength p0', 'p0', (0.85, 0.95, 1.0, 1.1, 1.25))):
        row = []
        for f in factors:
            p = dict(BASE); p[key] = BASE[key] * f
            t = threshold(p)
            row.append(dict(factor=f, value=p[key], threshold=t))
            print('  %-26s x%.2f -> A* %s' % (name, f, ('%.3f' % t) if t else 'NOT BISTABLE'))
        res['margin_scan'][name] = row

    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print('\nsaved %s' % OUT.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
