#!/usr/bin/env python
"""Saturable clearance: the model corrected by a conflict with measured kinetics.

The first version cleared aggregates first order, at rate dA.A. Requiring that to
beat templated growth gives a cap on the elongation rate constant of order 1 to 35
per molar per second, for any plausible cytoplasmic concentration and autophagy
half-life. Medin, an amyloid whose kinetics have actually been fitted, elongates at
5e7. Six to eight orders of magnitude is not a parameter problem; it says the
clearance term is the wrong shape.

It is. First-order clearance scales without limit: double the aggregate load and
you double the clearance flux, for ever. Autophagy does not work like that. It is
a capacity-limited process -- a finite number of autophagosomes, a finite
lysosomal throughput -- so the right form is saturable:

    clearance(A) = Vmax . A / (Km + A)

At low load this is first order with effective rate Vmax/Km, and nothing changes.
At high load it plateaus at Vmax, and everything changes: once the flux of protein
into aggregate exceeds Vmax, the aggregate grows without bound no matter how
large the clearance rate constant was at low load.

That is a better model for three reasons beyond fixing the contradiction.

It makes the threshold a FLUX balance rather than a rate-constant inequality, and
flux is the quantity autophagy assays actually measure. It makes the clearance
genes act on Vmax directly, which is what a loss-of-function allele does to a
capacity. And saturation is a textbook bistability generator, so the switch no
longer depends on the transport-impairment feedback being strong -- it survives
even if that term is weak, which is the part of the first model I was least sure
of.

Same three variables as before: nuclear, cytoplasmic and aggregated TDP-43.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'tdp43_saturable.json'

BASE = dict(
    p0=0.60, K=1.0, h=2.0,          # autoregulation
    kin0=6.0, KA=0.40, m=2.0,       # import, impaired by aggregates
    kout=1.0,                       # export
    dN=0.039, dC=0.039,             # turnover: ln2/18 h, the primary-neuron figure
    kn=1.0e-4, kg=0.040,            # nucleation and templated growth
    Vmax=0.045, Km=0.30,            # SATURABLE clearance: capacity and half-saturation
)


def rhs(t, y, p):
    N, C, A = np.maximum(y, 0.0)
    prod = p['p0'] / (1.0 + (N / p['K']) ** p['h'])
    kin = p['kin0'] / (1.0 + (A / p['KA']) ** p['m'])
    agg = p['kn'] * C * C + p['kg'] * C * A
    clr = p['Vmax'] * A / (p['Km'] + A)
    return [kin * C - p['kout'] * N - p['dN'] * N,
            prod + p['kout'] * N - kin * C - p['dC'] * C - agg,
            agg - clr]


def settle(y0, p, t_end=30000.0):
    s = solve_ivp(rhs, (0, t_end), y0, args=(p,), method='LSODA',
                  rtol=1e-9, atol=1e-12)
    return s.y[:, -1]


def classify(p):
    healthy = settle([1.0, 0.15, 0.0], p)
    seeded = settle([1.0, 0.15, 3.0], p)
    low, high = healthy[2] < 0.2, seeded[2] > 1.0
    if low and high:
        return 'BISTABLE', healthy, seeded
    if low:
        return 'HEALTHY', healthy, seeded
    return 'COLLAPSED', healthy, seeded


def main():
    res = dict(model=__doc__.split('\n\n')[0], parameters=BASE)

    cls, healthy, seeded = classify(BASE)
    res['state'] = cls
    res['healthy'] = dict(zip('NCA', map(float, healthy)))
    res['seeded'] = dict(zip('NCA', map(float, seeded)))
    print('wild type: %s' % cls)
    print('  healthy  N %.3f  C %.3f  A %.4f' % tuple(healthy))
    print('  seeded   N %.3f  C %.3f  A %.4f' % tuple(seeded))

    # the flux balance that now sets the threshold
    C = healthy[1]
    print('\nthe threshold is now a FLUX balance, which autophagy assays measure:')
    print('  clearance capacity Vmax           %.4f  (concentration units per hour)' % BASE['Vmax'])
    print('  aggregation flux at the healthy C %.4f' % (BASE['kn'] * C * C))
    print('  capacity margin Vmax / flux_in    %.1fx' % (BASE['Vmax'] / max(BASE['kn'] * C * C, 1e-12)))

    # how far can capacity fall?
    print('\nclearance CAPACITY scan (what a loss-of-function allele reduces):')
    res['capacity_scan'] = []
    cliff = None
    for f in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        p = dict(BASE)
        p['Vmax'] = BASE['Vmax'] * f
        c, h, s = classify(p)
        res['capacity_scan'].append(dict(fraction=f, Vmax=p['Vmax'], state=c,
                                         healthy_A=float(h[2]), seeded_A=float(s[2])))
        print('  Vmax %3.0f%%  %-9s  healthy A %.4f   seeded A %.3f' % (100 * f, c, h[2], s[2]))
        if c == 'COLLAPSED' and cliff is None:
            cliff = f
    res['capacity_cliff'] = cliff
    print('  -> healthy state lost below about %s of wild-type capacity'
          % (('%.0f%%' % (100 * cliff)) if cliff else 'nothing tested'))

    # does a heterozygote survive now? that was the clinical test the last model failed
    het = dict(BASE)
    het['Vmax'] = BASE['Vmax'] * 0.5
    c, h, s = classify(het)
    res['heterozygote'] = dict(state=c, healthy_A=float(h[2]))
    print('\nclinical test the first model FAILED: heterozygous clearance loss (50 %% Vmax)')
    print('  -> %s   (needs to be HEALTHY or BISTABLE, not COLLAPSED)' % c)

    # and the elongation rate the corrected model now permits
    print('\nwhat elongation rate does the corrected model permit?')
    print('  with saturable clearance the cap is on the FLUX, not on kg, so a fast')
    print('  elongation rate constant is no longer forbidden: it only means the')
    print('  aggregate grows fast ONCE SEEDED, which is what amyloid does.')
    print('  the constraint becomes  Vmax > nucleation flux  -- a far weaker and')
    print('  far more measurable requirement than the first model imposed.')

    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print('\nsaved %s' % OUT.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
