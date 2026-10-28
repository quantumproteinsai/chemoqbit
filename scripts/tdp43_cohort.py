#!/usr/bin/env python
"""A stochastic cohort: does the threshold model reproduce the mouse dose series?

The published SOD1-G93A copy-number series is a dose-response experiment that was
run for another purpose and happens to test exactly what a threshold model
predicts:

    ~25 copies   death at about 130 days, fully penetrant
    8-10 copies  paralysis at 24-34 weeks, fully penetrant
    ~4 copies    about 20 % of animals still symptom-free past 24 months

That last line is the interesting one. Between 4 and 8 copies -- a factor of two
in dose -- penetrance goes from partial to complete. A graded mechanism shifts
onset age smoothly with dose and does not do that. A threshold does: near it,
whether an individual crosses is decided by chance, so penetrance is partial and
onset is scattered; past it, everyone crosses and onset tightens.

Deterministic equations cannot show this at all -- they give one trajectory per
dose. Incomplete penetrance IS the stochastic part, so this simulates a cohort.

Nucleation is the stochastic step: aggregates appear by a rare fluctuation, and
the chemical Langevin form puts noise on the aggregate equation scaled by the
square root of the flux through it. Everything else is the saturable-clearance
model unchanged.

Dose is mapped to PRODUCTION, because that is what transgene copy number does: it
sets how much aggregation-prone protein the cell makes. A first attempt mapped it
to the nucleation constant instead, and nothing crossed at any dose up to 40x --
correctly, because multiplying nucleation raises the flux but barely moves the
cytoplasmic pool, and the pool is what drives templated growth. Raising production
raises the pool, so nucleation goes as the square of dose and templating goes
linearly, which is the dependence a copy-number series actually has.
The SOD1 mouse is also not a TDP-43 model, so what is being tested here is the
SHAPE -- a step in penetrance against a factor of two in dose -- and not the
numbers.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from tdp43_saturable import BASE          # noqa: E402

OUT = ROOT / 'results' / 'tdp43_cohort.json'

HOURS_PER_MONTH = 730.0
LIFESPAN_MONTHS = 28.0                    # a laboratory mouse, generously
PATHOLOGICAL = 3.0                        # aggregate load called disease onset
N_ANIMALS = 400
DT = 0.5                                  # hours, on the aggregate only
NOISE = 0.30                              # Langevin scale on the nucleation step


def fast_steady(A, p):
    """Nuclear and cytoplasmic TDP-43, at quasi-steady state for a given aggregate load.

    Transport and turnover run on hours; aggregation runs on months. Integrating
    them together with one timestep is what broke the first version of this: rate
    constants of 6 per hour against a 2 h step made Euler unstable and everything
    overflowed. Separating the timescales is both faster and correct -- the fast
    variables are always equilibrated to whatever A currently is."""
    kin = p['kin0'] / (1.0 + (A / p['KA']) ** p['m'])

    def residual(C):
        N = kin * C / (p['kout'] + p['dN'])
        prod = p['p0'] / (1.0 + (N / p['K']) ** p['h'])
        agg = p['kn'] * C * C + p['kg'] * C * A
        return prod + p['kout'] * N - kin * C - p['dC'] * C - agg

    lo, hi = 1e-9, 50.0
    if residual(lo) < 0:
        return 0.0, 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0:
            lo = mid
        else:
            hi = mid
    C = 0.5 * (lo + hi)
    return kin * C / (p['kout'] + p['dN']), C


def run_cohort(dose, n=N_ANIMALS, months=LIFESPAN_MONTHS, seed=0):
    """Euler-Maruyama on the aggregate alone, with the fast variables slaved to it.

    Returns onset time in months per animal, nan if it never crossed."""
    rng = np.random.default_rng(seed)
    p = BASE
    steps = int(months * HOURS_PER_MONTH / DT)
    A = np.zeros(n)
    onset = np.full(n, np.nan)
    sq = np.sqrt(DT)

    # dose scales production, so the fast subsystem depends on it; tabulate per dose
    pd = dict(p); pd['p0'] = p['p0'] * dose
    grid = np.concatenate([np.linspace(0, 1, 400), np.linspace(1.05, 20, 200)])
    tabC = np.array([fast_steady(a, pd)[1] for a in grid])

    for i in range(steps):
        C = np.interp(A, grid, tabC)
        nuc = p['kn'] * C * C
        agg = nuc + p['kg'] * C * A
        clr = p['Vmax'] * A / (p['Km'] + A)
        dW = rng.normal(0.0, 1.0, n) * sq
        A = np.maximum(A + (agg - clr) * DT + NOISE * np.sqrt(np.maximum(nuc, 0.0)) * dW, 0.0)
        newly = np.isnan(onset) & (A >= PATHOLOGICAL)
        if newly.any():
            onset[newly] = (i + 1) * DT / HOURS_PER_MONTH
    return onset


def main():
    res = dict(model=__doc__.split('\n\n')[0],
               settings=dict(n_animals=N_ANIMALS, lifespan_months=LIFESPAN_MONTHS,
                             pathological_load=PATHOLOGICAL, dt_hours=DT, noise=NOISE),
               observed=dict(
                   copies_25='death ~130 days (4.3 months), fully penetrant',
                   copies_8_10='paralysis 24-34 weeks (5.5-7.8 months), fully penetrant',
                   copies_4='~20 % symptom-free past 24 months'))

    print('dose   penetrance   median onset   IQR of onset')
    print('-' * 52)
    res['doses'] = []
    for dose in (1, 2, 4, 6, 8, 12, 16, 25, 40):
        onset = run_cohort(dose, seed=dose)
        hit = ~np.isnan(onset)
        pen = 100.0 * hit.mean()
        if hit.any():
            q1, med, q3 = np.nanpercentile(onset[hit], [25, 50, 75])
            print('%4d   %6.1f %%     %6.1f mo      %.1f - %.1f'
                  % (dose, pen, med, q1, q3))
        else:
            med = q1 = q3 = None
            print('%4d   %6.1f %%     %-12s %s' % (dose, pen, 'never', '-'))
        res['doses'].append(dict(dose=dose, penetrance_pct=float(pen),
                                 median_onset_months=None if med is None else float(med),
                                 iqr=None if med is None else [float(q1), float(q3)]))

    # the shape test: how fast does penetrance move with dose?
    d = {r['dose']: r['penetrance_pct'] for r in res['doses']}
    print()
    print('the shape test -- penetrance across a factor of two in dose:')
    for lo, hi in ((4, 8), (2, 4), (8, 16)):
        print('  %2d -> %2d copies:  %5.1f %% -> %5.1f %%' % (lo, hi, d[lo], d[hi]))
    res['step_4_to_8'] = dict(at_4=d[4], at_8=d[8])

    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print('\nsaved %s' % OUT.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
