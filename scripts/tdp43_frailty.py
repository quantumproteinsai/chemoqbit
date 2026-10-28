#!/usr/bin/env python
"""Where does the ceiling come from: a hard subpopulation, or spread in margin?

The compartment model says cells differ in MARGIN -- how far each sits from its
threshold -- but a deterministic model cannot say what fraction of a population
crosses, nor how wide the spread is. Fitting a ceiling p to the Choi et al.
charge series gave 22.8 % and R2 = 0.957, but it underpredicted every held-out
72 h point. A hard ceiling saturates; the data keeps climbing. So the ceiling is
probably not hard.

Three structures, fitted to the same 18 constructs at 48 h and judged on points
they never saw:

  A  homogeneous      f = 1 - exp(-nu.t)
                      every cell identical. Constant hazard forever.

  B  hard ceiling     f = p.(1 - exp(-nu.t))
                      a fraction p can cross, the rest never can.

  C  gamma frailty    f = 1 - (1 + nu.t/k)^(-k)
                      EVERY cell can cross, but at its own rate nu.Z with
                      Z ~ Gamma(k, 1/k), mean 1. Fast cells go first, so the
                      population hazard FALLS as k/(k + nu.t) without any
                      ceiling. k is the whole point: the spread of the margin
                      distribution is CV = 1/sqrt(k), which is the quantity the
                      compartment model implies and cannot measure.

In all three nu(q) = nu0.exp(-b.q), a barrier linear in motif net charge.
A and C have the same parameter count as B minus and plus one respectively, so
they are compared by held-out error and by AIC, not by R2.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'tdp43_frailty.json'

# (net PAM charge, percent fibril-bearing cells at 48 h). Read from the prose of
# Choi et al., bioRxiv 2026.10.15.759291, which is not peer reviewed and reports
# a single experiment with pooled proportions and no replicate statistics.
FIT48 = {
    'WT': (+2, 0.6), 'S266A/S273A': (+2, 0.9), '2E-K': (+6, 1.3), 'K263A': (+1, 3.0),
    'R268E': (0, 4.8), 'R272E': (0, 8.5), 'R275E': (0, 5.8),
    'S266D/S273D': (0, 5.5), 'S266E/S273E': (0, 5.8), '3R-A': (-1, 10.2),
    'R268E/R272E': (-2, 13.7), 'R272E/R275E': (-2, 11.6), 'R268E/R275E': (-2, 8.6),
    '3R-E': (-4, 19.5), '3R-E_S266A/S273A': (-4, 19.2),
    'K263E/3R-E': (-6, 19.8), '3R-E_S266D/S273D': (-6, 24.6),
    '3R-E_S266E/S273E': (-6, 23.5),
}
# held out entirely from every fit: other time points for constructs in FIT48
HELD = [('S266D/S273D', 0, 72, 12.1), ('S266E/S273E', 0, 72, 11.1),
        ('3R-A', -1, 72, 13.6), ('3R-E', -4, 12, 5.4), ('3R-E', -4, 24, 11.3)]

T = 48.0


def nu(q, nu0, b):
    return nu0 * np.exp(-b * q)


def f_homog(q, t, nu0, b):
    return 100.0 * (1.0 - np.exp(-nu(q, nu0, b) * t))


def f_ceiling(q, t, nu0, b, p):
    return 100.0 * p * (1.0 - np.exp(-nu(q, nu0, b) * t))


def f_frailty(q, t, nu0, b, k):
    """Gamma-distributed rate across cells: S(t) = (1 + nu.t/k)^(-k)."""
    return 100.0 * (1.0 - (1.0 + nu(q, nu0, b) * t / k) ** (-k))


MODELS = {
    'A homogeneous': (f_homog, [0.001, 0.4], ([1e-7, 0.01], [1.0, 3.0])),
    'B hard ceiling': (f_ceiling, [0.001, 0.4, 0.3], ([1e-7, 0.01, 0.02], [1.0, 3.0, 1.0])),
    'C gamma frailty': (f_frailty, [0.001, 0.4, 1.0], ([1e-7, 0.01, 0.02], [1.0, 3.0, 200.0])),
}


def main():
    q = np.array([v[0] for v in FIT48.values()], float)
    y = np.array([v[1] for v in FIT48.values()], float)
    res = {'source': 'Choi et al., bioRxiv 2026.10.15.759291 (not peer reviewed); '
                     'percentages read from the text, single experiment, no replicate statistics',
           'fitted_on': '18 constructs at 48 h', 'held_out': HELD, 'models': {}}

    print('fitted on the 48 h column only; the 72 h points and the 3R-E early')
    print('time course are never seen by any fit.')
    print()
    print('%-16s %6s  %-34s %7s %9s %7s' % ('model', 'n par', 'parameters', 'R2', 'held-out', 'AIC'))
    print('-' * 90)

    best = None
    for name, (fn, p0, bnds) in MODELS.items():
        try:
            par, _ = curve_fit(lambda qq, *a: fn(qq, T, *a), q, y, p0=p0,
                               bounds=bnds, maxfev=200000)
        except Exception as e:
            print('%-16s fit failed: %s' % (name, e))
            continue
        pred = fn(q, T, *par)
        rss = ((y - pred) ** 2).sum()
        r2 = 1 - rss / ((y - y.mean()) ** 2).sum()
        n, kp = len(y), len(par)
        aic = n * np.log(rss / n) + 2 * kp

        he = [abs(obs - fn(np.array([qq], float), tt, *par)[0]) for _, qq, tt, obs in HELD]
        mae = float(np.mean(he))

        lbl = {'A homogeneous': 'nu0=%.5f b=%.3f',
               'B hard ceiling': 'nu0=%.5f b=%.3f p=%.3f',
               'C gamma frailty': 'nu0=%.5f b=%.3f k=%.2f'}[name] % tuple(par)
        print('%-16s %6d  %-34s %7.3f %9.2f %7.1f' % (name, kp, lbl, r2, mae, aic))
        res['models'][name] = dict(params=[float(x) for x in par], R2=float(r2),
                                   held_out_MAE=mae, AIC=float(aic),
                                   held_out=[float(x) for x in he])
        if best is None or mae < best[1]:
            best = (name, mae, fn, par)

    print()
    print('held-out points, one row each:')
    print('  %-20s %4s %5s %8s %s' % ('construct', 'q', 't(h)', 'observed', 'predicted by each model'))
    for k_, qq, tt, obs in HELD:
        row = []
        for name, (fn, _, _) in MODELS.items():
            if name in res['models']:
                row.append('%s %5.1f' % (name[0], fn(np.array([qq], float), tt,
                                                     *res['models'][name]['params'])[0]))
        print('  %-20s %+4d %5d %8.1f   %s' % (k_, qq, tt, obs, '   '.join(row)))

    print()
    name, mae, fn, par = best
    print('best on held-out data: %s  (MAE %.2f percentage points)' % (name, mae))
    if name.startswith('C'):
        k = par[2]
        cv = 1.0 / np.sqrt(k)
        print()
        print('  the frailty parameter is the result, not a nuisance:')
        print('    k  = %.2f' % k)
        print('    CV = 1/sqrt(k) = %.2f' % cv)
        print('  i.e. the cell-to-cell spread in crossing rate has a coefficient of')
        print('  variation of %.0f %%. That is the margin distribution the compartment' % (100 * cv))
        print('  model implies and cannot measure, now measured from population data.')
        print()
        print('  and there is no ceiling: every cell crosses eventually. The apparent')
        print('  plateau is fast cells going first and leaving slow ones behind, which')
        print('  is why a hard ceiling underpredicts every later time point.')
        res['interpretation'] = dict(
            k=float(k), CV=float(cv),
            meaning='Cell-to-cell spread in crossing rate, CV = 1/sqrt(k). This is the '
                    'margin distribution the compartment model implies and cannot measure. '
                    'No ceiling: every cell crosses eventually, and the apparent plateau is '
                    'fast cells going first.')

    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print('\nsaved %s' % OUT.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
