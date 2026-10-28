#!/usr/bin/env python
"""What does a unit of net charge actually buy, once percentages become rates?

Choi et al. report a linear relationship between the net charge of the
polyampholyte motif and the percentage of cells bearing amyloid fibrils, and
read it as charge setting the RATE of conversion rather than a fixed endpoint.
Those two readings cannot both be linear. A percentage follows 1 - exp(-lambda.t)
and saturates, so if the percentage is linear in charge the rate is not, and a
constant increment of charge buys a shrinking multiple of the rate.

This converts their percentages to rates and reports the multiple per charge
step, which is the form the threshold argument needs. It exists because section
3.4 of the manuscript quoted these numbers without a script behind them, and a
paper that says its numbers were checked against deposited data has to deposit
them.

Two conventions are reported rather than one, because the first step depends on
which is used. Binning the observed percentages is the honest reading of the
data; the authors' own fitted line is the honest reading of their model, and it
does not pass through the wild-type point. The ORDERING is the claim, and it
survives either: the +2 -> 0 step is the steepest by a factor of two or more.

The grouping is by net charge alone and takes no account of which residue
carries it, which is the authors' own finding and a real limitation of reading
their series this way.
"""
import json
import sys
from math import exp, log
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'tdp43_charge_steps.json'

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdp43_frailty import FIT48, T          # noqa: E402  one copy of the data, not two

# The even net-charge ladder. K263A (+1), 3R-A (-1) and 2E-K (+6) sit off it and
# are excluded, so this is 15 of the 24 constructs Choi et al. scored -- not all
# of them, which is what section 3.4 used to claim.
LADDER = (2, 0, -2, -4, -6)

# The authors' least-squares fit of percentage against charge at 48 h, Figure 7F:
# Y = -2.413 X + 6.852, R2 = 0.85 over all 24 constructs. It crosses zero at
# +2.8 rather than at the wild-type charge of +2, where it predicts 2.0 % against
# the 0.6 % observed. That is why it gives a shallower first step.
FIT_SLOPE, FIT_INTERCEPT = -2.413, 6.852


def lam(pct, t=T):
    """Percentage converted -> first-order rate constant, per hour."""
    assert 0.0 < pct < 100.0, 'a percentage outside (0, 100) has no rate: %r' % pct
    return -log(1.0 - pct / 100.0) / t


def steps(values):
    return [b / a for a, b in zip(values, values[1:])]


def main():
    res = {
        'question': 'What multiple of the conversion rate does each step of motif net '
                    'charge buy, once the published percentages are converted to rates?',
        'source': 'Choi, Kang, Cho, Jeong, Jung, Pack, Jeong & Song, bioRxiv '
                  '2026.10.15.759291 (NOT peer reviewed). Percentages read from the text '
                  'of that preprint, a single experiment with pooled proportions and no '
                  'replicate statistics.',
        'why_rates': 'Percentage saturates as 1 - exp(-lambda.t); the rate does not. A '
                     'relationship that is linear in one cannot be linear in the other.',
    }

    # ---- A. observed percentages, binned by net charge ----------------------
    bins, means = {}, []
    for q in LADDER:
        members = {k: v[1] for k, v in FIT48.items() if v[0] == q}
        assert members, 'no construct at charge %+d' % q
        m = sum(lam(p) for p in members.values()) / len(members)
        means.append(m)
        bins['%+d' % q] = {'n': len(members), 'constructs': members,
                           'mean_lambda_per_hour': m}
    obs_steps = steps(means)

    # ---- B. the authors' fitted line ----------------------------------------
    fitted = [lam(FIT_SLOPE * q + FIT_INTERCEPT) for q in LADDER]
    fit_steps = steps(fitted)

    res['observed_bins'] = bins
    res['steps'] = [
        {'step': '%+d -> %+d' % (a, b), 'constructs': '%d -> %d' % (bins['%+d' % a]['n'],
                                                                   bins['%+d' % b]['n']),
         'rate_increase_observed': round(o, 2), 'rate_increase_fitted_line': round(f, 2)}
        for a, b, o, f in zip(LADDER, LADDER[1:], obs_steps, fit_steps)]

    steepest = max(range(len(obs_steps)), key=lambda i: obs_steps[i])
    runner_up = sorted(obs_steps, reverse=True)[1]
    res['ordering'] = {
        'steepest_step': res['steps'][steepest]['step'],
        'margin_over_next_observed': round(obs_steps[steepest] / runner_up, 2),
        'margin_over_next_fitted': round(max(fit_steps) / sorted(fit_steps, reverse=True)[1], 2),
        'claim': 'The +2 -> 0 step is the steepest under BOTH conventions, by a factor of '
                 'two or more over the next. The ordering is the claim; the ratio for the '
                 'first step is convention-dependent because the wild-type percentage is '
                 'small and the series is sparse there.'}

    # ---- C. the biological claim: phosphorylation sits on the steep segment --
    wt = FIT48['WT'][1]
    ala = FIT48['S266A/S273A'][1]
    res['phosphomimetic'] = {
        'why': 'Phosphorylating the two motif serines moves the charge from +2 to 0, the '
               'steepest segment. The alanine double substitution is the controlled '
               'comparison: same two positions, same substitution burden, no charge.',
        'S266D/S273D_over_WT': round(lam(FIT48['S266D/S273D'][1]) / lam(wt), 2),
        'S266E/S273E_over_WT': round(lam(FIT48['S266E/S273E'][1]) / lam(wt), 2),
        'S266D/S273D_over_alanine_control': round(lam(FIT48['S266D/S273D'][1]) / lam(ala), 2),
        'alanine_control_over_WT': round(lam(ala) / lam(wt), 2),
        'caveat': 'The alanine control is not quite inert. The authors score both it and '
                  'wild type as flat, and neither rose between 48 and 72 h.'}

    # ---- D. re-verification of the constant hazard quoted in the same section
    RE3 = [(12, 5.4), (24, 11.3), (48, 19.5)]
    lams = [lam(p, t) for t, p in RE3]
    mean = sum(lams) / len(lams)
    sd = (sum((x - mean) ** 2 for x in lams) / len(lams)) ** 0.5
    res['constant_hazard_3RE'] = {
        'points_hours_percent': RE3,
        'lambda_per_hour': [round(x, 5) for x in lams],
        'mean_lambda_per_hour': round(mean, 5),
        # Three different numbers have been called "the spread" of these three
        # estimates, and they differ by more than a factor of two. All three are
        # reported so that a sentence quoting one cannot be read as another.
        'spread_cv_percent': round(100 * sd / mean, 1),
        'spread_range_over_mean_percent': round(100 * (max(lams) - min(lams)) / mean, 1),
        'spread_max_deviation_percent': round(100 * max(abs(x - mean) for x in lams) / mean, 1),
        'predicted_from_mean': [round(100 * (1 - exp(-mean * t)), 1) for t, _ in RE3],
        'observed': [p for _, p in RE3]}

    OUT.write_text(json.dumps(res, indent=1) + '\n')

    print('charge step        constructs   observed   fitted line')
    for s in res['steps']:
        print('  %-14s %-12s x%-9.2f x%.2f'
              % (s['step'], s['constructs'], s['rate_increase_observed'],
                 s['rate_increase_fitted_line']))
    print()
    p = res['phosphomimetic']
    print('  S266D/S273D over wild type        x%.2f' % p['S266D/S273D_over_WT'])
    print('  S266D/S273D over alanine control  x%.2f' % p['S266D/S273D_over_alanine_control'])
    print('  alanine control over wild type    x%.2f' % p['alanine_control_over_WT'])
    print()
    c = res['constant_hazard_3RE']
    print('  3R-E hazard %s /h over a four-fold range of time' % c['lambda_per_hour'])
    print('    spread: CV %.1f %%, range %.1f %% of the mean, worst point %.1f %% off'
          % (c['spread_cv_percent'], c['spread_range_over_mean_percent'],
             c['spread_max_deviation_percent']))
    print('\nsaved %s' % OUT.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
