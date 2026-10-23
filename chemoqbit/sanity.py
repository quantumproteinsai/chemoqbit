"""Physical bounds checked at the moment a result is produced, not when someone
thinks to look.

Every quantity this project computes has a range outside which the number is
not a finding but a bug. Those ranges are written down here, with the reason,
and the drivers call them as results appear. A violation is printed loudly,
recorded in the checkpoint under '_sanity', and -- for the checks marked
fatal -- stops the stage rather than letting later stages build on it.

The prolyl scan is why this exists: a torsion restraint too soft to hold the
saddle gave amide rotation barriers of 80-120 kcal/mol and a cis amide below
trans, and the whole 7-model scan plus its DFT stage ran to completion on
those geometries before anyone compared them with experiment.
"""
from __future__ import annotations


class Violation(Exception):
    pass


# name -> (low, high, unit, why these bounds)
BOUNDS = {
    # prolyl amide rotation: experiments put prolyl cis/trans barriers at
    # 14-24 kcal/mol, and PPIases lower them by about 8
    'prolyl_barrier': (8.0, 40.0, 'kcal/mol',
                       'prolyl cis/trans barriers are measured at 14-24 kcal/mol'),
    'prolyl_cis_trans': (-4.0, 6.0, 'kcal/mol',
                         'trans is modestly favoured for Xaa-Pro; cis is populated at a few to 30 %'),
    'torsion_error': (0.0, 2.0, 'deg',
                      'a restrained scan must follow its coordinate, or it never samples the saddle'),
    # enzyme reaction steps in this project
    'reaction_barrier': (0.0, 80.0, 'kcal/mol',
                         'a step with a barrier above this does not turn over on any timescale'),
    'scan_drop': (0.0, 1.5, 'kcal/mol',
                  'along a scan that compresses or stretches one coordinate monotonically, the '
                  'energy should not FALL: a drop means a neighbouring point was trapped in a '
                  'worse arrangement, which a backward pass fixes. A steep but smooth rise is '
                  'physics, not an artefact, so plain step size is the wrong test'),
    # geometry of the covalent ternary complex
    'bond_CS': (1.6, 2.3, 'A', 'a C-S single bond; beyond this the adduct has broken'),
    'bond_CC': (1.3, 1.8, 'A', 'a C-C single bond'),
    'bond_CH': (0.9, 1.8, 'A', 'a C-H bond, or a hydrogen in flight between donor and acceptor'),
    # interaction energies of a drug in its pocket, or a zipper interface
    'interaction_energy': (-200.0, 0.0, 'kcal/mol',
                           'a bound complex must be downhill against its separated parts'),
    'alanine_ddE': (-20.0, 60.0, 'kcal/mol',
                    'removing one side chain cannot change a contact by more than this'),
}


def check(name, value, label='', fatal=False, log=print):
    """True if `value` is inside the bounds for `name`. Prints and optionally raises."""
    lo, hi, unit, why = BOUNDS[name]
    if value is None:
        return True
    ok = lo <= value <= hi
    if not ok:
        msg = (f'SANITY {name} out of range: {value:.3f} {unit} not in [{lo}, {hi}]'
               + (f' for {label}' if label else '') + f' -- {why}')
        log(msg)
        if fatal:
            raise Violation(msg)
    return ok


def record(res, name, value, label='', fatal=False, log=print):
    """check(), and keep the violation in the checkpoint under '_sanity'."""
    ok = check(name, value, label=label, fatal=fatal, log=log)
    if not ok:
        lo, hi, unit, why = BOUNDS[name]
        res.setdefault('_sanity', []).append(
            dict(check=name, value=value, label=label, low=lo, high=hi, unit=unit, why=why))
    return ok


def profile_steps(profile):
    """Largest jump between neighbouring points of a scan profile {coord: energy}."""
    ks = sorted(profile, key=float)
    if len(ks) < 2:
        return 0.0
    return max(abs(profile[b] - profile[a]) for a, b in zip(ks, ks[1:]))


def largest_drop(values):
    """Largest fall in energy along an ordered scan, which should be rising (or
    flat) if every point is properly relaxed. Distinguishes a conformational
    drop from a steep smooth wall, which plain step size does not: the tyr94
    approach rises 0.9 -> 22.0 kcal/mol per step with no drop at all and is
    sound, while wat327 falls 5.3 kcal/mol at one point and is not."""
    if len(values) < 2:
        return 0.0
    return max([0.0] + [a - b for a, b in zip(values, values[1:])])
