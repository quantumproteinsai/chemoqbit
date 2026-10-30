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
    'prolyl_barrier': (12.0, 32.0, 'kcal/mol',
                       'prolyl cis/trans barriers are measured at 14-24 kcal/mol, and our '
                       'neutral and monoanionic models all give 16-20. The first bound here '
                       'was 8-40, loose enough to pass the dianion at 9.94 whose amide had '
                       'stopped being an amide'),
    'amide_CO': (1.19, 1.27, 'A',
                 'the carbonyl of a conjugated amide. At 1.31 the pi system is gone and the '
                 'rotation barrier it is supposed to measure no longer exists'),
    'amide_CN': (1.29, 1.39, 'A',
                 'the C-N bond of a conjugated amide; 1.45 is a single bond'),
    'conformer_match': (0, 0, 'count',
                        'hydrogen bonds present in one of a pair of minima and absent in the '
                        'other. A cis/trans energy difference is only the amide if everything '
                        'else is the same arrangement; with a different hydrogen-bond network '
                        'it measures that network instead. Four of five prolyl models failed '
                        'this, and their phosphorylation effects came out with opposite signs '
                        'for Thr and Ser'),
    'prolyl_cis_trans': (-4.0, 6.0, 'kcal/mol',
                         'trans is modestly favoured for Xaa-Pro; cis is populated at a few to 30 %'),
    'scan_vs_free_min': (0.0, 2.0, 'kcal/mol',
                        'the lowest point of a relaxed scan must agree with a free minimisation '
                        'of the same molecule: if it does not, the scan is trapped in a strained '
                        'arrangement. Forcing omega from its embedded value to 180 deg in one '
                        'step wrenched Ac-Thr-Pro-NHMe by 175 deg and left the whole scan 41 '
                        'kcal/mol above the true minimum'),
    'start_wrench': (0.0, 20.0, 'deg',
                     'the first restrained point of a scan must be near the geometry it starts '
                     'from; a large jump strains the molecule and it never recovers'),
    'torsion_error': (0.0, 2.0, 'deg',
                      'a restrained scan must follow its coordinate, or it never samples the saddle'),
    # enzyme reaction steps in this project
    'reaction_barrier': (0.0, 80.0, 'kcal/mol',
                         'a step with a barrier above this does not turn over on any timescale'),
    'profile_segments': (1, 4, 'count',
                        'a full-circle amide rotation rises to a saddle, falls to cis, rises to '
                        'the other saddle and falls back: four monotonic runs. More means a point '
                        'is out of place'),
    'scan_drop': (0.0, 1.5, 'kcal/mol',
                  'along a scan that compresses or stretches one coordinate monotonically, the '
                  'energy should not FALL: a drop means a neighbouring point was trapped in a '
                  'worse arrangement, which a backward pass fixes. A steep but smooth rise is '
                  'physics, not an artefact, so plain step size is the wrong test'),
    # geometry of the covalent ternary complex
    'bond_CS': (1.6, 2.3, 'A', 'a C-S single bond; beyond this the adduct has broken'),
    'bond_CC': (1.3, 1.8, 'A', 'a C-C single bond'),
    'bond_CH': (0.9, 1.8, 'A', 'a C-H bond, or a hydrogen in flight between donor and acceptor'),
    'proton_bonded': (0.85, 1.35, 'A',
                      'the shorter of a transferring proton\'s two distances. In flight it is '
                      'bonded to one partner or the other; if BOTH are long it has been pulled '
                      'into the cavity. Testing the donor distance alone calls a completed '
                      'transfer a fault'),
    # interaction energies of a drug in its pocket, or a zipper interface
    'interaction_energy': (-200.0, 0.0, 'kcal/mol',
                           'a bound complex must be downhill against its separated parts'),
    'alanine_ddE': (-20.0, 60.0, 'kcal/mol',
                    'removing one side chain cannot change a contact by more than this'),
    'sheet_equivalence': (0.0, 0.5, 'kcal/mol',
                          'when the mating sheet is a symmetry image of the reference sheet it '
                          'is the SAME fragment in a rotated frame, so the two fragment energies '
                          'are equal by construction and |E_sheet0 - E_sheet1| is zero. Anything '
                          'larger is a per-sheet modelling difference -- an arbitrary hydrogen '
                          'rotamer, a stalled SCF. A static one cancels in E(both) - E(sheet0) '
                          '- E(sheet1), since the same coordinates are sliced for all three, so '
                          'this is a detector and not the error: it says the per-structure '
                          'modelling is arbitrary, and what then fails to cancel is which local '
                          'minimum each structure relaxed into. The interaction energy is a '
                          'difference of numbers two hundred times its own size, so that passes '
                          'into it whole'),
    # the quantities the two manuscripts rest on
    'kie': (0.5, 30.0, '-', 'a primary H/T effect of a hydride transfer; outside this the '
                            'partition functions or the mass assignment are wrong'),
    'kappa_tunnelling': (1.0, 10.0, '-', 'a tunnelling correction raises a rate, never lowers it'),
    'imag_freq_ts': (1, 1, 'count', 'a transition state has exactly one imaginary mode'),
    'imag_freq_min': (0, 0, 'count', 'a minimum has none'),
    'imag_wavenumber': (50.0, 2500.0, 'cm-1',
                        'the magnitude of a transition mode. The lower bound was 200, set '
                        'for a hydrogen transfer; an amide rotation turns a whole ring and '
                        'sits far lower, so 200 would have rejected real transition states. '
                        'Below 50 it is a shoulder, not a saddle'),
    'zpe_change': (-15.0, 5.0, 'kcal/mol', 'the zero-point change over one bond-forming step'),
    'entropy_term': (-10.0, 10.0, 'kcal/mol',
                     '-T dS for a unimolecular step inside a cluster with fixed atoms'),
    'n_imaginary': (1, 1, 'count',
                    'a transition state has exactly one imaginary mode. Zero means the '
                    'search fell back to a minimum, two or more means it found a '
                    'higher-order saddle, and either way the barrier is not a barrier'),
    'chalcogen_attack_angle': (150.0, 180.0, 'deg',
                               'an S_N2 at sulfur or selenium proceeds by backside attack: the '
                               'nucleophile, the atom attacked and the leaving atom must be '
                               'close to collinear. Pulled in sideways the restraint still '
                               'reaches its target xi, but what it reaches is a compressed '
                               'contact pair and not a displacement, which is how the summed '
                               'relay coordinate produced a saddle with a C5-H bond that had '
                               'never moved'),
    'chalcogen_barrier': (2.0, 30.0, 'kcal/mol',
                          'thiolate-disulfide exchange in water runs on a timescale of '
                          'milliseconds to seconds at millimolar thiol, so a barrier much '
                          'above 30 contradicts the measured second-order rates, and one '
                          'below 2 would mean no barrier at all, which contradicts their '
                          'being second order in the first place'),
    'scan_hysteresis': (0.0, 3.0, 'kcal/mol',
                        'the largest disagreement between a forward and a backward relaxed '
                        'scan at the same xi. They should agree: a gap means one direction is '
                        'trapped in an arrangement the other escapes. A single-direction '
                        'sweep left the trans end of a prolyl scan 30 kcal/mol high and the '
                        'error was invisible until the reverse pass was run'),
    'exchange_segments': (2, 2, 'count',
                          'a displacement profile rises to one saddle and falls to the '
                          'product: exactly two monotonic runs. The amide bound of 1 to 4 is '
                          'right for a full rotation and too loose here, and a loose bound is '
                          'what let a broken-amide dianion through at 9.94'),
    'approach_tracking': (0.0, 0.15, 'A',
                          'how far a restrained distance ends from its target. A scan that '
                          'does not follow its coordinate never samples what it claims to: a '
                          'soft torsion restraint drifted 21 deg off target in the prolyl work '
                          'and the barriers came out at 80-120 kcal/mol'),
    'covalent_stretch': (0.0, 1.25, 'ratio',
                         'the longest covalent bond of a restrained geometry divided by its '
                         'length in the relaxed reference. A cluster is allowed to strain and '
                         'not to come apart, and a restrained scan that cannot reach its '
                         'target any other way will tear a bond to get there: holding the '
                         'SOD1 backbone frozen and the Cys57-Cys146 distance fixed drove the '
                         'Cys57 CB-SG bond from 1.84 to 5.45 A, which is a sulfur leaving its '
                         'own beta carbon and not a conformational change. The cost of that '
                         'run was that conformational_work caught it only at the endpoint, by '
                         'the accident of 208 exceeding 200; checked per point it stops at the '
                         'rupture. The legitimate ceiling is low: over the same approach with '
                         'the backbone free, the worst stretch is 1.107, and that is the '
                         'attacked disulfide lengthening from 2.02 to 2.24 A as the '
                         'nucleophile closes'),
    'conformational_work': (0.0, 200.0, 'kcal/mol',
                            'the work to compress two buried side chains together against the '
                            'packing of the fold. It must be positive, because the crystal '
                            'structure is a minimum and squeezing it costs energy; a negative '
                            'value means the reference was strained by the truncation rather '
                            'than relaxed. Above 200 the cluster has broken rather than '
                            'strained, and what is being measured is a bond snapping'),
    'identity_asymmetry': (-1.0, 1.0, 'kcal/mol',
                           'reactants and products of an IDENTITY exchange are the same two '
                           'species, so its reaction energy is exactly zero by symmetry. Any '
                           'departure is scan error and nothing else, which makes this the one '
                           'check here with no physics in its tolerance: the identity systems '
                           'give the numerical noise floor, and a barrier difference smaller '
                           'than it cannot be claimed for se_s or se_nuc. Read it from ss '
                           '(0.021 kcal/mol on a smooth reversible scan), NOT from sese, whose '
                           'scan leaves the exchange path: its 1.10 is the gap between two ends '
                           'taken from different sweeps, and a failed scan is not a measurement '
                           'of noise'),
    'exchange_well': (0.0, 2.0, 'kcal/mol',
                      'how far the symmetric arrangement of a displacement sits BELOW the '
                      'reactant complex. For an S_N2 it should be about zero, because that '
                      'arrangement IS the transition state. GFN2-xTB puts it 8 kcal/mol lower '
                      'for thiolate plus disulfide, making the hypervalent trisulfur anion a '
                      'bound species and the whole surface a well instead of a barrier. That '
                      'is the same over-stabilisation of a thiolate the second manuscript '
                      'found, and here it changes the topology rather than just the number, so '
                      'no saddle search seeded from this surface can succeed'),
    'chalcogen_bond': (1.95, 2.45, 'A',
                       'a formed S-S bond is 2.03-2.08, Se-S about 2.19 and Se-Se about 2.33. '
                       'Outside this the species at the end of the scan is not the '
                       'dichalcogenide the reaction is supposed to make'),
    'residue_sasa': (0.0, 300.0, 'A^2',
                     'solvent-accessible area of one amino acid residue by Shrake-Rupley. '
                     'It cannot be negative, and the largest residue fully exposed in a '
                     'Gly-X-Gly tripeptide reaches about 250. A value above 300 means the '
                     'atom selection is wrong -- a residue number colliding between chains, '
                     'or alternate conformers summed together rather than one being chosen'),
    'circuit_error_mha': (0.0, 1.6, 'mHa',
                          'chemical accuracy against exact CASCI in the same active space'),
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


def monotonic_segments(values, tol=0.05):
    """Number of monotonic runs in an ordered series, ignoring wobbles below `tol`.
    A full-circle amide rotation profile has four: up to one saddle, down to cis,
    up to the other saddle, down to trans. More than that means a point is out of
    place. Testing the 'rising branch up to the global maximum' instead, as a
    first version did, counts the legitimate descent into cis as an artefact."""
    d = [b - a for a, b in zip(values, values[1:]) if abs(b - a) > tol]
    if not d:
        return 0
    return 1 + sum(1 for a, b in zip(d, d[1:]) if (a > 0) != (b > 0))


def segment_drop(values, tol=0.05):
    """Largest reversal inside a monotonic run: a genuine discontinuity, as
    opposed to the turns of the profile itself."""
    worst = 0.0
    run = [values[0]]
    for a, b in zip(values, values[1:]):
        if abs(b - a) <= tol:
            continue
        rising = b > a
        if len(run) > 1 and rising != (run[-1] > run[-2]):
            run = [a]
        run.append(b)
        if len(run) > 2:
            seq = run if run[-1] > run[0] else [-v for v in run]
            worst = max(worst, max([0.0] + [x - y for x, y in zip(seq, seq[1:])]))
    return worst


def largest_drop(values):
    """Largest fall in energy along an ordered scan, which should be rising (or
    flat) if every point is properly relaxed. Distinguishes a conformational
    drop from a steep smooth wall, which plain step size does not: the tyr94
    approach rises 0.9 -> 22.0 kcal/mol per step with no drop at all and is
    sound, while wat327 falls 5.3 kcal/mol at one point and is not."""
    if len(values) < 2:
        return 0.0
    return max([0.0] + [a - b for a, b in zip(values, values[1:])])
