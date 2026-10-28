#!/usr/bin/env python
"""Does losing protein clearance put a cell over the TDP-43 threshold?

The bistability model threw up a result that was not built into it: aggregate
clearance is the dominant lever on the threshold, and cutting it by 40 % abolishes
the healthy state entirely. That lands on a gene class ALS keeps implicating from
the other direction -- OPTN, TBK1, SQSTM1, UBQLN2 and VCP are all autophagy and
protein-clearance genes, and between them they account for a slice of familial
ALS that has never had a mechanism tying it to TDP-43.

This asks the question properly, as a two-parameter phase diagram over clearance
capacity and aggregation propensity, because those are the two axes the ALS
genetics actually moves:

    clearance  dA     lowered by OPTN, TBK1, SQSTM1, UBQLN2, VCP loss of function,
                      and by ageing, which is the one risk factor every patient has
    templating kg     raised by TARDBP and FUS mutations, and by anything that
                      makes the protein more aggregation-prone

Each point in that plane is one of three things, and which one is the whole
question:

    HEALTHY     only the healthy state exists; insults decay, whatever their size
    BISTABLE    both states exist; a sufficient insult is permanent
    COLLAPSED   only the collapsed state exists; the cell cannot stay healthy at all

The mapping from genes to parameters is a HYPOTHESIS, not a measurement. A
heterozygous loss-of-function allele is modelled as roughly half the clearance
capacity, which is the usual first assumption for haploinsufficiency and may be
wrong in either direction. What the diagram shows is the SHAPE of the dependence:
whether the healthy region has a cliff edge near the wild type, and how far a
realistic genetic insult moves a cell towards it.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from tdp43_bistability import BASE, settle, threshold, rhs   # noqa: E402
from scipy.integrate import solve_ivp                        # noqa: E402

OUT = ROOT / 'results' / 'tdp43_clearance.json'


def classify(p, tol=1e-3):
    """HEALTHY, BISTABLE or COLLAPSED for one parameter set."""
    healthy = settle([1.0, 0.15, 0.0], p)
    seeded = settle([1.0, 0.15, 3.0], p)
    high = seeded[2] > 1.0
    low = healthy[2] < 0.1
    if low and high:
        return 'BISTABLE'
    if low and not high:
        return 'HEALTHY'
    return 'COLLAPSED'


def push_to_collapse(p, max_hours=400, boost=14.0):
    """Hours of sustained export bias needed to make the collapse permanent.

    This is the quantity an experiment could measure, and the one a genotype
    should change: a cell closer to the cliff needs a shorter insult."""
    if classify(p) != 'BISTABLE':
        return None
    pp = dict(p)
    pp['kout'] = p['kout'] * boost
    lo, hi = 0.0, float(max_hours)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        y = solve_ivp(rhs, (0, mid), [1.0, 0.15, 0.0], args=(pp,),
                      method='LSODA', rtol=1e-9, atol=1e-12).y[:, -1]
        if settle(y, p)[2] > 1.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi) if hi < max_hours - 1e-6 else None


def main():
    res = dict(model=__doc__.split('\n\n')[0], base=BASE)

    # ---- 1. how far can clearance fall before the healthy state goes? -------
    print('clearance alone, everything else at wild type:')
    res['clearance_scan'] = []
    cliff = None
    for f in (1.00, 0.90, 0.80, 0.70, 0.65, 0.60, 0.55, 0.50, 0.40):
        p = dict(BASE)
        p['dA'] = BASE['dA'] * f
        cls = classify(p)
        thr = threshold(p) if cls == 'BISTABLE' else None
        hrs = push_to_collapse(p)
        res['clearance_scan'].append(dict(fraction=f, dA=p['dA'], state=cls,
                                          threshold=thr, push_hours=hrs))
        if cls == 'COLLAPSED' and cliff is None:
            cliff = f
        print('  clearance %3.0f%%  %-9s  A* %-7s  push to collapse %s'
              % (100 * f, cls, ('%.3f' % thr) if thr else '-',
                 ('%.0f h' % hrs) if hrs else '-'))
    res['clearance_cliff'] = cliff
    if cliff:
        print('  -> the healthy state disappears at about %.0f%% of wild-type clearance'
              % (100 * cliff))

    # ---- 2. the phase diagram over clearance x templating -------------------
    print('\nphase diagram (rows: clearance, cols: templating; both as fraction of WT)')
    cl = [1.3, 1.15, 1.0, 0.85, 0.7, 0.55, 0.4]
    kg = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    grid = []
    print('              ' + ''.join('%7.2f' % k for k in kg))
    for c in cl:
        row = []
        for k in kg:
            p = dict(BASE)
            p['dA'] = BASE['dA'] * c
            p['kg'] = BASE['kg'] * k
            row.append(classify(p))
        grid.append(dict(clearance=c, states=row))
        sym = {'HEALTHY': '   .   ', 'BISTABLE': '   B   ', 'COLLAPSED': '   X   '}
        print('  clear %4.2f ' % c + ''.join(sym[s] for s in row))
    res['phase_diagram'] = dict(clearance_factors=cl, templating_factors=kg, grid=grid)
    print('   . healthy only      B bistable      X collapsed only')

    # ---- 3. genotypes, as a hypothesis --------------------------------------
    print('\ngenotypes, mapping genes to parameters as a hypothesis:')
    GENO = [
        ('wild type', 1.00, 1.00),
        ('heterozygous clearance LoF (OPTN/TBK1/SQSTM1/UBQLN2/VCP)', 0.50, 1.00),
        ('mild clearance loss, e.g. ageing', 0.80, 1.00),
        ('aggregation-prone TARDBP/FUS variant', 1.00, 1.25),
        ('ageing plus an aggregation-prone variant', 0.80, 1.25),
        ('heterozygous clearance LoF plus ageing', 0.40, 1.00),
    ]
    res['genotypes'] = []
    for name, c, k in GENO:
        p = dict(BASE)
        p['dA'] = BASE['dA'] * c
        p['kg'] = BASE['kg'] * k
        cls = classify(p)
        thr = threshold(p) if cls == 'BISTABLE' else None
        hrs = push_to_collapse(p)
        res['genotypes'].append(dict(name=name, clearance=c, templating=k, state=cls,
                                     threshold=thr, push_hours=hrs))
        print('  %-56s %-9s push %s'
              % (name[:56], cls, ('%.0f h' % hrs) if hrs else ('-' if cls != 'BISTABLE' else '>400 h')))

    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print('\nsaved %s' % OUT.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
