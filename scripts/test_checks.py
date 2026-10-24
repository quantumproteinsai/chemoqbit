"""Test the checks against data whose verdict is already known.

A check that has never been shown to fire is not a check. Three of ours flagged
sound physics before this existed -- a steep repulsive wall, the descent into a
cis minimum, and one raw sweep that the paired sweep had already repaired -- and
each was found only because a human asked for the status.

Every archived failure in results/ is a labelled positive: we know what went
wrong and which bound should catch it. Every result the manuscripts rest on is a
labelled negative: it has been audited and must stay silent. This script asserts
both directions and exits non-zero if any case is wrong, so the checker is itself
under test.

    python scripts/test_checks.py
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit import sanity

RES = ROOT / 'results'
K = 627.509474
rows = []


def record(name, kind, fired, expect, detail):
    ok = (fired == expect)
    rows.append((name, kind, fired, expect, ok, detail))
    return ok


def load(f):
    p = RES / f
    return json.loads(p.read_text()) if p.exists() else None


def xtb_scan_drop(d):
    """largest fall on the rising branch of a GFN2-xTB reaction scan"""
    if not d or 'scan' not in d:
        return None
    sc = sorted(d['scan'], key=lambda q: q['xi'])
    e0 = min(q['E'] for q in sc)
    rel = [(q['E'] - e0) * K for q in sc]
    i = int(np.argmax(rel))
    return sanity.largest_drop(rel[:i + 1])


def prolyl_profile(d, model):
    s = (d or {}).get('scan', {}).get(model)
    if not s:
        return None
    prof = {int(k): v for k, v in s['profile'].items()}
    return [prof[k] for k in sorted(prof)]


# ---------------------------------------------------------------- positives
# each archived failure must trip the bound that was added because of it

d = load('ts_cluster_run.prebackfill_20261018.json')
v = xtb_scan_drop(d)
record('xTB scan before the backward pass', 'scan_drop',
       v is not None and not sanity.check('scan_drop', v, log=lambda m: None), True,
       f'largest fall {v:.2f} kcal/mol' if v is not None else 'missing')

d = load('ts_abstraction.nobackward_20261023.json')
worst = 0.0
for nm, a in (d or {}).get('acceptor', {}).items():
    e = [x['rel'] for x in sorted(a['scan'], key=lambda x: -x['r']) if x.get('rel') is not None]
    worst = max(worst, sanity.largest_drop(e))
record('approach scans before the backward pass', 'scan_drop',
       not sanity.check('scan_drop', worst, log=lambda m: None), True,
       f'largest fall {worst:.2f} kcal/mol')

d = load('prolyl.softtorsion_20261023.json')
bad = []
for m in (d or {}).get('scan', {}):
    seq = prolyl_profile(d, m)
    if seq and not sanity.check('prolyl_barrier', max(seq) - seq[0], log=lambda m_: None):
        bad.append(m)
record('prolyl with a restraint too soft to hold the saddle', 'prolyl_barrier',
       bool(bad), True, f'barriers out of range for {len(bad)} models: {bad[:4]}')

worst_err = 0.0
for m, s in (d or {}).get('scan', {}).items():
    for p in s['points'].values():
        if p.get('omega') is not None:
            worst_err = max(worst_err, abs(((p['omega'] - p['omega_target']) + 180) % 360 - 180))
record('the same run, torsion tracking', 'torsion_error',
       not sanity.check('torsion_error', worst_err, log=lambda m_: None), True,
       f'worst deviation {worst_err:.1f} deg from target')

# That archive predates the e_free field, so take the reference from the current
# correct run of the same molecule at the same level: a valid comparison, since
# only the scan's starting geometry differed.
d = load('prolyl.wrenched_20261023.json')
cur0 = load('prolyl.json') or {}
# The live check tests (free minimum - scan minimum); either sign leaving [0, 2]
# is a fault, so look for any model that trips it rather than the largest value.
fired, which, val = False, None, None
for m, sc_ in (d or {}).get('scan', {}).items():
    ref = (cur0.get('scan', {}).get(m) or {}).get('e_free')
    if ref is None:
        continue
    gap = (ref - sc_['e_min']) * K
    if not sanity.check('scan_vs_free_min', gap, log=lambda m_: None):
        fired, which, val = True, m, gap
record('prolyl wrenched 175 deg at the scan start', 'scan_vs_free_min', fired, True,
       f'{which}: the scan sat {-val:.0f} kcal/mol above a free minimisation'
       if fired else 'no overlapping model trips the bound')

d = load('prolyl.shoulder_20261024.json')
n_bad = sum(1 for m, pts in (d or {}).get('points', {}).items()
            for tag, p in pts.items() if tag == 'saddle' and p.get('n_imaginary') == 0)
record('saddle searches that stopped on a shoulder', 'imag_freq_ts',
       n_bad > 0, True, f'{n_bad} "transition states" with no imaginary mode')

d = load('ts_cluster_run.N5adduct_20261017.json')
sc = sorted((d or {}).get('scan', []), key=lambda q: q['xi'])
if sc:
    e0 = min(q['E'] for q in sc)
    top = max((q['E'] - e0) * K for q in sc)
    record('N5-C11 broken in the bond table only', 'reaction_barrier',
           not sanity.check('reaction_barrier', top, log=lambda m_: None), True,
           f'scan reached {top:.0f} kcal/mol')

d = load('ts_cluster_run.capsmovedTS_20261018.json')
if d and 'ts' in d and 'reactant' in d:
    b = (d['ts']['E'] - d['reactant']['E']) * K
    moved = 0.0
    if 'prep' in d:
        X0 = np.array(d['prep'].get('X0') or d['reactant']['coords'])
        Xt = np.array(d['ts']['coords'])
        fz = d['prep']['frozen']
        moved = float(np.linalg.norm(Xt[fz] - X0[fz], axis=1).max())
    record('saddle search that displaced the frozen caps', 'constraint',
           moved > 0.05, True, f'frozen atoms moved up to {moved:.2f} A (barrier {b:.1f})')

# ---------------------------------------------------------------- negatives
# everything the manuscripts rest on, and the current clean runs, must stay silent

d = load('ts_cluster_run.json')
v = xtb_scan_drop(d)
record('the published xTB scan (after the backward pass)', 'scan_drop',
       not sanity.check('scan_drop', v, log=lambda m_: None), False,
       f'largest fall {v:.2f} kcal/mol')

for pair, dd in (d or {}).get('analysis', {}).get('kie', {}).items():
    for m_ in ('semiclassical', 'wigner', 'eckart'):
        record(f'published KIE {pair} {m_}', 'kie',
               not sanity.check('kie', dd[m_], log=lambda x: None), False, f"{dd[m_]:.2f}")

o = load('ts_oniom_opt.json')
if o and 'summary' in o:
    record('published ONIOM barrier', 'reaction_barrier',
           not sanity.check('reaction_barrier', o['summary']['barrier_tzvp'], log=lambda m_: None),
           False, f"{o['summary']['barrier_tzvp']:.1f} kcal/mol")

cur = load('prolyl.json')
for m in (cur or {}).get('scan', {}):
    seq = prolyl_profile(cur, m)
    if not seq:
        continue
    record(f'current prolyl {m}, merged profile', 'profile_segments',
           not sanity.check('profile_segments', sanity.monotonic_segments(seq), log=lambda x: None),
           False, f'{sanity.monotonic_segments(seq)} monotonic runs')
    record(f'current prolyl {m}, merged profile', 'scan_drop',
           not sanity.check('scan_drop', sanity.segment_drop(seq), log=lambda x: None),
           False, f'reversal {sanity.segment_drop(seq):.2f} kcal/mol')

a = load('ts_abstraction.json')
for nm, s in (a or {}).get('acceptor', {}).items():
    e = [x['rel'] for x in sorted(s['scan'], key=lambda x: -x['r']) if x.get('rel') is not None]
    record(f'current approach scan, {nm}', 'scan_drop',
           not sanity.check('scan_drop', sanity.largest_drop(e), log=lambda x: None),
           False, f'largest fall {sanity.largest_drop(e):.2f}')
# the tyr94 wall is the case that fooled the first version of this test
tyr = (a or {}).get('acceptor', {}).get('tyr94')
if tyr:
    e = [x['rel'] for x in sorted(tyr['scan'], key=lambda x: -x['r']) if x.get('rel') is not None]
    record('tyr94 steep repulsive wall (a false positive once)', 'scan_drop',
           not sanity.check('scan_drop', sanity.largest_drop(e), log=lambda x: None),
           False, f'rises {e[0]:+.1f} to {e[-1]:+.1f}, no fall')


def main():
    w = max(len(r[0]) for r in rows)
    print(f'{"case":{w}}  {"check":18}  {"fires":5}  {"want":5}  verdict')
    print('-' * (w + 52))
    bad = 0
    for name, kind, fired, expect, ok, detail in rows:
        bad += (not ok)
        print(f'{name:{w}}  {kind:18}  {str(fired):5}  {str(expect):5}  '
              f'{"ok" if ok else "WRONG"}   {detail}')
    pos = sum(1 for r in rows if r[3])
    print(f'\n{len(rows)} cases: {pos} known failures that must fire, '
          f'{len(rows) - pos} verified-good that must stay silent')
    print('all correct' if not bad else f'{bad} CASE(S) WRONG')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
