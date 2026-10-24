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

# the dianion whose amide stopped being an amide at GFN2-xTB
import numpy as _np
cur_ = load('prolyl.json') or {}
v_ = (cur_.get('scan') or {}).get('pthr2')
if v_:
    prof_ = {int(k): x for k, x in v_['profile'].items()}
    tr_ = min((k for k in prof_ if abs(abs(k) - 180) <= 30), key=lambda k: prof_[k])
    X_ = _np.array(v_['points'][str(tr_)]['X']); sym_ = v_['symbols']; om_ = v_['omega_atoms']
    c_, n_ = om_[1], om_[2]
    co_ = min((i for i, t in enumerate(sym_)
               if t == 'O' and _np.linalg.norm(X_[i] - X_[c_]) < 1.6),
              key=lambda i: _np.linalg.norm(X_[i] - X_[c_]))
    fired_ = (not sanity.check('amide_CO', float(_np.linalg.norm(X_[co_] - X_[c_])), log=lambda m: None)
              or not sanity.check('amide_CN', float(_np.linalg.norm(X_[n_] - X_[c_])), log=lambda m: None))
    record('phosphate dianion: the amide lost its conjugation', 'amide_CO/CN', fired_, True,
           f'C=O {_np.linalg.norm(X_[co_] - X_[c_]):.3f}, C-N {_np.linalg.norm(X_[n_] - X_[c_]):.3f} A')
    record('the same model, its barrier', 'prolyl_barrier',
           not sanity.check('prolyl_barrier', max(prof_.values()), log=lambda m: None), True,
           f'{max(prof_.values()):.2f} kcal/mol, half the other models')

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
    if not seq or m == 'pthr2':       # a known-bad model, tested among the positives
        continue
    record(f'current prolyl {m}, barrier', 'prolyl_barrier',
           not sanity.check('prolyl_barrier', max(seq), log=lambda x: None), False,
           f'{max(seq):.2f} kcal/mol')
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


# ---------------------------------------------- bounds that had no test at all
# Each is exercised in both directions: a value that must fire and one that must not.
# Where a real result exists it is used; where the job has not run yet (the drug
# pockets) the bound itself is tested on values whose verdict is not in doubt.

prep_ = ((load('ts_abstraction.json') or {}).get('prep') or {}).get('H') or {}
g_ = prep_.get('geometry') or {}
if g_:
    record('covalent adduct C6-S in the relaxed reactant', 'bond_CS',
           not sanity.check('bond_CS', g_['C6-SG'], log=lambda m: None), False,
           f"{g_['C6-SG']:.2f} A")
    record('covalent adduct C5-C11', 'bond_CC',
           not sanity.check('bond_CC', g_['C5-C11'], log=lambda m: None), False,
           f"{g_['C5-C11']:.2f} A")
    record('a broken C-S bond at 3.2 A', 'bond_CS',
           not sanity.check('bond_CS', 3.2, log=lambda m: None), True, 'the adduct has come apart')
    record('C5-H in the relaxed reactant', 'bond_CH',
           not sanity.check('bond_CH', g_.get('C5-H', 1.1), log=lambda m: None), False,
           f"{g_.get('C5-H', 1.1):.2f} A")
    record('a proton lost to the cavity at 2.1 A', 'bond_CH',
           not sanity.check('bond_CH', 2.1, log=lambda m: None), True, 'bonded to neither partner')

th_ = load('ts_thermo.json') or {}
for model, conds in th_.items():
    for cond, v in list(conds.items())[:1]:
        record(f'published free energy, {model}', 'zpe_change',
               not sanity.check('zpe_change', v['dZPE'], log=lambda m: None), False,
               f"dZPE {v['dZPE']:.2f}")
        record(f'published free energy, {model}', 'entropy_term',
               not sanity.check('entropy_term', v['TdS'], log=lambda m: None), False,
               f"-TdS {v['TdS']:.2f}")
record('an entropy term of -40 kcal/mol', 'entropy_term',
       not sanity.check('entropy_term', -40.0, log=lambda m: None), True,
       'impossible for a unimolecular step in a fixed cluster')

r_ = load('ts_cluster_run.json') or {}
for pair, dd in (r_.get('analysis', {}).get('kie', {})).items():
    for iso, kap in (dd.get('kappa') or {}).items():
        for m_ in ('wigner', 'eckart'):
            record(f'published tunnelling factor {pair} {iso} {m_}', 'kappa_tunnelling',
                   not sanity.check('kappa_tunnelling', kap[m_], log=lambda x: None), False,
                   f"{kap[m_]:.3f}")
        break
    break
record('a tunnelling factor below 1', 'kappa_tunnelling',
       not sanity.check('kappa_tunnelling', 0.8, log=lambda m: None), True,
       'tunnelling cannot slow a reaction')

cur_p = load('prolyl.json') or {}
for m, v in (cur_p.get('scan') or {}).items():
    if m == 'pthr2' or 'e_free' not in v:
        continue
    record(f'current prolyl {m}, sweep start', 'start_wrench',
           not sanity.check('start_wrench',
                            abs(((v['omega_free'] - round(v['omega_free'] / 15) * 15) + 180) % 360 - 180),
                            log=lambda x: None), False, 'sweep begins at the free minimum')
    prof_m = {int(k): x for k, x in v['profile'].items()}
    ci_ = min((k for k in prof_m if abs(k) <= 30), key=lambda k: prof_m[k], default=None)
    tr_m = min((k for k in prof_m if abs(abs(k) - 180) <= 30), key=lambda k: prof_m[k], default=None)
    if ci_ is not None and tr_m is not None:
        record(f'current prolyl {m}, cis minus trans', 'prolyl_cis_trans',
               not sanity.check('prolyl_cis_trans', prof_m[ci_] - prof_m[tr_m], log=lambda x: None),
               False, f'{prof_m[ci_] - prof_m[tr_m]:+.2f} kcal/mol')
    break
record('a cis amide 20 kcal/mol below trans', 'prolyl_cis_trans',
       not sanity.check('prolyl_cis_trans', -20.0, log=lambda m: None), True,
       'the inversion the soft-restraint run produced')

record('a bound complex that is uphill', 'interaction_energy',
       not sanity.check('interaction_energy', +12.0, log=lambda m: None), True,
       'a ligand that does not bind')
record('a plausible pocket interaction energy', 'interaction_energy',
       not sanity.check('interaction_energy', -84.0, log=lambda m: None), False, '-84 kcal/mol')
record('an alanine truncation worth 120 kcal/mol', 'alanine_ddE',
       not sanity.check('alanine_ddE', 120.0, log=lambda m: None), True,
       'one side chain cannot be worth that')
record('a plausible alanine ddE', 'alanine_ddE',
       not sanity.check('alanine_ddE', 3.4, log=lambda m: None), False, '+3.4 kcal/mol')

o_ = load('ts_oniom_opt.json') or {}
an_ = (o_.get('analysis') or {}).get('kie', {})
record('published ONIOM imaginary wavenumber', 'imag_wavenumber',
       not sanity.check('imag_wavenumber', 695.0, log=lambda m: None), False, '695i cm-1')
record('an imaginary mode of 40i cm-1', 'imag_wavenumber',
       not sanity.check('imag_wavenumber', 40.0, log=lambda m: None), True,
       'the shoulder the saddle search used to accept')
record('a minimum with one imaginary mode', 'imag_freq_min',
       not sanity.check('imag_freq_min', 1, log=lambda m: None), True, 'not a minimum')
record('a minimum with none', 'imag_freq_min',
       not sanity.check('imag_freq_min', 0, log=lambda m: None), False, 'as it should be')

a_ = load('ts_active_20261019.json') or {}
n_circ = 0
for key, v in a_.items():
    if not isinstance(v, dict):
        continue
    for tag in ('uccsd', 'adapt'):
        e = (v.get(tag) or {}).get('err_mha')
        if e is not None:
            record(f'published circuit error, {key} {tag}', 'circuit_error_mha',
                   not sanity.check('circuit_error_mha', abs(float(e)), log=lambda x: None),
                   False, f'{abs(float(e)):.4f} mHa')
            n_circ += 1
record('an uncapped fragment at 58 mHa', 'circuit_error_mha',
       not sanity.check('circuit_error_mha', 58.0, log=lambda m: None), True,
       '36x chemical accuracy')


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
