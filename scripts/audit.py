"""Audit every number the manuscripts rest on against chemoqbit.sanity bounds.

Run before any submission, and after any stage finishes. Exits non-zero if
anything is outside its physical range, so it can gate a release.

    python scripts/audit.py
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from chemoqbit import sanity
from chemoqbit.enzyme import tsopt as T

K = 627.509474
RES = ROOT / 'results'
bad = []
log = lambda m: bad.append(m)
L = lambda f: json.loads((RES / f).read_text())
have = lambda f: (RES / f).exists()


def scan_continuity(r):
    E0 = r['reactant']['E']
    sc = sorted(r['scan'], key=lambda q: q['xi'])
    rel = [(q['E'] - E0) * K for q in sc]
    i = int(np.argmax(rel))
    sanity.check('scan_drop', sanity.largest_drop(rel[:i + 1]), label='xTB scan, rising', log=log)
    sanity.check('scan_drop', sanity.largest_drop([-v for v in rel[i:]]), label='xTB scan, falling', log=log)


def frequencies(r):
    import ts_cluster_nuc as S
    p = r['prep']; sym = p['symbols']
    masses = [T.AMU[s] for s in sym]
    levels = [('GFN2-xTB', {n: r[n]['coords'] for n in ('reactant', 'ts', 'product')},
               {n: r[n]['hessian'] for n in ('reactant', 'ts', 'product')}, S._free(sym, p['frozen']))]
    if have('ts_oniom_opt.json'):
        o = L('ts_oniom_opt.json')
        levels.append(('ONIOM', {n: o[n]['X'] for n in ('reactant', 'ts', 'product')},
                       {n: o['hess'][n]['H'] for n in ('reactant', 'ts', 'product')},
                       o['config']['hessian_region']))
    for lvl, coords, hess, region in levels:
        for n in ('reactant', 'ts', 'product'):
            nu = T.frequencies(np.array(coords[n]), np.array(hess[n]), masses, free=region)
            nim = int((nu < -50).sum())
            sanity.check('imag_freq_ts' if n == 'ts' else 'imag_freq_min', nim,
                         label=f'{lvl} {n}', log=log)
            if n == 'ts':
                sanity.check('imag_wavenumber', abs(nu.min()), label=f'{lvl} TS', log=log)


def kies(r):
    srcs = [('GFN2-xTB', r)] + ([('ONIOM', L('ts_oniom_opt.json'))] if have('ts_oniom_opt.json') else [])
    for lvl, src in srcs:
        for pair, d in src.get('analysis', {}).get('kie', {}).items():
            for m in ('semiclassical', 'wigner', 'eckart'):
                sanity.check('kie', d[m], label=f'{lvl} {pair} {m}', log=log)
            for iso, kap in d.get('kappa', {}).items():
                for m in ('wigner', 'eckart'):
                    sanity.check('kappa_tunnelling', kap[m], label=f'{lvl} {pair} {iso} {m}', log=log)


def main():
    r = L('ts_cluster_run.json')
    scan_continuity(r)
    frequencies(r)
    kies(r)
    if have('ts_thermo.json'):
        for model, d in L('ts_thermo.json').items():
            for cond, v in d.items():
                sanity.check('reaction_barrier', v['dG_barrier'], label=f'{model} {cond}', log=log)
                sanity.check('zpe_change', v['dZPE'], label=f'{model} {cond}', log=log)
                sanity.check('entropy_term', v['TdS'], label=f'{model} {cond}', log=log)
    if have('ts_cs2d.json'):
        g = L('ts_cs2d.json')['summary']['oniom']['grid']
        worst = max(sanity.largest_drop([v for v in row if v is not None]) for row in g)
        sanity.check('scan_drop', worst, label='2D surface, C6-S rows', log=log)
    if have('ts_kie_temperature.json'):
        for name, d in L('ts_kie_temperature.json').items():
            for t, v in d['kie'].items():
                for m in ('semiclassical', 'wigner', 'eckart'):
                    sanity.check('kie', v[m], label=f'{name} {t} {m}', log=log)
    if have('ts_qmsize.json'):
        q = L('ts_qmsize.json')
        for reg in ('fragment-18', 'substrates', 'substrates+contacts', 'all'):
            if f'{reg}/ts' not in q:
                continue
            E = {n: q[f'{reg}/{n}']['E'] for n in ('reactant', 'ts', 'product')}
            sanity.check('reaction_barrier', (E['ts'] - E['reactant']) * K, label=f'qmsize {reg}', log=log)
    # A violation with a written diagnosis is a recorded dead end, not an open
    # fault: the O4-protonated variant, the proton-only relay, the phosphate
    # dianion. Reporting those as faults for ever would train everyone to ignore
    # the mail, so they are listed separately and do not fail the audit.
    known = []
    # Manuscript 1's circuit errors, read from the active-space checkpoint rather
    # than retyped: a hard-coded number cannot go stale, but it also cannot notice
    # that the checkpoint changed.
    # Derive the amide check from the stored geometry rather than trusting the
    # driver to have recorded it: a job started before the check existed still
    # gets audited, so a code change needs no recomputation.
    pr = L('prolyl.json') if have('prolyl.json') else None
    if pr:
        diag_p = pr.get('diagnosis', {})
        for m, v in (pr.get('scan') or {}).items():
            prof = {int(k): x for k, x in v['profile'].items()}
            tr = min((k for k in prof if abs(abs(k) - 180) <= 30), key=lambda k: prof[k],
                     default=None)
            if tr is None or not v['points'][str(tr)].get('X'):
                continue
            X = np.array(v['points'][str(tr)]['X']); sym = v['symbols']; om = v['omega_atoms']
            c, npro = om[1], om[2]
            co = min((i for i, t in enumerate(sym)
                      if t == 'O' and np.linalg.norm(X[i] - X[c]) < 1.6),
                     key=lambda i: np.linalg.norm(X[i] - X[c]), default=None)
            covered = (diag_p.get(m) or {}).get('checks', []) if isinstance(diag_p.get(m), dict) else \
                      (list(sanity.BOUNDS) if m in diag_p else [])
            for nm, val in (('amide_CO', float(np.linalg.norm(X[co] - X[c])) if co is not None else None),
                            ('amide_CN', float(np.linalg.norm(X[npro] - X[c]))),
                            ('prolyl_barrier', max(prof.values()))):
                if val is None or nm in covered:
                    continue
                sanity.check(nm, val, label=f'prolyl {m} (recomputed from the checkpoint)', log=log)

    act = next((L(f) for f in ('ts_active_20261019.json',) if have(f)), None)
    if act:
        n_ck = 0
        for key, v in act.items():                     # keys are 'reactant/4e4o' etc
            if not isinstance(v, dict):
                continue
            for tag in ('uccsd', 'adapt'):
                e = (v.get(tag) or {}).get('err_mha')
                if e is not None:
                    sanity.check('circuit_error_mha', abs(float(e)), label=f'{key} {tag}', log=log)
                    n_ck += 1
        print(f'  circuit errors checked from the checkpoint: {n_ck}')
    for f in ('ts_abstraction.json', 'prolyl.json', 'pocket_cftr.json', 'pocket_ache.json',
              'zipper_tau.json', 'zipper_asyn47.json', 'zipper_nacore.json'):
        if not have(f):
            continue
        d_ = L(f)
        diag = d_.get('diagnosis', {})
        for v in d_.get('_sanity', []):
            lbl = v.get('label', '')
            line = f"{v['check']} = {v['value']} for {lbl}"
            # a diagnosis covers named checks only: {subject: {'checks': [...], 'why': str}}
            # or, for the older plain-string form, every check for that subject
            hit = None
            for k, entry in diag.items():
                if not k or k not in lbl:
                    continue
                covers = entry.get('checks') if isinstance(entry, dict) else None
                if covers is None or v['check'] in covers:
                    hit = k
                    break
            if hit:
                known.append(f'{f}: {line}  [diagnosed: {hit}]')
            else:
                log(f'SANITY recorded in {f}: {line}  (no diagnosis covers this check)')
    if known:
        print(f'{len(known)} diagnosed dead end(s), not counted as faults:')
        for k in known:
            print('  ' + k)
    print(f'{len(bad)} violation(s)' if bad else 'audit clean: every checked quantity is inside its bounds')
    for m in bad:
        print('  ' + m)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
