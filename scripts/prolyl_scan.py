"""cis/trans isomerisation of a prolyl amide, and what phosphorylation does to it.

The conformational switch behind cis-pThr231-Pro tau. For each capped dipeptide
Ac-Xaa-Pro-NHMe the amide torsion omega is scanned right round, so both senses
of rotation are covered, with GFN2-xTB (ALPB water); the stationary points are
then re-optimised and given DFT single points. The observable that matters is
the DIFFERENCE phosphorylated minus unphosphorylated, for the barrier and for
the cis/trans equilibrium; the absolute barrier is also meaningful here, since
the system is small enough for a reliable level of theory and experiment puts
prolyl barriers at 14-24 kcal/mol.

Stages, resumable from results/prolyl.json:

  scan    relaxed omega scan, every model, GFN2-xTB/ALPB water
  points  trans and cis minima and the two saddles, refined
  dft     wB97X-D3(BJ)/def2-TZVP with C-PCM(water) at those geometries

    python scripts/prolyl_scan.py [stage ...]
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit.prolyl import MODELS, build, torsion
from chemoqbit.enzyme.frozen import optimise_frozen
from chemoqbit import sanity

OUT = ROOT / 'results' / 'prolyl.json'
K = 627.509474
STEP = 15
OMEGA = list(range(-180, 180, STEP))         # the full circle: both rotation senses
SOLVENT = 'water'
KTOR = 3.0     # Eh/rad^2. At 0.05 the restraint could not hold the torsion against
               # the rotation barrier: omega came out 13-21 deg from target near the
               # saddle, so the scan never sampled it. At 3.0 it tracks to 0.2 deg.
XC, BASIS = 'wb97x-d3bj', 'def2-tzvp'


def save(d):
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(OUT)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


MAXITER, GTOL = 500, 5e-4      # the soft torsion restraint never reaches a tighter
                               # gradient, so cap the iterations instead of grinding


def relax(sym, X, q, cons, k=KTOR, maxiter=MAXITER):
    try:
        Xn, E = optimise_frozen(sym, X, charge=q, constraints=cons, k=k, solvent=SOLVENT,
                                maxiter=maxiter, gtol=GTOL)
        return np.array(Xn), float(E), True
    except Exception as e:
        print(f'    failed: {type(e).__name__}: {e}', flush=True)
        return np.array(X), None, False


def sweep(sym, X, om, q, targets, label):
    """Relax along `targets` in order, each point starting from the previous, then
    a backward pass keeping the lower energy -- the same guard the enzyme scans
    need, since the rest of the molecule can drop into another conformer mid-scan
    and leave a step in the profile."""
    rows, Xc = {}, X.copy()
    for w in targets:
        t0 = time.time()
        Xn, E, ok = relax(sym, Xc, q, [('tor', *om, float(w))])
        if ok:
            Xc = Xn
        w_act = torsion(Xc, om) if ok else None
        rows[w] = dict(omega_target=w, omega=w_act, E=E,
                       X=Xc.tolist() if ok else None, seconds=time.time() - t0)
        if w_act is not None:
            err = abs(((w_act - w) + 180) % 360 - 180)
            rows[w]['omega_error'] = err
            sanity.check('torsion_error', err, label=f'{label} target {w:+d}')
        print(f'  {label} omega {w:+4d} -> {rows[w]["omega"] if ok else float("nan"):+7.1f}  '
              + ('failed' if E is None else f'E {E:.6f}') + f'  ({time.time() - t0:.0f}s)', flush=True)
    for w in list(reversed(targets))[1:]:
        prev = rows[targets[targets.index(w) + 1]]
        if not prev['X']:
            continue
        Xn, E, ok = relax(sym, np.array(prev['X']), q, [('tor', *om, float(w))])
        if ok and (rows[w]['E'] is None or E < rows[w]['E'] - 1e-7):
            print(f'  {label} omega {w:+4d} backward pass lowers '
                  f'{rows[w]["E"] if rows[w]["E"] is not None else float("nan"):.6f} -> {E:.6f}', flush=True)
            rows[w] = dict(omega_target=w, omega=torsion(Xn, om), E=E, X=Xn.tolist(), backward=True)
    return rows


def stage_scan(res):
    res.setdefault('scan', {})
    for name in MODELS:
        if name in res['scan']:
            continue
        sym, X0, om, q = build(name)
        # start from the relaxed trans amide, then sweep each rotation sense
        # monotonically away from it. Ordering by distance from 180 alternated
        # between the two senses, so consecutive points started from geometries on
        # opposite branches and the profile mixed conformers.
        Xt, Et, ok = relax(sym, X0, q, [('tor', *om, 180.0)])
        plus = list(range(180, -1, -STEP))                   # 180 -> 0 through +90
        minus = list(range(-180, 1, STEP))                   # -180 -> 0 through -90
        assert plus[0] == 180 and minus[0] == -180, 'both sweeps must start at the trans amide'
        rp = sweep(sym, Xt, om, q, plus, f'{name:6s} +')
        rm = sweep(sym, Xt, om, q, minus, f'{name:6s} -')
        rows = {}
        for w, v in list(rp.items()) + list(rm.items()):
            k_ = str(((w + 180) % 360) - 180)
            if k_ not in rows or (v['E'] is not None and (rows[k_]['E'] is None or v['E'] < rows[k_]['E'])):
                rows[k_] = v
        good = {int(k_): v for k_, v in rows.items() if v['E'] is not None}
        e0 = min(v['E'] for v in good.values())
        res['scan'][name] = dict(charge=q, omega_atoms=list(om), symbols=sym, points=rows, e_min=e0,
                                 profile={k_: (v['E'] - e0) * K for k_, v in sorted(good.items())})
        save(res)
        prof = res['scan'][name]['profile']
        tr = min((k_ for k_ in prof if abs(abs(k_) - 180) <= 30), key=lambda k_: prof[k_], default=None)
        ci = min((k_ for k_ in prof if abs(k_) <= 30), key=lambda k_: prof[k_], default=None)
        top = max(prof.values())
        print(f'  {name}: trans {tr} ({prof.get(tr, float("nan")):.2f}), cis {ci} '
              f'({prof.get(ci, float("nan")):.2f}), highest {top:.2f} kcal/mol', flush=True)
        # the profile must look like an amide rotation before the next model runs
        e_tr = prof.get(tr)
        sanity.record(res, 'prolyl_barrier', top - (e_tr or 0.0), label=f'{name} scan')
        if e_tr is not None and ci in prof:
            sanity.record(res, 'prolyl_cis_trans', prof[ci] - e_tr, label=f'{name} scan')
        for lab_, rr in (('+', rp), ('-', rm)):
            e = [v['E'] for v in rr.values() if v['E'] is not None]
            if len(e) > 2:
                i = int(np.argmax(e))
                sanity.record(res, 'scan_drop',
                              sanity.largest_drop([(v - e[0]) * K for v in e[:i + 1]]),
                              label=f'{name} sweep {lab_}, rising branch')
        worst = max((v.get('omega_error', 0.0) for v in rows.values() if v.get('omega_error')), default=0.0)
        sanity.record(res, 'torsion_error', worst, label=f'{name} worst point')
        save(res)


def _stationary(prof):
    """(trans, cis, saddle_pos, saddle_neg) omega keys from the scan profile"""
    ks = sorted(prof)
    tr = min((k for k in ks if abs(abs(k) - 180) <= 40), key=lambda k: prof[k], default=None)
    ci = min((k for k in ks if abs(k) <= 40), key=lambda k: prof[k], default=None)
    sp = max((k for k in ks if 20 < k < 160), key=lambda k: prof[k], default=None)
    sn = max((k for k in ks if -160 < k < -20), key=lambda k: prof[k], default=None)
    return tr, ci, sp, sn


def stage_points(res):
    res.setdefault('points', {})
    for name, d in res['scan'].items():
        if name in res['points']:
            continue
        prof = {int(k): v for k, v in d['profile'].items()}
        tr, ci, sp, sn = _stationary(prof)
        sym, q, om = d['symbols'], d['charge'], tuple(d['omega_atoms'])
        out = {}
        for tag, w in (('trans', tr), ('cis', ci), ('saddle+', sp), ('saddle-', sn)):
            if w is None:
                continue
            X = np.array(d['points'][str(w)]['X'])
            if tag in ('trans', 'cis'):           # free relaxation: a true minimum
                Xn, E, ok = relax(sym, X, q, [], k=0.0)
            else:                                 # hold the torsion at the saddle
                Xn, E, ok = relax(sym, X, q, [('tor', *om, float(w))], k=1.0)
            out[tag] = dict(omega_scan=w, omega=torsion(Xn, om) if ok else None, E=E,
                            X=Xn.tolist() if ok else None)
            print(f'  point {name:6s} {tag:8s} omega {out[tag]["omega"]}  '
                  + ('failed' if E is None else f'E {E:.6f}'), flush=True)
        if 'trans' in out and out['trans']['E'] is not None:
            e_tr = out['trans']['E']
            for tag in ('saddle+', 'saddle-'):
                if tag in out and out[tag]['E'] is not None:
                    sanity.record(res, 'prolyl_barrier', (out[tag]['E'] - e_tr) * K,
                                  label=f'{name} {tag}')
            if 'cis' in out and out['cis']['E'] is not None:
                sanity.record(res, 'prolyl_cis_trans', (out['cis']['E'] - e_tr) * K, label=f'{name} cis')
        res['points'][name] = out
        save(res)


def stage_dft(res):
    from pyscf import gto, dft
    res.setdefault('dft', {})
    for name, pts in res['points'].items():
        res['dft'].setdefault(name, {})
        sym, q = res['scan'][name]['symbols'], res['scan'][name]['charge']
        for tag, p in pts.items():
            if tag in res['dft'][name] or not p['X']:
                continue
            t0 = time.time()
            mol = gto.M(atom=[(s, x) for s, x in zip(sym, p['X'])], basis=BASIS, charge=q,
                        unit='Angstrom', verbose=0)
            mf = dft.RKS(mol, xc=XC).density_fit()
            mf = mf.PCM()
            mf.with_solvent.eps = 78.4
            mf.conv_tol = 1e-9
            E = float(mf.kernel())
            res['dft'][name][tag] = dict(E=E, converged=bool(mf.converged), seconds=time.time() - t0)
            save(res)
            print(f'  dft {name:6s} {tag:8s} E {E:.6f} ({time.time() - t0:.0f}s)', flush=True)
        d = res['dft'][name]
        if 'trans' in d and 'cis' in d:
            sad = [d[t]['E'] for t in ('saddle+', 'saddle-') if t in d]
            print(f"  {name}: DFT cis-trans {(d['cis']['E'] - d['trans']['E']) * K:+.2f}, "
                  f"barrier trans->cis {(min(sad) - d['trans']['E']) * K:.2f} kcal/mol" if sad else '', flush=True)


STAGES = dict(scan=stage_scan, points=stage_points, dft=stage_dft)


def main(names):
    res = load()
    for n in names:
        print(f'=== {n}', flush=True)
        STAGES[n](res)
        res['_stages_done'] = sorted(set(res.get('_stages_done', [])) | {n})
        save(res)


if __name__ == '__main__':
    main(sys.argv[1:] or ['scan', 'points', 'dft'])
