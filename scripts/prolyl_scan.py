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
from chemoqbit.prolyl import MODELS, build, torsion, rotation_guide
from chemoqbit.enzyme.frozen import optimise_frozen
from chemoqbit.enzyme import tsopt as TS
from chemoqbit.enzyme.xtb import energy_grad
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
        # Relax freely FIRST, then sweep away from wherever that lands. Forcing
        # omega to 180 from the embedded geometry wrenched Ac-Thr-Pro-NHMe by 175
        # deg in one step and left every point of its scan about 41 kcal/mol above
        # the true minimum; Ac-Ala-Pro-NHMe happened to embed near trans and was
        # unaffected, which is why only one model looked wrong.
        Xf, Ef, okf = relax(sym, X0, q, [], k=0.0, maxiter=2000)
        w0 = torsion(Xf, om)
        start = int(round(w0 / STEP) * STEP)
        sanity.record(res, 'start_wrench', abs(((w0 - start) + 180) % 360 - 180),
                      label=f'{name} sweep start')
        print(f'  {name}: free minimum E {Ef:.6f} at omega {w0:+.1f}; sweeping from {start:+d}',
              flush=True)
        # walk the full circle in both directions from there, so no step exceeds STEP
        fwd = [((start - k * STEP + 180) % 360) - 180 for k in range(1, 360 // STEP + 1)]
        bwd = [((start + k * STEP + 180) % 360) - 180 for k in range(1, 360 // STEP + 1)]
        r0 = {start: dict(omega_target=start, omega=w0, E=Ef, X=Xf.tolist(), omega_error=0.0)}
        rp = sweep(sym, Xf, om, q, fwd, f'{name:6s} >')
        rm = sweep(sym, Xf, om, q, bwd, f'{name:6s} <')
        rows = {}
        for w, v in list(r0.items()) + list(rp.items()) + list(rm.items()):
            k_ = str(((int(w) + 180) % 360) - 180)
            if k_ not in rows or (v['E'] is not None and (rows[k_]['E'] is None or v['E'] < rows[k_]['E'])):
                rows[k_] = v
        good = {int(k_): v for k_, v in rows.items() if v['E'] is not None}
        e0 = min(v['E'] for v in good.values())
        # If the sweep found an arrangement better than the free minimisation it
        # started from, that start was not the global minimum -- the phosphate and
        # hydroxyl rotamers give several basins -- so re-relax from the best point
        # and sweep again from there. One refinement, to keep it bounded.
        if e0 < Ef - 1.0 / K:
            best = min(good.values(), key=lambda v: v['E'])
            Xb, Eb, okb = relax(sym, np.array(best['X']), q, [], k=0.0, maxiter=2000)
            if okb and Eb < Ef:
                print(f'  {name}: the sweep found a basin {(Ef - Eb) * K:.2f} kcal/mol lower; '
                      f'restarting it from there', flush=True)
                Ef, Xf = Eb, Xb
                w0 = torsion(Xf, om)
                start = int(round(w0 / STEP) * STEP)
                fwd = [((start - k * STEP + 180) % 360) - 180 for k in range(1, 360 // STEP + 1)]
                bwd = [((start + k * STEP + 180) % 360) - 180 for k in range(1, 360 // STEP + 1)]
                r0 = {start: dict(omega_target=start, omega=w0, E=Ef, X=Xf.tolist(), omega_error=0.0)}
                rp = sweep(sym, Xf, om, q, fwd, f'{name:6s} >2')
                rm = sweep(sym, Xf, om, q, bwd, f'{name:6s} <2')
                rows = {}
                for w, v in list(r0.items()) + list(rp.items()) + list(rm.items()):
                    k_ = str(((int(w) + 180) % 360) - 180)
                    if k_ not in rows or (v['E'] is not None and (rows[k_]['E'] is None or v['E'] < rows[k_]['E'])):
                        rows[k_] = v
                good = {int(k_): v for k_, v in rows.items() if v['E'] is not None}
                e0 = min(v['E'] for v in good.values())
        res['scan'][name] = dict(charge=q, omega_atoms=list(om), symbols=sym, points=rows,
                                 e_min=e0, e_free=Ef, omega_free=w0,
                                 profile={k_: (v['E'] - e0) * K for k_, v in sorted(good.items())})
        # the scan must not sit above a free minimisation of the same molecule
        sanity.record(res, 'scan_vs_free_min', (Ef - e0) * K, label=f'{name} scan vs free minimum')
        prof = res['scan'][name]['profile']
        tr = min((k_ for k_ in prof if abs(abs(k_) - 180) <= 30), key=lambda k_: prof[k_], default=None)
        ci = min((k_ for k_ in prof if abs(k_) <= 30), key=lambda k_: prof[k_], default=None)
        print(f'  {name}: trans {tr} ({prof.get(tr, float("nan")):.2f}), cis {ci} '
              f'({prof.get(ci, float("nan")):.2f}), highest {max(prof.values()):.2f} kcal/mol', flush=True)
        # Check the MERGED profile, which is what every later stage uses -- not each
        # raw sweep. The two sweeps exist precisely so that a point trapped in one
        # is replaced by the other's better value, and testing a sweep alone reports
        # that repair as a fault: it flagged thr with 5 monotonic runs while the
        # merged profile had the textbook 4.
        prof = res['scan'][name]['profile']
        prof = {int(k_): x for k_, x in prof.items()}
        seq = [prof[k_] for k_ in sorted(prof)]
        # the amide must still be an amide: a dianion at GFN2-xTB delocalised its
        # excess charge into the carbonyl, leaving C=O 1.31 and C-N 1.46 (single
        # bonds) and a 'rotation barrier' of 9.9 with no pi system left to break
        tr_ = min((k_ for k_ in prof if abs(abs(k_) - 180) <= 30), key=lambda k_: prof[k_],
                  default=None)
        if tr_ is not None and rows[str(tr_)].get('X'):
            Xt = np.array(rows[str(tr_)]['X'])
            c_, n_ = om[1], om[2]
            co = min((i for i, s_ in enumerate(sym)
                      if s_ == 'O' and np.linalg.norm(Xt[i] - Xt[c_]) < 1.6),
                     key=lambda i: np.linalg.norm(Xt[i] - Xt[c_]), default=None)
            if co is not None:
                sanity.record(res, 'amide_CO', float(np.linalg.norm(Xt[co] - Xt[c_])),
                              label=f'{name} trans')
            sanity.record(res, 'amide_CN', float(np.linalg.norm(Xt[n_] - Xt[c_])),
                          label=f'{name} trans')
        sanity.record(res, 'profile_segments', sanity.monotonic_segments(seq),
                      label=f'{name} merged profile')
        sanity.record(res, 'scan_drop', sanity.segment_drop(seq),
                      label=f'{name} merged profile, inside a monotonic run')
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


def torsion_guide(X, om, h=1e-3):
    """d(omega)/dX as a 3N displacement, for the saddle search to follow."""
    g = np.zeros_like(X)
    for a in om:
        for c in range(3):
            Xp = X.copy(); Xp[a, c] += h
            Xm = X.copy(); Xm[a, c] -= h
            dp = ((torsion(Xp, om) - torsion(Xm, om)) + 180) % 360 - 180
            g[a, c] = dp / (2 * h)
    return g


def stage_points(res):
    """Trans and cis as free minima, the barrier as a genuine P-RFO saddle.

    A torsion-restrained point is not a stationary point, and with a rotatable
    side chain the relaxed scan mixes conformers: for Ac-Thr-Pro-NHMe the scan
    minimum came out at omega = -135 deg, which is meaningless for an amide,
    and the barrier read off it was 5 kcal/mol. A saddle search gives a
    well-defined energy and the frequencies prove what it is: exactly one
    imaginary mode at the transition state, none at the minima.
    """
    res.setdefault('points', {})
    for name, d in res['scan'].items():
        if name in res['points']:
            continue
        prof = {int(k): v for k, v in d['profile'].items()}
        tr, ci, sp, sn = _stationary(prof)
        sym, q, om = d['symbols'], d['charge'], tuple(d['omega_atoms'])
        masses = [TS.AMU[s_] for s_ in sym]
        fn = lambda Y: (lambda r: (float(r[0]), r[1]))(energy_grad(sym, Y, q, solvent=SOLVENT))
        gfn = lambda Y: energy_grad(sym, Y, q, solvent=SOLVENT)[1]
        out = {}
        for tag, w in (('trans', tr), ('cis', ci), ('saddle', sp if sp is not None else sn)):
            if w is None or not d['points'][str(w)].get('X'):
                continue
            X = np.array(d['points'][str(w)]['X'])
            t0 = time.time()
            info = {}
            try:
                if tag in ('trans', 'cis'):
                    Xn, E, ok = relax(sym, X, q, [], k=0.0, maxiter=1500)
                    kind = 'minimum'
                else:
                    # Curvature in Eh/A^2 and frequency in cm-1 differ by the reduced
                    # mass. This mode rotates the whole proline ring, so a genuine
                    # transition state shows only -3e-4 Eh/A^2: demanding -0.02, set for
                    # a light local mode, rejected all seven. The optimiser's job is to
                    # find a stationary point; classifying it is the frequencies' job,
                    # and n_imaginary == 1 is checked straight after.
                    Xn, E, info = TS.prfo_follow(X, fn, rotation_guide(sym, X, om), grad_fn=gfn,
                                                 trust=0.05, gtol=3e-4, maxiter=60, neg=1e-4,
                                                 log=lambda m: None)
                    E, ok, kind = float(E), True, 'saddle'
            except Exception as e:
                print(f'  point {name:6s} {tag:8s} failed: {type(e).__name__}: {e}', flush=True)
                out[tag] = dict(failed=f'{type(e).__name__}: {e}'); continue
            nu = TS.frequencies(np.array(Xn), TS.fd_hessian(np.array(Xn), gfn), masses)
            nim = int((nu < -50).sum())
            out[tag] = dict(omega_scan=w, omega=torsion(Xn, om), E=E, X=Xn.tolist(), kind=kind,
                            n_imaginary=nim, nu_lowest=float(nu.min()),
                            nu_imag=float(nu.min()) if nim else None,
                            followed_eig=(info.get('followed_eig') if tag == 'saddle' else None),
                            seconds=time.time() - t0)
            sanity.record(res, 'imag_freq_ts' if tag == 'saddle' else 'imag_freq_min', nim,
                          label=f'{name} {tag}')
            if nim:
                sanity.record(res, 'imag_wavenumber', abs(nu.min()), label=f'{name} {tag}')
            print(f'  point {name:6s} {tag:8s} omega {out[tag]["omega"]:+7.1f}  E {E:.6f}  '
                  f'imaginary {nim}' + (f' ({abs(nu.min()):.0f}i cm-1)' if nim else '')
                  + f'  ({time.time() - t0:.0f}s)', flush=True)
        res['points'][name] = out
        if 'trans' in out and out['trans'].get('E') is not None:
            e_tr = out['trans']['E']
            if out.get('saddle', {}).get('E') is not None:
                sanity.record(res, 'prolyl_barrier', (out['saddle']['E'] - e_tr) * K,
                              label=f'{name} saddle')
                print(f"  {name}: barrier {(out['saddle']['E'] - e_tr) * K:.2f} kcal/mol", flush=True)
            if out.get('cis', {}).get('E') is not None:
                sanity.record(res, 'prolyl_cis_trans', (out['cis']['E'] - e_tr) * K, label=f'{name} cis')
        save(res)


def stage_dft(res):
    from pyscf import gto, dft
    res.setdefault('dft', {})
    for name, pts in res['points'].items():
        res['dft'].setdefault(name, {})
        sym, q = res['scan'][name]['symbols'], res['scan'][name]['charge']
        for tag, p in pts.items():
            if tag in res['dft'][name] or not p.get('X'):
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
