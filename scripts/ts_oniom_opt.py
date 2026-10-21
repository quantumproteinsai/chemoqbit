"""Reactant, transition state and product of the thymidylate synthase hydride
transfer optimised at ONIOM(wB97X-D3(BJ)/def2-SVP : GFN2-xTB), mechanical
embedding, with kinetic isotope effects from partial Hessians and final
ONIOM(wB97X-D3(BJ)/def2-TZVP : GFN2-xTB) energies.

Why: at the xTB geometries the barrier is ~37-38 kcal/mol by two independent
routes (ts_profile_check.py, ONIOM(CCSD(T):B3LYP)); the next question is
whether letting the structures relax on a surface that gets this barrier
right changes it. B3LYP is ~10 kcal/mol too low here, hence wB97X-D3.

Steps, each checkpointed in results/ts_oniom_opt.json (written atomically)
and resumed after an interruption, mid-optimisation included:

  ts        Bofill-updated P-RFO from the profile maximum (xi = +0.3), initial
            Hessian = the GFN2-xTB Hessian at the xTB TS, following d(xi)/dX
  reactant  L-BFGS from the xTB reactant
  product   L-BFGS from the xTB product
  hess      central-difference Hessians over the free atoms within 5 A of the
            hydride at the xTB TS (34 atoms): on the xTB Hessians this region
            reproduces the full-Hessian H/T KIE within 0.2 %
  kie       H/D/T kinetic isotope effects (kie_analysis, as for xTB)
  sp        ONIOM energies with wB97X-D3(BJ)/def2-TZVP for the fragment

    python scripts/ts_oniom_opt.py
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'results' / 'ts_oniom_opt.json'
SUFFIX = ''             # model variant, as ts_cluster_nuc.py --variant (e.g. '.his147p')
XC, BASIS, BASIS_SP = 'wb97x-d3bj', 'def2-svp', 'def2-tzvp'
START_XI = 0.3          # maximum of the ONIOM(wB97X-D3:xTB) profile along the xTB path
REGION_R = 5.0          # A, partial-Hessian region around the hydride
NAMES = ('reactant', 'ts', 'product')


def log(*a):
    print(*a, flush=True)


def save(res):
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(res, indent=1, default=float))
    tmp.replace(OUT)


def main():
    import ts_cluster_nuc as S
    from chemoqbit.enzyme import oniom as O, tsopt as T
    from chemoqbit.enzyme.frozen import penalty
    from chemoqbit.enzyme.xtb import energy_grad
    r = json.loads((ROOT / 'results' / f'ts_cluster_run{SUFFIX}.json').read_text())
    p = r['prep']; sym, k, frozen = p['symbols'], p['key'], p['frozen']
    rest = [tuple(c) for c in p.get('restraints', [])]
    atoms, cuts = S.active_region(r)
    free = S._free(sym, frozen)
    high = O.HighLevel(xc=XC, basis=BASIS)
    fn = O.oniom_fn(sym, p['charge'], atoms, cuts, high, extra=lambda X: penalty(X, rest))
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res.setdefault('config', dict(xc=XC, basis=BASIS, basis_sp=BASIS_SP, start_xi=START_XI,
                                  region_radius=REGION_R, fragment_atoms=atoms,
                                  cuts=[list(c) for c in cuts], restraints=[list(c) for c in rest]))
    t_start = time.time()

    # --- transition state
    if 'ts' not in res or 'X' not in res['ts']:
        X0 = np.array(next(q for q in r['scan'] if abs(q['xi'] - START_XI) < 1e-9)['coords'])
        u1 = X0[k['Hd']] - X0[k['Cd']]; u1 /= np.linalg.norm(u1)
        u2 = X0[k['C11']] - X0[k['Hd']]; u2 /= np.linalg.norm(u2)
        guide = np.zeros_like(X0); guide[k['Cd']] -= u1; guide[k['Hd']] += u1 + u2; guide[k['C11']] -= u2
        st = res.get('ts', {}).get('state')

        def keep(s):
            res['ts'] = dict(state=s); save(res)
        Xts, Ets, info = T.prfo_bofill(X0, fn, np.array(r['ts']['hessian']), guide, frozen,
                                       state=st, save=keep, log=log)
        res['ts'] = dict(X=Xts.tolist(), E=Ets, info=info); save(res)
        log(f"ONIOM-OPT ts: E={Ets:.6f} after {info['iterations']} steps, followed eig "
            f"{info['followed_eig']:+.4f} ({time.time() - t_start:.0f}s)")

    # --- minima
    for name in ('reactant', 'product'):
        if name in res and 'X' in res[name]:
            continue
        X = np.array(res.get(name, {}).get('current', r[name]['coords']))
        idx = T._free_coords(len(X), free)
        base = X.copy()

        def f(v):
            Y = base.ravel().copy(); Y[idx] = v; E, G = fn(Y.reshape(-1, 3))
            return E, G.ravel()[idx]

        def keep(v):
            Y = base.ravel().copy(); Y[idx] = v
            res[name] = dict(current=Y.reshape(-1, 3).tolist()); save(res)
        t0 = time.time()
        out = minimize(f, X.ravel()[idx], jac=True, method='L-BFGS-B', callback=keep,
                       # ftol tiny: stop on the gradient (gtol), not on the relative energy
                       # change, which for |E| ~ 3000 Eh would allow ~1e-5 Eh
                       options=dict(maxiter=600, gtol=3e-4, ftol=1e-15, maxls=40))
        Y = base.ravel().copy(); Y[idx] = out.x; Xm = Y.reshape(-1, 3)
        Em, Gm = fn(Xm)
        gmax = float(np.abs(Gm.ravel()[idx]).max()); grms = float(np.sqrt(np.mean(Gm.ravel()[idx] ** 2)))
        res[name] = dict(X=Xm.tolist(), E=float(Em), success=bool(out.success), nit=int(out.nit),
                         message=str(out.message), max_grad=gmax, rms_grad=grms); save(res)
        log(f"ONIOM-OPT {name}: E={Em:.6f} after {out.nit} iterations, max|g| {gmax:.1e} "
            f"rms {grms:.1e} Eh/A ({out.message}) ({time.time() - t0:.0f}s)")

    # --- partial Hessians
    Xts_xtb = np.array(r['ts']['coords'])
    region = [a for a in free if np.linalg.norm(Xts_xtb[a] - Xts_xtb[k['Hd']]) < REGION_R]
    res['config']['hessian_region'] = region
    ridx = T._free_coords(len(Xts_xtb), region)
    h = 0.005
    for name in NAMES:
        hs = res.setdefault('hess', {}).setdefault(name, dict(rows=[]))
        if 'H' in hs:
            continue
        X = np.array(res[name]['X']); x0 = X.ravel()
        t0 = time.time()
        for j in range(len(hs['rows']), len(ridx)):
            c = ridx[j]
            xp, xm = x0.copy(), x0.copy(); xp[c] += h; xm[c] -= h
            gp = fn(xp.reshape(-1, 3))[1].ravel()[ridx]; gm = fn(xm.reshape(-1, 3))[1].ravel()[ridx]
            hs['rows'].append(((gp - gm) / (2 * h)).tolist()); save(res)
        Hm = np.array(hs['rows']); hs['H'] = (0.5 * (Hm + Hm.T)).tolist(); del hs['rows']; save(res)
        log(f"ONIOM-OPT hessian {name}: {len(ridx)} coordinates ({time.time() - t0:.0f}s)")

    # --- kinetic isotope effects at the ONIOM(def2-SVP) level
    if 'analysis' not in res:
        # a saddle point has exactly one imaginary mode and minima none; anything else
        # means an optimisation stopped short or the region misses the reaction mode
        m = [T.AMU[s_] for s_ in sym]
        n_im = {n: int(np.sum(T.frequencies(np.array(res[n]['X']), np.array(res['hess'][n]['H']), m,
                                            free=region) < -50)) for n in NAMES}
        if n_im != dict(reactant=0, ts=1, product=0):
            raise RuntimeError(f'imaginary modes (< -50 cm-1) per structure {n_im}; expected '
                               f'reactant 0, ts 1, product 0 - not computing KIEs')
        coords = {n: res[n]['X'] for n in NAMES}
        out, dE, dEr, kie = S.kie_analysis(sym, k, region, coords, {n: res['hess'][n]['H'] for n in NAMES},
                                           {n: res[n]['E'] for n in NAMES})
        res['analysis'] = dict(iso=out, dE_barrier=dE, dE_rxn=dEr, kie=kie); save(res)
        TS = out['ts']['H']
        log(f"ONIOM-OPT barrier {dE:.2f}, reaction {dEr:+.2f} kcal/mol ({XC}/{BASIS}:xTB); "
            f"TS imag {TS['imag']:.0f} cm-1, n_imag {TS['n_imag']}; reactant n_imag "
            f"{out['reactant']['H']['n_imag']}, product n_imag {out['product']['H']['n_imag']}")
        for kk, v in kie.items():
            log(f"ONIOM-OPT KIE {kk}: semiclassical {v['semiclassical']:.2f}, Wigner {v['wigner']:.2f}, "
                f"Eckart {v['eckart']:.2f}   [experiment H/T 6.91 +- 0.05 at 293 K]")

    # --- final energies with def2-TZVP for the fragment
    sp = res.setdefault('sp', {})
    for name in NAMES:
        if name in sp:
            continue
        from pyscf import gto, dft
        X = np.array(res[name]['X'])
        fsym, fX = O.fragment(sym, X, atoms, cuts)
        mol = gto.M(atom=[(s_, x_) for s_, x_ in zip(fsym, fX)], basis=BASIS_SP, charge=0,
                    unit='Angstrom', verbose=0)
        mf = dft.RKS(mol, xc=XC); mf.conv_tol = 1e-10
        e_hi = float(mf.kernel())
        sp[name] = dict(e_xtb_cluster=float(energy_grad(sym, X, p['charge'])[0]),
                        e_xtb_frag=float(energy_grad(fsym, fX, 0, solvent=None)[0]),
                        e_high=e_hi, converged=bool(mf.converged))
        sp[name]['E'] = sp[name]['e_xtb_cluster'] + e_hi - sp[name]['e_xtb_frag']
        save(res)
    if 'summary' not in res:
        E = {n: sp[n]['E'] for n in NAMES}
        a = res['analysis']['iso']
        zpe = lambda n: a[n]['H']['zpe']
        dE = (E['ts'] - E['reactant']) * S.HARTREE_KCAL
        dEr = (E['product'] - E['reactant']) * S.HARTREE_KCAL
        dZ = (zpe('ts') - zpe('reactant')) * S.HARTREE_KCAL
        res['summary'] = dict(barrier_tzvp=dE, reaction_tzvp=dEr, barrier_tzvp_zpe=dE + dZ); save(res)
        log(f"ONIOM-OPT barrier {dE:.2f} kcal/mol ({XC}/{BASIS_SP}:xTB at the optimised geometries), "
            f"{dE + dZ:.2f} with ZPE; reaction {dEr:+.2f} kcal/mol")


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--variant' and sys.argv[2] != 'default':
        SUFFIX = f'.{sys.argv[2]}'
        OUT = OUT.with_name(f'ts_oniom_opt{SUFFIX}.json')
    main()
