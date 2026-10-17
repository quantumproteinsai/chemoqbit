"""Thymidylate synthase hydride transfer, reduced model, GFN2-xTB level.

1. fine scan near the maximum of the anti-face scan -> TS guess
2. finite-difference Hessian, geomeTRIC saddle optimisation
3. reactant and product complexes (free optimisation from the scan ends)
4. Hessians, isotope substitution of the transferred hydride (H, D, T)
5. semiclassical TST KIE (Bigeleisen via partition functions) and
   tunnelling-corrected KIE (Wigner, Eckart) at 293.15 K
-> results/ts_kie.json (checkpointed per stage)
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit.enzyme.build import to_xyz
from chemoqbit.enzyme.xtb import energy_grad, optimise
from chemoqbit.enzyme import tsopt as T
from chemoqbit import kinetics as K
from chemoqbit.eckart import eckart_kappa
from chemoqbit.chem import HARTREE_KCAL

OUT = ROOT / 'results' / 'ts_kie.json'
res = json.loads(OUT.read_text()) if OUT.exists() else {}
save = lambda: OUT.write_text(json.dumps(res, indent=1, default=float))
scan = json.loads((ROOT / 'results' / 'ts_scan.json').read_text())['anti']
key, q = scan['key'], scan['charge']
read_xyz = lambda s: (s.split('\n')[2:][:-1], )
parse = lambda s: ([l.split()[0] for l in s.strip().split('\n')[2:]],
                   np.array([[float(v) for v in l.split()[1:4]] for l in s.strip().split('\n')[2:]]))
sym, _ = parse(scan['points'][0]['xyz'])
fn = lambda X: energy_grad(sym, X, q)
gfn = lambda X: energy_grad(sym, X, q)[1]
log = lambda s: print(s, flush=True)
TEMP = 293.15

# 1 -- scan of xi = d(Cd-H) - d(H-C7) from the reactant complex (smooth hydride path)
if 'xi_scan' not in res:
    _, X = parse(scan['points'][0]['xyz'])
    pts = []
    for xi in np.round(np.arange(-1.6, 1.61, 0.1), 2):
        X, E = optimise(sym, X, q, constraints=[(key['Cd'], key['Hd'], key['Hd'], key['C7'], float(xi))], k=20.0)
        dd = lambda a, b: float(np.linalg.norm(X[key[a]] - X[key[b]]))
        pts.append(dict(xi=float(xi), E=float(E), CdH=dd('Cd', 'Hd'), HC7=dd('Hd', 'C7'),
                        C6S=dd('C6', 'S'), xyz=to_xyz(sym, X)))
        log(f'xi={xi:+.2f} E={E:.6f} CdH={pts[-1]["CdH"]:.3f} HC7={pts[-1]["HC7"]:.3f} C6S={pts[-1]["C6S"]:.3f}')
    res['xi_scan'] = pts; save()

# 2 -- saddle point: P-RFO following the scan tangent at the maximum
if 'ts' not in res:
    pts = res['xi_scan']
    imax = int(np.argmax([p['E'] for p in pts]))
    _, X0 = parse(pts[imax]['xyz'])
    _, Xa = parse(pts[max(imax - 1, 0)]['xyz'])
    _, Xb = parse(pts[min(imax + 1, len(pts) - 1)]['xyz'])
    guide = Xb - Xa
    log(f'TS guess at xi={pts[imax]["xi"]}, following scan tangent')
    Xts, Ets, info = T.prfo_follow(X0, fn, guide, grad_fn=gfn, log=log)
    res['ts'] = dict(xyz=to_xyz(sym, Xts), E=float(Ets), prfo=info); save()
    log(f'TS E={Ets:.6f}')

# 3 -- minima
for name, src in (('reactant', scan['points'][0]['xyz']), ('product', scan['points'][-1]['xyz'])):
    if name not in res:
        _, X0 = parse(src)
        Xm, Em = T.run(sym, X0, fn)
        res[name] = dict(xyz=to_xyz(sym, Xm), E=float(Em)); save()
        log(f'{name} E={Em:.6f}')

# 4 -- Hessians and frequencies
for name in ('reactant', 'ts', 'product'):
    if 'hessian' not in res[name]:
        _, X = parse(res[name]['xyz'])
        res[name]['hessian'] = T.fd_hessian(X, gfn).tolist(); save()
        log(f'{name} Hessian done')

# 5 -- isotopologues and KIE
masses = {iso: [T.AMU[s] for s in sym] for iso in ('H', 'D', 'T')}
for iso in ('D', 'T'):
    masses[iso][key['Hd']] = T.AMU[iso]
out = {}
for name in ('reactant', 'ts', 'product'):
    _, X = parse(res[name]['xyz'])
    Hs = np.array(res[name]['hessian'])
    out[name] = {}
    for iso in ('H', 'D', 'T'):
        nu = T.frequencies(X, Hs, masses[iso])
        real = nu[nu > 0]
        out[name][iso] = dict(n_imag=int(np.sum(nu < -50)), imag=float(nu[0]) if nu[0] < 0 else None,
                              lowest=[float(v) for v in nu[:4]], zpe=T.zpe(real),
                              lnq=T.ln_q_vib_rot(X, real, masses[iso], TEMP))
R, TS, P = out['reactant'], out['ts'], out['product']
dE = (res['ts']['E'] - res['reactant']['E']) * HARTREE_KCAL
dErxn = (res['product']['E'] - res['reactant']['E']) * HARTREE_KCAL
kie = {}
for iso in ('D', 'T'):
    ln_sc = (TS['H']['lnq'] - R['H']['lnq']) - (TS[iso]['lnq'] - R[iso]['lnq'])
    kie_sc = float(np.exp(ln_sc))
    k = {}
    for lab in ('H', iso):
        V1 = dE + (TS[lab]['zpe'] - R[lab]['zpe']) * HARTREE_KCAL
        V2 = dErxn + (P[lab]['zpe'] - R[lab]['zpe']) * HARTREE_KCAL
        nu = abs(TS[lab]['imag'])
        k[lab] = dict(wigner=K.wigner(nu, TEMP), eckart=eckart_kappa(nu, V1, max(V2, -V1 + 0.1), TEMP)[0],
                      V1_zpe=V1, nu=nu)
    kie[f'H/{iso}'] = dict(semiclassical=kie_sc,
                           wigner=kie_sc * k['H']['wigner'] / k[iso]['wigner'],
                           eckart=kie_sc * k['H']['eckart'] / k[iso]['eckart'], kappa=k)
res['analysis'] = dict(T=TEMP, dE_barrier=dE, dE_rxn=dErxn, isotopologues=out, kie=kie)
save()
log(f'barrier {dE:.2f} kcal/mol, reaction {dErxn:.2f}; TS imag {TS["H"]["imag"]:.0f} cm-1 '
    f'(n_imag={TS["H"]["n_imag"]}); reactant n_imag={R["H"]["n_imag"]}')
for kk, v in kie.items():
    log(f'KIE {kk}: semiclassical {v["semiclassical"]:.2f}  Wigner {v["wigner"]:.2f}  Eckart {v["eckart"]:.2f}')
