"""Full thymidylate synthase cluster pipeline -- intended for the NUC, not this container.

Stages (each checkpointed in results/ts_cluster_run.json; re-running resumes):

  0 prep      build the 96-atom cluster from 1TLS, relax H, relax with caps frozen
  1 scan      GFN2-xTB relaxed scan of xi = d(Cd-H) - d(H-C11), caps frozen
  2 ts        P-RFO saddle search at the scan maximum (xTB), then a DFT saddle
              search started from it if --dft-ts is given
  3 hess      finite-difference Hessian at reactant / TS / product
  4 kie       isotopologues H/D/T, semiclassical TST + Wigner + Eckart tunnelling
  5 dft       B3LYP-D3(BJ)/def2-SVP (then def2-TZVP) single points, C-PCM eps=4
  6 active    CCSD(T) and chemoqbit circuits on a small active space around the
              transferred hydride (donor C, H, acceptor C, plus S if requested)

Usage on the NUC (inside the repo, with the venv built as in README):

    OMP_NUM_THREADS=8 .venv/bin/python scripts/ts_cluster_nuc.py --stages 0,1,2,3,4
    OMP_NUM_THREADS=8 .venv/bin/python scripts/ts_cluster_nuc.py --stages 5,6 --basis def2-svp

Rough cost on 8 cores: stages 0-4 a few hours (xTB); stage 5 about 20 min per
point at def2-SVP for 96 atoms; stage 6 minutes.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit.enzyme.cluster import build, add_hydrogens
from chemoqbit.enzyme.xtb import energy_grad
from chemoqbit.enzyme.frozen import optimise_frozen
from chemoqbit.enzyme import tsopt as T
from chemoqbit import kinetics as K
from chemoqbit.eckart import eckart_kappa
from chemoqbit.chem import HARTREE_KCAL

OUT = ROOT / 'results' / 'ts_cluster_run.json'
TEMP = 293.15
CAP_FREEZE = ('tyr94:CG', 'his147:CB', 'asn177:CB', 'arg166:CD')


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(r):
    OUT.write_text(json.dumps(r, indent=1, default=float))


def log(*a):
    print(*a, flush=True)


def stage_prep(r):
    sym0, X0, label, _ = build(ROOT / 'data/1TLS.pdb')
    mh = add_hydrogens(sym0, X0, label, charge=1)
    conf = mh.GetConformer()
    sym = [a.GetSymbol() for a in mh.GetAtoms()]
    X = np.array([list(conf.GetAtomPosition(i)) for i in range(mh.GetNumAtoms())])
    lab = label + [f'H{i}' for i in range(len(sym) - len(label))]
    idx = {l: i for i, l in enumerate(label)}
    heavy = np.array([s != 'H' for s in sym])
    X, _ = optimise_frozen(sym, X, 1, frozen=np.flatnonzero(heavy))   # H only
    frozen = [i for i, l in enumerate(lab) if ':cap' in l] + [lab.index(n) for n in CAP_FREEZE]
    X, E = optimise_frozen(sym, X, 1, frozen=frozen)
    d = lambda a, b: float(np.linalg.norm(X[a] - X[b]))
    Cd, C11 = idx['folate:C6'], idx['methylene:C11']
    Hd = min((i for i in range(len(sym)) if sym[i] == 'H' and d(i, Cd) < 1.25),
             key=lambda i: d(i, C11))
    r['prep'] = dict(symbols=sym, coords=X.tolist(), labels=lab, charge=1, E=E,
                     frozen=frozen,
                     key=dict(Cd=Cd, Hd=int(Hd), C11=C11, C5u=idx['nucleotide:C5'],
                              C6u=idx['nucleotide:C6'], S=idx['cys146:SG'],
                              N5=idx['folate:N5'], Cg=idx['arg166:CZ']))
    log(f'prep: {len(sym)} atoms, E={E:.6f}, DAD={d(Cd, C11):.2f} A, H..C11={d(Hd, C11):.2f} A')
    return r


def _ctx(r):
    p = r['prep']
    sym, X, k = p['symbols'], np.array(p['coords']), p['key']
    fn = lambda Y: energy_grad(sym, Y, p['charge'])
    return sym, X, k, p['frozen'], fn


def stage_scan(r, lo=-1.4, hi=1.4, step=0.1):
    sym, X, k, frozen, fn = _ctx(r)
    pts = r.get('scan', [])
    grid = np.round(np.arange(lo, hi + 1e-9, step), 2)
    if pts:
        X = np.array(pts[-1]['coords'])
    for xi in grid[len(pts):]:
        t0 = time.time()
        X, E = optimise_frozen(sym, X, 1, frozen=frozen,
                               constraints=[(k['Cd'], k['Hd'], k['Hd'], k['C11'], float(xi))])
        d = lambda a, b: float(np.linalg.norm(X[a] - X[b]))
        pts.append(dict(xi=float(xi), E=float(E), coords=X.tolist(),
                        CdH=d(k['Cd'], k['Hd']), HC11=d(k['Hd'], k['C11']),
                        DAD=d(k['Cd'], k['C11']), C6uS=d(k['C6u'], k['S'])))
        r['scan'] = pts; save(r)
        log(f"xi={xi:+.2f} E={E:.6f} ({(E - pts[0]['E']) * HARTREE_KCAL:+6.2f} kcal/mol) "
            f"CdH={pts[-1]['CdH']:.2f} H..C11={pts[-1]['HC11']:.2f} DAD={pts[-1]['DAD']:.2f} "
            f"C6u-S={pts[-1]['C6uS']:.2f} ({time.time() - t0:.0f}s)")
    return r


def stage_ts(r):
    sym, X0, k, frozen, fn = _ctx(r)
    pts = r['scan']
    i = int(np.argmax([p['E'] for p in pts]))
    Xg = np.array(pts[i]['coords'])
    guide = np.array(pts[min(i + 1, len(pts) - 1)]['coords']) - np.array(pts[max(i - 1, 0)]['coords'])
    log(f"TS guess at xi={pts[i]['xi']}")
    Xts, Ets, info = T.prfo_follow(Xg, fn, guide, grad_fn=lambda Y: fn(Y)[1], log=log)
    r['ts'] = dict(coords=Xts.tolist(), E=float(Ets), prfo=info); save(r)
    # end points: relax from the two ends of the scan
    for name, j in (('reactant', 0), ('product', len(pts) - 1)):
        Xm, Em = optimise_frozen(sym, np.array(pts[j]['coords']), 1, frozen=frozen)
        r[name] = dict(coords=Xm.tolist(), E=float(Em)); save(r)
        log(f'{name}: E={Em:.6f}')
    return r


def stage_hess(r):
    sym, _, k, frozen, fn = _ctx(r)
    gfn = lambda Y: fn(Y)[1]
    for name in ('reactant', 'ts', 'product'):
        if 'hessian' in r[name]:
            continue
        t0 = time.time()
        r[name]['hessian'] = T.fd_hessian(np.array(r[name]['coords']), gfn).tolist()
        save(r); log(f'{name} Hessian ({time.time() - t0:.0f}s)')
    return r


def stage_kie(r):
    sym, _, k, _, _ = _ctx(r)
    masses = {i: [T.AMU[s] for s in sym] for i in ('H', 'D', 'T')}
    for iso in ('D', 'T'):
        masses[iso][k['Hd']] = T.AMU[iso]
    out = {}
    for name in ('reactant', 'ts', 'product'):
        X = np.array(r[name]['coords']); H = np.array(r[name]['hessian'])
        out[name] = {}
        for iso in ('H', 'D', 'T'):
            nu = T.frequencies(X, H, masses[iso])
            real = nu[nu > 0]
            out[name][iso] = dict(n_imag=int(np.sum(nu < -50)),
                                  imag=float(nu[0]) if nu[0] < 0 else None,
                                  zpe=T.zpe(real), lnq=T.ln_q_vib_rot(X, real, masses[iso], TEMP))
    R, TS, P = out['reactant'], out['ts'], out['product']
    dE = (r['ts']['E'] - r['reactant']['E']) * HARTREE_KCAL
    dEr = (r['product']['E'] - r['reactant']['E']) * HARTREE_KCAL
    kie = {}
    for iso in ('D', 'T'):
        sc = float(np.exp((TS['H']['lnq'] - R['H']['lnq']) - (TS[iso]['lnq'] - R[iso]['lnq'])))
        kap = {}
        for lab in ('H', iso):
            V1 = dE + (TS[lab]['zpe'] - R[lab]['zpe']) * HARTREE_KCAL
            V2 = dEr + (P[lab]['zpe'] - R[lab]['zpe']) * HARTREE_KCAL
            nu = abs(TS[lab]['imag'])
            kap[lab] = dict(wigner=K.wigner(nu, TEMP),
                            eckart=eckart_kappa(nu, V1, min(V2, V1 - 0.1), TEMP)[0], nu=nu, V1=V1)
        kie[f'H/{iso}'] = dict(semiclassical=sc,
                               wigner=sc * kap['H']['wigner'] / kap[iso]['wigner'],
                               eckart=sc * kap['H']['eckart'] / kap[iso]['eckart'], kappa=kap)
    r['analysis'] = dict(T=TEMP, dE_barrier=dE, dE_rxn=dEr, iso=out, kie=kie)
    save(r)
    log(f"barrier {dE:.2f}, reaction {dEr:.2f} kcal/mol; TS imag {TS['H']['imag']:.0f} cm-1 "
        f"(n_imag {TS['H']['n_imag']})")
    for kk, v in kie.items():
        log(f"KIE {kk}: semiclassical {v['semiclassical']:.2f}  Wigner {v['wigner']:.2f}  "
            f"Eckart {v['eckart']:.2f}   [experiment H/T = 6.91 +- 0.05 at 293 K]")
    return r


def stage_dft(r, basis='def2-svp', xc='b3lyp'):
    from pyscf import gto, dft as pdft
    sym = r['prep']['symbols']
    r.setdefault('dft', {})
    for name in ('reactant', 'ts', 'product'):
        tag = f'{name}/{basis}'
        if tag in r['dft']:
            continue
        t0 = time.time()
        X = np.array(r[name]['coords'])
        atom = '\n'.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in zip(sym, X))
        mol = gto.M(atom=atom, basis=basis, charge=1, verbose=0)
        mf = pdft.RKS(mol); mf.xc = xc; mf.disp = 'd3bj'
        mf = mf.PCM(); mf.with_solvent.eps = 4.0
        mf.conv_tol = 1e-8
        mf.kernel()
        r['dft'][tag] = dict(E=float(mf.e_tot), converged=bool(mf.converged),
                             nao=int(mol.nao), seconds=time.time() - t0)
        save(r); log(f'DFT {tag}: E={mf.e_tot:.6f} conv={mf.converged} ({time.time() - t0:.0f}s)')
    b = r['dft']
    if all(f'{n}/{basis}' in b for n in ('reactant', 'ts', 'product')):
        log(f"DFT barrier {(b[f'ts/{basis}']['E'] - b[f'reactant/{basis}']['E']) * HARTREE_KCAL:.2f} "
            f"kcal/mol, reaction "
            f"{(b[f'product/{basis}']['E'] - b[f'reactant/{basis}']['E']) * HARTREE_KCAL:.2f}")
    return r


def stage_active(r, ne=4, no=4, basis='cc-pvdz', radius=2.6):
    """Circuits on a small active space around the transferring hydride.

    The QM region is the donor C, the hydride, the acceptor C and whatever
    heavy atoms lie within `radius` of the hydride, capped with H.
    """
    from chemoqbit.chem import build_problem, run_vqe, run_adapt
    sym = r['prep']['symbols']; k = r['prep']['key']
    r.setdefault('active', {})
    for name in ('reactant', 'ts', 'product'):
        tag = f'{name}/{ne}e{no}o'
        if tag in r['active']:
            continue
        X = np.array(r[name]['coords'])
        h = X[k['Hd']]
        core = [i for i in range(len(sym)) if np.linalg.norm(X[i] - h) < radius]
        for idx_ in (k['Cd'], k['Hd'], k['C11']):
            if idx_ not in core:
                core.append(idx_)
        atom = '\n'.join(f'{sym[i]} {X[i][0]:.6f} {X[i][1]:.6f} {X[i][2]:.6f}' for i in core)
        nel = sum({'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16}[sym[i]] for i in core)
        charge = 0 if nel % 2 == 0 else 1
        t0 = time.time()
        p = build_problem(atom, basis, no, ne, with_ccsd_t=True, charge=charge)
        res = {}
        for m, runner in (('uccsd', run_vqe), ('adapt', run_adapt)):
            o = runner(p)
            res[m] = dict(e_vqe=o.energy, e_vqe_fixed=o.fixed_energy, err_mha=o.err_mha,
                          n_params=o.n_params, cnots=o.gate_counts.get('CNOT', 0))
        r['active'][tag] = dict(atoms=core, n_atoms=len(core), charge=charge,
                                e_hf=p.e_hf, e_ccsd_t=p.e_ccsd_t, e_casci=p.e_casci,
                                n_qubits=p.n_qubits, **res, seconds=time.time() - t0)
        save(r)
        log(f"active {tag}: {len(core)} atoms, {p.n_qubits} qubits, "
            f"CASCI={p.e_casci:.6f}, ADAPT err={res['adapt']['err_mha']:.2e} mHa "
            f"({time.time() - t0:.0f}s)")
    return r


STAGES = dict(prep=stage_prep, scan=stage_scan, ts=stage_ts, hess=stage_hess,
              kie=stage_kie, dft=stage_dft, active=stage_active)
ORDER = ['prep', 'scan', 'ts', 'hess', 'kie', 'dft', 'active']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', default='0,1,2,3,4',
                    help='comma list of stage numbers or names (0=prep ... 6=active)')
    ap.add_argument('--basis', default='def2-svp')
    args = ap.parse_args()
    sel = [ORDER[int(s)] if s.strip().isdigit() else s.strip() for s in args.stages.split(',')]
    r = load()
    for s in sel:
        log(f'=== stage {s} ===')
        r = stage_dft(r, basis=args.basis) if s == 'dft' else STAGES[s](r)
        save(r)
    log('done')
