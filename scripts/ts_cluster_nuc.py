"""Full thymidylate synthase cluster pipeline -- intended for the NUC, not this container.

Stages (each checkpointed in results/ts_cluster_run.json; re-running resumes):

  0 prep      build the 96-atom cluster from 1TLS, relax H, relax with caps frozen
              and folate N5 held >= N5_C11_MIN from C11 so the exocyclic
              methylene (C5=C11) survives. The hold stays on in every later
              stage: the methylene is a minimum on its own, but pulling the
              folate in during the scan pushes N5 back onto C11 (the enzyme
              prevents this by a conformational change the cluster lacks)
  1 scan      GFN2-xTB relaxed scan of xi = d(Cd-H) - d(H-C11), caps frozen;
              stops on an energy jump or a bond change outside the reaction
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
from chemoqbit.enzyme.frozen import optimise_frozen, penalty
from chemoqbit.enzyme import tsopt as T
from chemoqbit import kinetics as K
from chemoqbit.eckart import eckart_kappa
from chemoqbit.chem import HARTREE_KCAL

OUT = ROOT / 'results' / 'ts_cluster_run.json'
# model variants (--variant): cluster charge, His147 protonation, checkpoint suffix
VARIANTS = {'default': dict(charge=1, his_protonated=False, suffix=''),
            'his147p': dict(charge=2, his_protonated=True, suffix='.his147p'),
            'o4h': dict(charge=2, his_protonated=False, o4_protonated=True, suffix='.o4h')}
VARIANT = VARIANTS['default']
TEMP = 293.15
CAP_FREEZE = ('tyr94:CG', 'his147:CB', 'asn177:CB', 'arg166:CD')
N5_C11_MIN = 2.8          # A; flat-bottom, no force beyond it
MAX_JUMP = 12.0           # kcal/mol between neighbouring scan points
RCOV = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05}


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(r):
    OUT.write_text(json.dumps(r, indent=1, default=float))


def log(*a):
    print(*a, flush=True)


def stage_prep(r):
    if 'restraints' in r.get('prep', {}):
        log('prep: already done'); return r
    sym0, X0, label, _ = build(ROOT / 'data/1TLS.pdb')
    mh = add_hydrogens(sym0, X0, label, charge=VARIANT['charge'], his_protonated=VARIANT['his_protonated'],
                       o4_protonated=VARIANT.get('o4_protonated', False))
    conf = mh.GetConformer()
    sym = [a.GetSymbol() for a in mh.GetAtoms()]
    X = np.array([list(conf.GetAtomPosition(i)) for i in range(mh.GetNumAtoms())])
    lab = label + [f'H{i}' for i in range(len(sym) - len(label))]
    idx = {l: i for i, l in enumerate(label)}
    heavy = np.array([s != 'H' for s in sym])
    q = VARIANT['charge']
    X, _ = optimise_frozen(sym, X, q, frozen=np.flatnonzero(heavy))   # H only
    frozen = [i for i, l in enumerate(lab) if ':cap' in l] + [lab.index(n) for n in CAP_FREEZE]
    # add_hydrogens cuts N5-C11 only in the topology; C11 still sits at the crystal
    # 1.5 A from N5, and an unrestrained xTB relaxation simply re-forms the bond
    N5, C11 = idx['folate:N5'], idx['methylene:C11']
    restraints = [('min', N5, C11, N5_C11_MIN)]
    X, E = optimise_frozen(sym, X, q, frozen=frozen, constraints=restraints)
    d = lambda a, b: float(np.linalg.norm(X[a] - X[b]))
    Cd = idx['folate:C6']
    Hd = min((i for i in range(len(sym)) if sym[i] == 'H' and d(i, Cd) < 1.25),
             key=lambda i: d(i, C11))
    r['prep'] = dict(symbols=sym, coords=X.tolist(), labels=lab, charge=q, E=E,
                     frozen=frozen, restraints=restraints,
                     key=dict(Cd=Cd, Hd=int(Hd), C11=C11, C5u=idx['nucleotide:C5'],
                              C6u=idx['nucleotide:C6'], S=idx['cys146:SG'],
                              N5=idx['folate:N5'], Cg=idx['arg166:CZ']))
    log(f'prep: {len(sym)} atoms, E={E:.6f}, DAD={d(Cd, C11):.2f} A, H..C11={d(Hd, C11):.2f} A, '
        f'N5-C11={d(N5, C11):.2f} A')
    return r


def _ctx(r):
    """symbols, prep coords, key atoms, frozen atoms, prep restraints, and the
    energy/gradient function with those restraints included"""
    p = r['prep']
    sym, X, k = p['symbols'], np.array(p['coords']), p['key']
    rest = [tuple(c) for c in p.get('restraints', [])]

    def fn(Y):
        E, G = energy_grad(sym, Y, p['charge'])
        Ep, Gp = penalty(Y, rest)
        return E + Ep, G + Gp
    return sym, X, k, p['frozen'], rest, fn


def _free(sym, frozen):
    """atoms not held fixed; Hessians and frequencies cover only these"""
    return [i for i in range(len(sym)) if i not in set(frozen)]


def _bonds(sym, X):
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    R = np.array([RCOV[s] for s in sym])
    i, j = np.nonzero(np.triu(D < 1.2 * (R[:, None] + R[None]), 1))
    return set(zip(i.tolist(), j.tolist()))


def _check_step(sym, prev, X, E, k, labels, X_prep):
    """refuse a scan point whose energy jumps from the previous point, or whose
    bonding differs from the previous point (or from prep, for the first point)
    anywhere other than the hydride (Hd) and the Cys146 S-C6 bond"""
    if prev is not None:
        dE = (E - prev['E']) * HARTREE_KCAL
        if abs(dE) > MAX_JUMP:
            raise RuntimeError(f'scan: energy jumps {dE:+.1f} kcal/mol from xi={prev["xi"]}; '
                               f'the relaxation has left the reaction path')
        if dE < -3.0:
            log(f'  note: {dE:+.1f} kcal/mol drop from xi={prev["xi"]} (conformational relaxation?)')
    dH = min(np.linalg.norm(X[k['Hd']] - X[k['Cd']]), np.linalg.norm(X[k['Hd']] - X[k['C11']]))
    if dH > 1.6:
        raise RuntimeError(f'scan: the hydride is {dH:.2f} A from both donor and acceptor')
    Xref = X_prep if prev is None else np.array(prev['coords'])
    allowed = {k['Hd']}
    changed = [(a, b) for a, b in _bonds(sym, Xref) ^ _bonds(sym, X)
               if a not in allowed and b not in allowed and {a, b} != {k['C6u'], k['S']}]
    if changed:
        raise RuntimeError('scan: bonds changed outside the reaction: '
                           + ', '.join(f'{labels[a]}-{labels[b]}' for a, b in changed))


def stage_scan(r, lo=-1.4, hi=1.4, step=0.1):
    sym, X, k, frozen, rest, _ = _ctx(r)
    pts = r.get('scan', [])
    grid = np.round(np.arange(lo, hi + 1e-9, step), 2)
    if pts:
        X = np.array(pts[-1]['coords'])
    else:
        # approach the first grid point in 0.2 A steps: a single jump from the
        # relaxed reactant (xi ~ -2.7) puts a large force on the restraint, and the
        # optimiser's first step can land where the xTB SCF does not converge
        xi0 = float(np.linalg.norm(X[k['Cd']] - X[k['Hd']]) - np.linalg.norm(X[k['Hd']] - X[k['C11']]))
        for t in np.arange(xi0 + 0.2, lo, 0.2):
            X, _ = optimise_frozen(sym, X, r['prep']['charge'], frozen=frozen,
                                   constraints=rest + [(k['Cd'], k['Hd'], k['Hd'], k['C11'], float(t))])
            log(f'  ramp to the first scan point: xi={t:+.2f}')
    for xi in grid[len(pts):]:
        t0 = time.time()
        X, E = optimise_frozen(sym, X, r['prep']['charge'], frozen=frozen,
                               constraints=rest + [(k['Cd'], k['Hd'], k['Hd'], k['C11'], float(xi))])
        _check_step(sym, pts[-1] if pts else None, X, E, k, r['prep']['labels'],
                    np.array(r['prep']['coords']))
        d = lambda a, b: float(np.linalg.norm(X[a] - X[b]))
        pts.append(dict(xi=float(xi), E=float(E), coords=X.tolist(),
                        CdH=d(k['Cd'], k['Hd']), HC11=d(k['Hd'], k['C11']),
                        DAD=d(k['Cd'], k['C11']), C6uS=d(k['C6u'], k['S'])))
        r['scan'] = pts; save(r)
        log(f"xi={xi:+.2f} E={E:.6f} ({(E - pts[0]['E']) * HARTREE_KCAL:+6.2f} kcal/mol) "
            f"CdH={pts[-1]['CdH']:.2f} H..C11={pts[-1]['HC11']:.2f} DAD={pts[-1]['DAD']:.2f} "
            f"C6u-S={pts[-1]['C6uS']:.2f} ({time.time() - t0:.0f}s)")
    if len(pts) == len(grid) and not r.get('scan_backfilled'):
        backfill_scan(r)
        r['scan_backfilled'] = True; save(r)
    return r


def backfill_scan(r, tol=0.1, start=None):
    """Sweep back from the last scan point (or `start`), re-optimising each
    earlier point from its neighbour's geometry and keeping it if lower by more
    than `tol` kcal/mol. Removes the steps that conformational drops part-way
    along the forward scan leave in the profile, including a spurious maximum
    just before one. Modifies r['scan'] in place."""
    sym, _, k, frozen, rest, _ = _ctx(r)
    pts = r['scan']
    E = [p['E'] for p in pts]
    j = len(pts) - 1 if start is None else start
    replaced = []
    for i in range(j - 1, -1, -1):
        t0 = time.time()
        X, Ei = optimise_frozen(sym, np.array(pts[i + 1]['coords']), r['prep']['charge'], frozen=frozen,
                                constraints=rest + [(k['Cd'], k['Hd'], k['Hd'], k['C11'], pts[i]['xi'])])
        dE = (Ei - pts[i]['E']) * HARTREE_KCAL
        log(f"backfill xi={pts[i]['xi']:+.2f} E={Ei:.6f} ({dE:+.2f} vs forward) ({time.time() - t0:.0f}s)")
        if dE > -tol:
            continue
        _check_step(sym, pts[i + 1], X, Ei, k, r['prep']['labels'], None)
        d = lambda a, b: float(np.linalg.norm(X[a] - X[b]))
        pts[i] = dict(xi=pts[i]['xi'], E=float(Ei), coords=X.tolist(),
                      CdH=d(k['Cd'], k['Hd']), HC11=d(k['Hd'], k['C11']),
                      DAD=d(k['Cd'], k['C11']), C6uS=d(k['C6u'], k['S']), backfilled=True)
        replaced.append(pts[i]['xi'])
    log(f'backfill: replaced {len(replaced)} points {replaced}')
    return replaced


def stage_ts(r):
    if all(n in r for n in ('ts', 'reactant', 'product')):
        log('ts: already done'); return r
    sym, X0, k, frozen, rest, fn = _ctx(r)
    pts = r['scan']
    i = int(np.argmax([p['E'] for p in pts]))
    if i in (0, len(pts) - 1):
        raise RuntimeError(f"scan maximum is at the end point xi={pts[i]['xi']}; widen the scan")
    Xg = np.array(pts[i]['coords'])
    # follow the hydride coordinate itself, d xi / dX: the difference of
    # neighbouring scan points is dominated by any cluster rearrangement
    u1 = Xg[k['Hd']] - Xg[k['Cd']]; u1 /= np.linalg.norm(u1)
    u2 = Xg[k['C11']] - Xg[k['Hd']]; u2 /= np.linalg.norm(u2)
    guide = np.zeros_like(Xg)
    guide[k['Cd']] -= u1; guide[k['Hd']] += u1 + u2; guide[k['C11']] -= u2
    log(f"TS guess at xi={pts[i]['xi']}")
    Xts, Ets, info = T.prfo_follow(Xg, fn, guide, grad_fn=lambda Y: fn(Y)[1], frozen=frozen, log=log)
    r['ts'] = dict(coords=Xts.tolist(), E=float(Ets), prfo=info); save(r)
    # end points: relax from the lowest scan point on each side of the maximum
    # (the cluster can drop into a lower conformer part-way along the scan)
    E = [p['E'] for p in pts]
    for name, j in (('reactant', int(np.argmin(E[:i]))), ('product', i + 1 + int(np.argmin(E[i + 1:])))):
        Xm, Em = optimise_frozen(sym, np.array(pts[j]['coords']), r['prep']['charge'], frozen=frozen,
                                 constraints=rest)
        r[name] = dict(coords=Xm.tolist(), E=float(Em)); save(r)
        log(f"{name}: E={Em:.6f} (from xi={pts[j]['xi']})")
    return r


def stage_hess(r):
    sym, _, k, frozen, _, fn = _ctx(r)
    gfn = lambda Y: fn(Y)[1]
    free = _free(sym, frozen)
    for name in ('reactant', 'ts', 'product'):
        if 'hessian' in r[name]:
            continue
        t0 = time.time()
        r[name]['hessian'] = T.fd_hessian(np.array(r[name]['coords']), gfn, free=free).tolist()
        save(r); log(f'{name} Hessian ({time.time() - t0:.0f}s)')
    return r


def kie_analysis(sym, k, free, coords, hessians, energies, T_K=None, rot=False):
    """Frequencies, ZPEs and H/D/T kinetic isotope effects (semiclassical TST,
    Wigner, Eckart) from reactant/ts/product `coords`, free-atom `hessians`
    (over the atoms in `free`) and `energies` (Eh). Returns (iso, dE, dEr, kie)."""
    T_K = TEMP if T_K is None else T_K
    masses = {i: [T.AMU[s] for s in sym] for i in ('H', 'D', 'T')}
    for iso in ('D', 'T'):
        masses[iso][k['Hd']] = T.AMU[iso]
    out = {}
    for name in ('reactant', 'ts', 'product'):
        X = np.array(coords[name]); H = np.array(hessians[name])
        out[name] = {}
        for iso in ('H', 'D', 'T'):
            nu = T.frequencies(X, H, masses[iso], free=free)
            real = nu[nu > 0]
            out[name][iso] = dict(n_imag=int(np.sum(nu < -50)),
                                  imag=float(nu[0]) if nu[0] < 0 else None,
                                  zpe=T.zpe(real),
                                  lnq=T.ln_q_vib_rot(X, real, masses[iso], T_K, rot=rot))
    R, TS, P = out['reactant'], out['ts'], out['product']
    dE = (energies['ts'] - energies['reactant']) * HARTREE_KCAL
    dEr = (energies['product'] - energies['reactant']) * HARTREE_KCAL
    kie = {}
    for iso in ('D', 'T'):
        sc = float(np.exp((TS['H']['lnq'] - R['H']['lnq']) - (TS[iso]['lnq'] - R[iso]['lnq'])))
        kap = {}
        for lab in ('H', iso):
            V1 = dE + (TS[lab]['zpe'] - R[lab]['zpe']) * HARTREE_KCAL
            V2 = dEr + (P[lab]['zpe'] - R[lab]['zpe']) * HARTREE_KCAL
            nu = abs(TS[lab]['imag'])
            kap[lab] = dict(wigner=K.wigner(nu, T_K),
                            eckart=eckart_kappa(nu, V1, min(V2, V1 - 0.1), T_K)[0], nu=nu, V1=V1)
        kie[f'H/{iso}'] = dict(semiclassical=sc,
                               wigner=sc * kap['H']['wigner'] / kap[iso]['wigner'],
                               eckart=sc * kap['H']['eckart'] / kap[iso]['eckart'], kappa=kap)
    return out, dE, dEr, kie


def stage_kie(r):
    sym, _, k, frozen, _, _ = _ctx(r)
    names = ('reactant', 'ts', 'product')
    out, dE, dEr, kie = kie_analysis(sym, k, _free(sym, frozen),
                                     {n: r[n]['coords'] for n in names},
                                     {n: r[n]['hessian'] for n in names},
                                     {n: r[n]['E'] for n in names}, rot=not frozen)
    TS = out['ts']
    r['analysis'] = dict(T=TEMP, dE_barrier=dE, dE_rxn=dEr, iso=out, kie=kie)
    save(r)
    log(f"barrier {dE:.2f}, reaction {dEr:.2f} kcal/mol; TS imag {TS['H']['imag']:.0f} cm-1 "
        f"(n_imag {TS['H']['n_imag']})")
    for kk, v in kie.items():
        log(f"KIE {kk}: semiclassical {v['semiclassical']:.2f}  Wigner {v['wigner']:.2f}  "
            f"Eckart {v['eckart']:.2f}   [experiment H/T = 6.91 +- 0.05 at 293 K]")
    return r


SCRATCH = ROOT / 'scratch'      # survives reboots, unlike /tmp; one structure's files at a time


def _dft_energy(atom, basis, xc, df, charge, tag, eps=4.0, conv_tol=1e-8, max_cycle=50):
    """B3LYP-D3(BJ)-style single point with PCM, resumable after an interruption:
    the density-fitting integrals go to SCRATCH under a name derived from the
    geometry and are reused if a complete file exists (it is renamed into place
    only once fully written), and the SCF is checkpointed every iteration and
    restarted from the last density. Both files are removed on success.
    Logs one line per SCF iteration."""
    import hashlib, os
    from pyscf import gto, dft as pdft
    SCRATCH.mkdir(exist_ok=True)
    key = hashlib.sha1(f'{atom}|{basis}|{xc}|{df}|{charge}|{eps}'.encode()).hexdigest()[:12]
    stem = SCRATCH / f"{tag.replace('/', '_')}_{key}"
    cderi, chk = Path(f'{stem}.cderi.h5'), Path(f'{stem}.chk')
    mol = gto.M(atom=atom, basis=basis, charge=charge, verbose=0)
    mf = pdft.RKS(mol)
    if df:
        mf = mf.density_fit()
        if cderi.exists():
            mf.with_df._cderi = str(cderi)
            log(f'  {tag}: reusing density-fitting integrals {cderi.name}')
        else:
            t0 = time.time()
            part = Path(f'{cderi}.part')
            part.unlink(missing_ok=True)
            mf.with_df._cderi_to_save = str(part)
            mf.with_df.build()
            os.replace(part, cderi)
            mf.with_df._cderi = str(cderi)
            log(f'  {tag}: density-fitting integrals built ({time.time() - t0:.0f}s, '
                f'{cderi.stat().st_size / 1e9:.1f} GB)')
    mf.xc = xc; mf.disp = 'd3bj'
    mf = mf.PCM(); mf.with_solvent.eps = eps
    mf.conv_tol = conv_tol; mf.max_cycle = max_cycle
    dm0 = None
    if chk.exists():
        try:
            dm0 = mf.from_chk(str(chk))
            log(f'  {tag}: restarting the SCF from {chk.name}')
        except Exception as ex:
            log(f'  {tag}: checkpoint unreadable ({ex}); starting fresh')
    mf.chkfile = str(chk)
    t_it = [time.time()]

    def progress(env):
        now = time.time()
        log(f"  {tag}: SCF cycle {env['cycle'] + 1} E={env['e_tot']:.8f} "
            f"|g|={env.get('norm_gorb', float('nan')):.1e} ({now - t_it[0]:.0f}s)")
        t_it[0] = now
    mf.callback = progress
    mf.kernel(dm0)
    if mf.converged:
        cderi.unlink(missing_ok=True); chk.unlink(missing_ok=True)
    return mf, mol


def stage_dft(r, basis='def2-svp', xc='b3lyp', df=None):
    """df: density fitting (RI-JK, def2-universal-jkfit); by default on for
    triple-zeta and larger, where the 4-index integrals for 96 atoms are too
    slow and too large for the NUC's memory"""
    if df is None:
        df = any(t in basis.lower() for t in ('tzv', 'qzv'))
    sym = r['prep']['symbols']
    r.setdefault('dft', {})
    for name in ('reactant', 'ts', 'product'):
        tag = f'{name}/{basis}'
        if tag in r['dft']:
            continue
        t0 = time.time()
        X = np.array(r[name]['coords'])
        atom = '\n'.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in zip(sym, X))
        mf, mol = _dft_energy(atom, basis, xc, df, r['prep']['charge'], tag)
        r['dft'][tag] = dict(E=float(mf.e_tot), converged=bool(mf.converged),
                             nao=int(mol.nao), density_fit=bool(df), seconds=time.time() - t0)
        save(r); log(f'DFT {tag}: E={mf.e_tot:.6f} conv={mf.converged} ({time.time() - t0:.0f}s)')
    b = r['dft']
    if all(f'{n}/{basis}' in b for n in ('reactant', 'ts', 'product')):
        log(f"DFT barrier {(b[f'ts/{basis}']['E'] - b[f'reactant/{basis}']['E']) * HARTREE_KCAL:.2f} "
            f"kcal/mol, reaction "
            f"{(b[f'product/{basis}']['E'] - b[f'reactant/{basis}']['E']) * HARTREE_KCAL:.2f}")
    return r


CAP = {'C': 1.09, 'N': 1.01, 'O': 0.97, 'S': 1.34}     # X-H cap length, A


def active_region(r, radius=2.6, extra=('nucleotide:C4', 'nucleotide:O4')):
    """One QM region for reactant, TS and product, chosen at the TS: heavy
    atoms within `radius` of the hydride plus `extra` (C4=O4 takes the enolate
    charge in the product), the hydride, and every H bonded to those heavy
    atoms. Bonds from the region to outside heavy atoms are cut and capped
    with H. Returns (atoms, cuts) as index lists."""
    sym = r['prep']['symbols']; k = r['prep']['key']; lab = r['prep']['labels']
    X = np.array(r['ts']['coords'])
    near = np.linalg.norm(X - X[k['Hd']], axis=1) < radius
    heavy = {i for i in range(len(sym)) if sym[i] != 'H' and near[i]}
    heavy |= {lab.index(n) for n in extra} | {k['Cd'], k['C11']}
    B = [(a, b) for a, b in _bonds(sym, X) if k['Hd'] not in (a, b)]
    atoms = set(heavy) | {k['Hd']}
    atoms |= {b if a in heavy else a for a, b in B
              if (a in heavy and sym[b] == 'H') or (b in heavy and sym[a] == 'H')}
    cuts = [(a, b) if a in atoms else (b, a) for a, b in B if (a in atoms) != (b in atoms)]
    return sorted(atoms), cuts


def _fragment(sym, X, atoms, cuts):
    xyz = [(sym[i], X[i]) for i in atoms]
    for a, b in cuts:                       # a inside, b outside
        u = X[b] - X[a]
        xyz.append(('H', X[a] + CAP[sym[a]] * u / np.linalg.norm(u)))
    return xyz


def stage_active(r, ne=4, no=4, basis='cc-pvdz', radius=2.6):
    """Circuits on a small active space around the transferring hydride, with
    the same capped QM region (see active_region) at reactant, TS and product
    so that their energies can be compared."""
    from chemoqbit.chem import build_problem, run_vqe, run_adapt
    Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16}
    sym = r['prep']['symbols']; lab = r['prep']['labels']
    atoms, cuts = active_region(r, radius)
    r.setdefault('active', {})
    for name in ('reactant', 'ts', 'product'):
        tag = f'{name}/{ne}e{no}o'
        if tag in r['active']:
            continue
        xyz = _fragment(sym, np.array(r[name]['coords']), atoms, cuts)
        nel = sum(Z[s] for s, _ in xyz)
        if nel % 2:
            raise RuntimeError(f'active region has {nel} electrons; the capping left an odd count')
        atom = '\n'.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in xyz)
        t0 = time.time()
        p = build_problem(atom, basis, no, ne, with_ccsd_t=True, charge=0)
        res = {}
        for m, runner in (('uccsd', run_vqe), ('adapt', run_adapt)):
            o = runner(p)
            res[m] = dict(e_vqe=o.energy, e_vqe_fixed=o.fixed_energy, err_mha=o.err_mha,
                          n_params=o.n_params, cnots=o.gate_counts.get('CNOT', 0))
        r['active'][tag] = dict(atoms=atoms, region=[lab[i] for i in atoms], n_caps=len(cuts),
                                n_atoms=len(xyz), charge=0, e_hf=p.e_hf, e_ccsd_t=p.e_ccsd_t,
                                e_casci=p.e_casci, n_qubits=p.n_qubits, **res,
                                seconds=time.time() - t0)
        save(r)
        log(f"active {tag}: {len(xyz)} atoms, {p.n_qubits} qubits, CASCI={p.e_casci:.6f}, "
            f"UCCSD err={res['uccsd']['err_mha']:.2e}, ADAPT err={res['adapt']['err_mha']:.2e} mHa "
            f"({time.time() - t0:.0f}s)")
    a = r['active']
    if all(f'{n}/{ne}e{no}o' in a for n in ('reactant', 'ts', 'product')):
        for m in ('e_ccsd_t', 'e_casci'):
            e = {n: a[f'{n}/{ne}e{no}o'][m] for n in ('reactant', 'ts', 'product')}
            log(f"active {m}: barrier {(e['ts'] - e['reactant']) * HARTREE_KCAL:.2f}, "
                f"reaction {(e['product'] - e['reactant']) * HARTREE_KCAL:.2f} kcal/mol")
    return r


def _environment(r, name, atoms, cuts):
    """point charges for the embedded fragment: GFN2-xTB charges of every cluster
    atom outside the QM region, with the atoms replaced by link hydrogens (the
    outer partners of the cut bonds) set to zero so they do not sit 1 A from a
    link atom and over-polarise it"""
    from chemoqbit.enzyme.xtb import charges
    sym = r['prep']['symbols']
    X = np.array(r[name]['coords'])
    q = charges(sym, X, r['prep']['charge'])
    drop = set(atoms) | {b for _, b in cuts}
    env = [i for i in range(len(sym)) if i not in drop]
    return X[env], q[env], float(q[[b for _, b in cuts]].sum())


def stage_oniom(r, ne=4, no=4, basis='cc-pvdz', low_basis='def2-svp', xc='b3lyp'):
    """ONIOM with electrostatic embedding, high level on the active_region fragment:

        E = E_DFT(cluster) + E_high(fragment in charges) - E_DFT(fragment in charges)

    E_DFT(cluster) is the stage_dft single point at `low_basis`; the fragment
    DFT uses the same functional and basis, without PCM (the dielectric is
    carried by the cluster term). High levels: CCSD(T), and CASCI(4,4) with the
    UCCSD- and ADAPT-VQE circuits run on the embedded Hamiltonian."""
    from pyscf import gto, dft as pdft, qmmm
    from chemoqbit.chem import build_problem, run_vqe, run_adapt
    Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16}
    sym = r['prep']['symbols']
    atoms, cuts = active_region(r)
    r.setdefault('oniom', {})
    for name in ('reactant', 'ts', 'product'):
        if name in r['oniom']:
            continue
        t0 = time.time()
        xyz = _fragment(sym, np.array(r[name]['coords']), atoms, cuts)
        if sum(Z[s] for s, _ in xyz) % 2:
            raise RuntimeError('embedded fragment has an odd electron count')
        atom = '\n'.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in xyz)
        Xe, qe, q_link = _environment(r, name, atoms, cuts)
        # low level on the model system
        mol = gto.M(atom=atom, basis=low_basis, charge=0, unit='Angstrom', verbose=0)
        mfd = pdft.RKS(mol); mfd.xc = xc; mfd.disp = 'd3bj'
        mfd = qmmm.mm_charge(mfd, Xe, qe, unit='Angstrom')
        mfd.conv_tol = 1e-9
        e_low = float(mfd.kernel())
        # high level
        p = build_problem(atom, basis, no, ne, with_ccsd_t=True, charge=0, point_charges=(Xe, qe))
        res = {}
        for m, runner in (('uccsd', run_vqe), ('adapt', run_adapt)):
            o = runner(p)
            res[m] = dict(e_vqe=o.energy, e_vqe_fixed=o.fixed_energy, err_mha=o.err_mha,
                          n_params=o.n_params, cnots=o.gate_counts.get('CNOT', 0))
        r['oniom'][name] = dict(e_dft_model=e_low, e_dft_model_conv=bool(mfd.converged), e_hf=p.e_hf,
                                e_ccsd_t=p.e_ccsd_t, e_casci=p.e_casci, n_qubits=p.n_qubits,
                                n_charges=len(qe), env_charge=float(qe.sum()), zeroed_link_charge=q_link,
                                **res, seconds=time.time() - t0)
        save(r)
        log(f"oniom {name}: {len(qe)} point charges (sum {qe.sum():+.2f}), DFT model {e_low:.6f}, "
            f"CCSD(T) {p.e_ccsd_t:.6f}, UCCSD err {res['uccsd']['err_mha']:.2e}, "
            f"ADAPT err {res['adapt']['err_mha']:.2e} mHa ({time.time() - t0:.0f}s)")
    o = r['oniom']
    names = ('reactant', 'ts', 'product')
    if all(n in o for n in names) and all(f'{n}/{low_basis}' in r.get('dft', {}) for n in names):
        cl = {n: r['dft'][f'{n}/{low_basis}']['E'] for n in names}
        for hi in ('e_ccsd_t', 'e_casci'):
            E = {n: cl[n] + o[n][hi] - o[n]['e_dft_model'] for n in names}
            frag = {n: o[n][hi] for n in names}
            log(f"ONIOM({hi[2:]}:{xc}/{low_basis}) barrier {(E['ts'] - E['reactant']) * HARTREE_KCAL:.2f}, "
                f"reaction {(E['product'] - E['reactant']) * HARTREE_KCAL:+.2f} kcal/mol;  embedded fragment "
                f"alone: barrier {(frag['ts'] - frag['reactant']) * HARTREE_KCAL:.2f}, "
                f"reaction {(frag['product'] - frag['reactant']) * HARTREE_KCAL:+.2f}")
    return r


def stage_focal(r, small='cc-pvdz', large='cc-pvtz'):
    """Basis-set correction for the ONIOM high level (focal-point style): DF-MP2
    on the same embedded fragment in `small` and `large` bases, giving

        CCSD(T)/small + [MP2/large - MP2/small]                      ('tz')
        CCSD(T)/small + [HF/large - HF/small] + extrapolated corr     ('cbs')

    the second with the two-point X^-3 extrapolation of the correlation energy
    (Helgaker et al. 1997; X = 2, 3 for DZ, TZ). Needs stage_oniom first."""
    from pyscf import gto, scf, qmmm
    from pyscf.mp import dfmp2
    sym = r['prep']['symbols']
    atoms, cuts = active_region(r)
    for name in ('reactant', 'ts', 'product'):
        o = r['oniom'][name]
        if 'mp2' in o and all(b in o['mp2'] for b in (small, large)):
            continue
        o.setdefault('mp2', {})
        xyz = _fragment(sym, np.array(r[name]['coords']), atoms, cuts)
        atom = '\n'.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in xyz)
        Xe, qe, _ = _environment(r, name, atoms, cuts)
        for b in (small, large):
            if b in o['mp2']:
                continue
            t0 = time.time()
            mol = gto.M(atom=atom, basis=b, charge=0, unit='Angstrom', verbose=0)
            mf = qmmm.mm_charge(scf.RHF(mol).density_fit(), Xe, qe, unit='Angstrom')
            mf.conv_tol = 1e-10
            mf.kernel()
            pt = dfmp2.DFMP2(mf); pt.kernel()
            o['mp2'][b] = dict(e_hf=float(mf.e_tot), e_corr=float(pt.e_corr), nao=int(mol.nao),
                               converged=bool(mf.converged), seconds=time.time() - t0)
            save(r)
            log(f"focal {name}/{b}: HF {mf.e_tot:.6f}, MP2 corr {pt.e_corr:.6f} ({time.time() - t0:.0f}s)")
    names = ('reactant', 'ts', 'product')
    hi = {}
    for n in names:
        o = r['oniom'][n]; s_, l_ = o['mp2'][small], o['mp2'][large]
        hi[n] = dict(tz=o['e_ccsd_t'] + (l_['e_hf'] + l_['e_corr']) - (s_['e_hf'] + s_['e_corr']),
                     cbs=o['e_ccsd_t'] + (l_['e_hf'] - s_['e_hf'])
                     + (27 * l_['e_corr'] - 8 * s_['e_corr']) / 19 - s_['e_corr'])
        o['e_ccsd_t_focal'] = hi[n]
    save(r)
    if all(f'{n}/def2-svp' in r.get('dft', {}) for n in names):
        cl = {n: r['dft'][f'{n}/def2-svp']['E'] for n in names}
        for kind in ('tz', 'cbs'):
            E = {n: cl[n] + hi[n][kind] - r['oniom'][n]['e_dft_model'] for n in names}
            F = {n: hi[n][kind] for n in names}
            log(f"ONIOM(CCSD(T)[{kind}]:B3LYP/def2-svp) barrier {(E['ts'] - E['reactant']) * HARTREE_KCAL:.2f}, "
                f"reaction {(E['product'] - E['reactant']) * HARTREE_KCAL:+.2f} kcal/mol; fragment barrier "
                f"{(F['ts'] - F['reactant']) * HARTREE_KCAL:.2f}")
    return r

STAGES = dict(prep=stage_prep, scan=stage_scan, ts=stage_ts, hess=stage_hess,
              kie=stage_kie, dft=stage_dft, active=stage_active, oniom=stage_oniom, focal=stage_focal)
ORDER = ['prep', 'scan', 'ts', 'hess', 'kie', 'dft', 'active', 'oniom', 'focal']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', default='0,1,2,3,4',
                    help='comma list of stage numbers or names (0=prep ... 6=active)')
    ap.add_argument('--basis', default='def2-svp')
    ap.add_argument('--variant', default='default', choices=sorted(VARIANTS),
                    help='model variant; his147p = His147 protonated, o4h = O4 protonated; cluster charge +2')
    args = ap.parse_args()
    VARIANT = VARIANTS[args.variant]
    OUT = OUT.with_name(f"ts_cluster_run{VARIANT['suffix']}.json")
    sel = [ORDER[int(s)] if s.strip().isdigit() else s.strip() for s in args.stages.split(',')]
    r = load()
    for s in sel:
        log(f'=== stage {s} ===')
        r = stage_dft(r, basis=args.basis) if s == 'dft' else STAGES[s](r)
        save(r)
    log('done')
