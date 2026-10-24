"""C5 proton abstraction in thymidylate synthase: dUMP against FdUMP.

Why 5-fluorouracil traps the enzyme, as a calculation rather than an assumption.
In the covalent ternary complex the C5 proton is removed and the Cys146 thiolate
eliminated in a concerted step (Kholodar et al., Biochemistry 60, 1243 (2021)).
FdUMP carries fluorine at C5 and so has no proton to remove; the standard account
is that the complex is therefore trapped, which this script tests by computing
both routes for both substrates.

No candidate base is within reach of the proton in any crystal structure. In
1TLS the nearest acceptors to C5 are Tyr94 OH (4.14 A) and water 387 (4.55 A),
and in the 2021 intermediate-analog structures 7JX1/7JXF (superposed onto 1TLS
here to 0.3 A RMSD over 26 CA) the nearest water overlays water 387, at 4.07 A.
The base must therefore approach before the proton can move, which is the
objection Ghosh et al. (Phys. Chem. Chem. Phys. 17, 30867 (2015)) raise against
any single residue acting as the general base. The coordinates are accordingly
the approach r = d(C5...A) and the transfer xi = d(C5-H) - d(H...A).

Stages, each resumable from results/ts_abstraction.json:

  prep      build the 103-atom cluster from PDB 1TLS (unedited covalent complex,
            plus Glu58) for c5 = H and F, relax with GFN2-xTB
  acceptor  dUMP: how cheaply each candidate base can approach C5
  cleave    both: C6-S cleavage alone, with C5 keeping its substituent -- the
            only route open to FdUMP
  band      dUMP: 2D band over the approach and the proton transfer

    python scripts/ts_abstraction.py [stage ...]      # default: all, in order
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from chemoqbit.enzyme import abstraction as A
from chemoqbit.enzyme.frozen import optimise_frozen
from chemoqbit import sanity

OUT = ROOT / 'results' / 'ts_abstraction.json'
K = 627.509474
APPROACH = np.round(np.arange(2.5, 4.81, 0.2), 2)   # d(C5...acceptor)
XIR = np.round(np.arange(-2.6, 0.61, 0.2), 2)       # d(C5-H) - d(H...acceptor)
RCS = np.round(np.arange(1.95, 3.56, 0.2), 2)       # d(C6-S)
KRES = 20.0
# the band keeps the proton bonded to C5 or to the acceptor: for a collinear
# C5-H...A, D_LO <= d(C5-H) <= D_HI. Outside it a restrained xi merely tears the
# proton off into the cavity, which is what an absolute xi range did.
D_LO, D_HI = 0.95, 1.70


def band(r):
    """the xi values reachable at approach r without tearing the proton off"""
    return [float(x) for x in XIR if 2 * D_LO - r <= x <= 2 * D_HI - r]


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(d):
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(OUT)


def model(c5):
    """(symbols, X, labels, frozen indices, index of the C5 hydrogen or None)"""
    sym, X0, lab, fr = A.build(c5=c5)
    molH = A.add_hydrogens(sym, X0, lab)
    conf = molH.GetConformer()
    P = np.array([list(conf.GetAtomPosition(i)) for i in range(molH.GetNumAtoms())])
    S = [a.GetSymbol() for a in molH.GetAtoms()]
    return S, P, lab, [i for i, f in enumerate(fr) if f], A.c5_hydrogen(molH, lab)


def geom(X, lab, h):
    i = lambda k: lab.index(k)
    d = lambda a, b: float(np.linalg.norm(X[a] - X[b]))
    g = {'C6-SG': d(i('nucleotide:C6'), i('cys146:SG')),
         'C5-C11': d(i('nucleotide:C5'), i('methylene:C11')),
         'C11-N5': d(i('methylene:C11'), i('folate:N5')),
         'C5-C6': d(i('nucleotide:C5'), i('nucleotide:C6'))}
    if h is not None:
        g['C5-H'] = d(i('nucleotide:C5'), h)
        for name, key in A.ACCEPTORS.items():
            g[f'H-{name}'] = d(h, i(key))
            g[f'C5-{name}'] = d(i('nucleotide:C5'), i(key))
    return g


def _relax(p, X, cons):
    return optimise_frozen(p['symbols'], X, charge=0, frozen=p['frozen'], constraints=cons, k=KRES)


def _relax_safe(p, X, cons):
    """(X, E, ok): an xTB SCF failure marks the point instead of ending the run."""
    try:
        Xn, E = _relax(p, X, cons)
        return Xn, float(E), True
    except Exception as e:
        print(f'    point failed: {type(e).__name__}: {e}', flush=True)
        return np.array(X), None, False


def stage_prep(res):
    res.setdefault('prep', {})
    for c5 in ('H', 'F'):
        if c5 in res['prep']:
            continue
        S, P, lab, frozen, h = model(c5)
        t0 = time.time()
        X, E = optimise_frozen(S, P, charge=0, frozen=frozen)
        g = geom(X, lab, h)
        res['prep'][c5] = dict(symbols=S, labels=lab, frozen=frozen, hydrogen=h, X=X.tolist(),
                               E=float(E), geometry=g, seconds=time.time() - t0)
        # the covalent ternary complex must survive relaxation, or there is no reactant
        sanity.record(res, 'bond_CS', g['C6-SG'], label=f'prep {c5} C6-S', fatal=True)
        sanity.record(res, 'bond_CC', g['C5-C11'], label=f'prep {c5} C5-C11', fatal=True)
        sanity.record(res, 'bond_CC', g['C11-N5'], label=f'prep {c5} C11-N5')
        save(res)
        print(f"prep {c5}: E {E:.6f}  "
              + '  '.join(f'{k} {v:.2f}' for k, v in res['prep'][c5]['geometry'].items()), flush=True)


def stage_acceptor(res):
    """How cheaply can each candidate base approach C5? The proton keeps its
    natural position; only d(C5...A) is restrained."""
    p = res['prep']['H']; lab, h = p['labels'], p['hydrogen']
    i5 = lab.index('nucleotide:C5')
    res.setdefault('acceptor', {})
    for name, key in A.ACCEPTORS.items():
        if name in res['acceptor']:
            continue
        ja = lab.index(key)
        X = np.array(p['X']); r0 = float(np.linalg.norm(X[i5] - X[ja]))
        rows = []
        for r in [v for v in APPROACH if v <= r0 + 1e-9][::-1]:
            X, E, ok = _relax_safe(p, X, [(i5, ja, float(r))])
            rows.append(dict(r=float(r), E=E, rel=None if E is None else float((E - p['E']) * K),
                             converged=ok, X=X.tolist() if ok else None, **geom(X, lab, h)))
            if ok:
                sanity.record(res, 'bond_CS', rows[-1]['C6-SG'], label=f'approach {name} r {r}')
            print(f"  approach {name} r {r:.2f}  "
                  + ('failed' if E is None else f"{(E - p['E']) * K:+8.2f} kcal/mol")
                  + f"  C5-H {rows[-1]['C5-H']:.2f} C6-SG {rows[-1]['C6-SG']:.2f}", flush=True)
        # backward pass: re-relax each point from its inner neighbour and keep the
        # lower energy. Without it these scans fell by 3-5 kcal/mol at single
        # points, which is as large as the difference between the two waters.
        for i in range(len(rows) - 2, -1, -1):
            nxt, cur = rows[i + 1], rows[i]
            if not nxt.get('X') or cur['rel'] is None:
                continue
            Xb, Eb, okb = _relax_safe(p, np.array(nxt['X']), [(i5, ja, float(cur['r']))])
            if okb and Eb < cur['E'] - 1e-7:
                print(f"  backward {name} r {cur['r']:.2f}  {cur['rel']:+.2f} -> "
                      f"{(Eb - p['E']) * K:+.2f} kcal/mol", flush=True)
                rows[i] = dict(r=cur['r'], E=Eb, rel=float((Eb - p['E']) * K), converged=True,
                               X=Xb.tolist(), backward=True, **geom(Xb, lab, h))
        good = [x for x in rows if x['rel'] is not None]
        drop = sanity.largest_drop([x['rel'] for x in sorted(good, key=lambda x: -x['r'])])
        sanity.record(res, 'scan_drop', drop, label=f'approach {name} after the backward pass')
        res['acceptor'][name] = dict(
            atom=key, r_reactant=r0, scan=rows,
            cost_close=(min(good, key=lambda x: x['r'])['rel'] if good else None),
            r_close=(min(good, key=lambda x: x['r'])['r'] if good else None),
            cost_min=min((x['rel'] for x in good), default=None))
        save(res)
    # rank on the stored scans, not on a field that an older run may have left
    # empty: the approach grid holds no 2.6 A point, so an exact lookup found none
    ranked = {}
    for k, v in res['acceptor'].items():
        good = [x for x in v['scan'] if x.get('rel') is not None]
        if good:
            c = min(good, key=lambda x: x['r'])
            ranked[k] = (c['rel'], c['r'])
    assert ranked, 'no acceptor produced a usable approach scan'
    best = min(ranked, key=lambda k: ranked[k][0])
    res['best_acceptor'] = best
    save(res)
    print('closest approach reached and its cost: '
          + ', '.join(f'{k} {r:.2f} A {c:+.2f}' for k, (c, r) in ranked.items())
          + f'  -> best {best}', flush=True)


def stage_cleave(res):
    """C6-S cleavage with the C5 substituent kept: the only route open to FdUMP."""
    res.setdefault('cleave', {})
    for c5 in ('H', 'F'):
        if c5 in res['cleave']:
            continue
        p = res['prep'][c5]; lab, h = p['labels'], p['hydrogen']
        i6, isg = lab.index('nucleotide:C6'), lab.index('cys146:SG')
        X = np.array(p['X']); rows = []
        for rcs in RCS:
            X, E, ok = _relax_safe(p, X, [(i6, isg, float(rcs))])
            rows.append(dict(rcs=float(rcs), E=E, rel=None if E is None else float((E - p['E']) * K),
                             converged=ok, **geom(X, lab, h)))
            print(f"  cleave {c5} rCS {rcs:.2f}  "
                  + ('failed' if E is None else f"{(E - p['E']) * K:+8.2f} kcal/mol"), flush=True)
        good = [x for x in rows if x['rel'] is not None]
        res['cleave'][c5] = dict(scan=rows, barrier=max((x['rel'] for x in good), default=None),
                                 final=good[-1]['rel'] if good else None)
        save(res)


def relay_hops(p, X):
    """The two hops of the proton relay: C5's proton to water 327, and one of that
    water's protons to the Glu58 carboxylate, which sits 2.8 A away. Driving only
    the first hop builds a hydronium the 103-atom cluster cannot stabilise, which
    cost over 40 kcal/mol before the proton was even halfway; the relay is charge
    balanced, since Glu58 goes from -1 to neutral as the thiolate forms."""
    lab, h = p['labels'], p['hydrogen']
    i = lambda k: lab.index(k)
    c5, ow, oe = i('nucleotide:C5'), i('wat327:O'), i('glu58:OE2')
    hw = [j for j, s_ in enumerate(p['symbols'])
          if s_ == 'H' and np.linalg.norm(X[j] - X[ow]) < 1.2]
    assert len(hw) == 2, f'water 327 has {len(hw)} hydrogens'
    h2 = min(hw, key=lambda j: np.linalg.norm(X[j] - X[oe]))     # the one facing Glu58
    return [(c5, h, ow), (ow, h2, oe)], h2


def _backward(res, p, s, hops, lab, h, h2, label, extra=None):
    """Re-relax each point from its neighbour on the far side and keep the lower
    energy, so a cheaper channel found late in the scan propagates back.

    Without it the relay profile fell 11 kcal/mol at one point: the scan had
    switched from moving the C5 proton to protonating Glu58 first, and the two
    channels were stitched together into a path that is not a path.
    """
    if s.get('backward_converged'):
        print(f'  {label}: backward pass already converged, skipping', flush=True)
        return
    i5 = lab.index('nucleotide:C5')
    pts = sorted([v for v in s['points'].values() if v.get('rel') is not None],
                 key=lambda v: v['xi'])
    d = lambda X, a, b: float(np.linalg.norm(X[a] - X[b]))
    improved, best_gain = 0, 0.0
    for i in range(len(pts) - 2, -1, -1):
        nxt, cur = pts[i + 1], pts[i]
        if not nxt.get('X'):
            continue
        cons = [('relay', hops, float(cur['xi']))]
        if extra is not None and cur.get('rcs_target') is not None:
            cons.append((extra[0], extra[1], float(cur['rcs_target'])))
        X, E, ok = _relax_safe(p, np.array(nxt['X']), cons)
        if ok and E < cur['E'] - 1e-7:
            g = geom(X, lab, h)
            rec = dict(cur); rec.update(E=E, rel=float((E - p['E']) * K), X=X.tolist(),
                                        backward=True, C5_H=d(X, i5, h),
                                        H_O327=d(X, h, lab.index('wat327:O')),
                                        O327_H2=d(X, lab.index('wat327:O'), h2),
                                        H2_Glu58=d(X, h2, lab.index('glu58:OE2')), **g)
            s['points'][str(cur['xi'])] = rec
            pts[i] = rec
            improved += 1
            best_gain = max(best_gain, cur['rel'] - rec['rel'])
            print(f"  backward {label} xi {cur['xi']:+5.2f}  {cur['rel']:+.2f} -> "
                  f"{rec['rel']:+.2f} kcal/mol", flush=True)
            save(res)
    # One full pass costs an hour on 26 points, and it is repeated on every
    # restart. Once a pass changes nothing beyond numerical noise there is
    # nothing left for it to find, so record that and skip it next time.
    if best_gain < 0.05:
        s['backward_converged'] = True
    print(f'  {label}: backward pass improved {improved} points '
          f'(largest gain {best_gain:.2f} kcal/mol)'
          + ('; converged, it will be skipped from now on' if best_gain < 0.05 else ''),
          flush=True)
    save(res)


def stage_relay(res):
    """The concerted two-proton relay: the route the network makes available."""
    p = res['prep']['H']; lab, h = p['labels'], p['hydrogen']
    X0 = np.array(p['X'])
    hops, h2 = relay_hops(p, X0)
    xi0 = sum(float(np.linalg.norm(X0[d] - X0[hy]) - np.linalg.norm(X0[hy] - X0[a]))
              for d, hy, a in hops)
    s = res.setdefault('relay', dict(hops=[list(t) for t in hops], water_h=h2, xi_reactant=xi0,
                                     points={}))
    targets = [round(v, 2) for v in np.arange(round(xi0 * 2) / 2, 1.61, 0.2)]
    X = X0.copy()
    for xi in targets:
        k_ = str(xi)
        if k_ in s['points']:
            if s['points'][k_].get('X'):
                X = np.array(s['points'][k_]['X'])
            continue
        t0 = time.time()
        X, E, ok = _relax_safe(p, X, [('relay', hops, float(xi))])
        g = geom(X, lab, h)
        d = lambda a, b: float(np.linalg.norm(X[a] - X[b]))
        s['points'][k_] = dict(xi=xi, E=E, rel=None if E is None else float((E - p['E']) * K),
                               converged=ok, X=X.tolist() if ok else None,
                               C5_H=d(lab.index('nucleotide:C5'), h),
                               H_O327=d(h, lab.index('wat327:O')),
                               O327_H2=d(lab.index('wat327:O'), h2),
                               H2_Glu58=d(h2, lab.index('glu58:OE2')), **g)
        if ok:
            sanity.record(res, 'bond_CS', g['C6-SG'], label=f'relay xi {xi}')
            sanity.record(res, 'proton_bonded', min(v['C5_H'], v['H_O327']),
                          label=f'relay xi {xi}')
        save(res)
        v = s['points'][k_]
        print(f"  relay xi {xi:+5.2f}  " + ('failed' if E is None else f"{v['rel']:+8.2f} kcal/mol")
              + f"  C5-H {v['C5_H']:.2f} H..O327 {v['H_O327']:.2f} | O327-H {v['O327_H2']:.2f} "
                f"H..Glu58 {v['H2_Glu58']:.2f}  C6-S {g['C6-SG']:.2f}  ({time.time() - t0:.0f}s)",
              flush=True)
    _backward(res, p, s, hops, lab, h, h2, label='relay')
    good = [v for v in s['points'].values() if v['rel'] is not None]
    if good:
        s['barrier'] = max(v['rel'] for v in good)
        sanity.record(res, 'reaction_barrier', s['barrier'], label='relay barrier')
        seq = [v['rel'] for v in sorted(good, key=lambda v: v['xi'])]
        sanity.record(res, 'scan_drop', sanity.largest_drop(seq), label='relay profile')
        print(f"relay: highest point {s['barrier']:+.2f} kcal/mol "
              f"(FdUMP's cleavage-only route costs {res['cleave']['F']['barrier']:+.2f})", flush=True)
    save(res)


def relay_guide(X, hops):
    """d(xi_relay)/dX: the collective displacement a saddle search should follow.

    Each hop contributes the unit vectors of its donor-H and H-acceptor bonds, so
    the components are O(1) and well scaled -- unlike d(omega)/dX for a torsion,
    which blows up with the lever arms and made the search lock onto a stretch.
    This is the same form of guide that worked for the hydride transfer.
    """
    g = np.zeros_like(X)
    for d, h, a in hops:
        u1 = X[d] - X[h]; u1 = u1 / np.linalg.norm(u1)
        u2 = X[h] - X[a]; u2 = u2 / np.linalg.norm(u2)
        g[d] += u1; g[h] += -u1 - u2; g[a] += u2
    return g


def stage_stepwise(res):
    """Drive the C5 proton explicitly, from the state where Glu58 is already
    protonated.

    One restraint on the SUM of the two hops is degenerate: across all 26 relay
    points C5-H never left 1.09-1.15 A, because the sum could be satisfied by the
    water-to-Glu58 hop alone and then by distances drifting. The relay scan
    therefore measured the cost of protonating Glu58 (about +10 kcal/mol), not
    the abstraction.

    The water donates its proton to Glu58 early and cheaply, which leaves
    hydroxide beside C5 -- a far stronger base. So the question that remains is
    the cost of that hydroxide taking the C5 proton, and it is driven here by its
    own antisymmetric stretch, with C6-S ramped alongside since the elimination
    is concerted.
    """
    p = res['prep']['H']; lab, h = p['labels'], p['hydrogen']
    hops, h2 = relay_hops(p, np.array(p['X']))
    i5, iow = lab.index('nucleotide:C5'), lab.index('wat327:O')
    i6, isg = lab.index('nucleotide:C6'), lab.index('cys146:SG')
    # start from the relay point where Glu58 is protonated and the energy lowest
    cand = [v for v in res['relay']['points'].values()
            if v.get('rel') is not None and v.get('X') and v['H2_Glu58'] < 1.2]
    assert cand, 'no relay point has Glu58 protonated'
    start = min(cand, key=lambda v: v['rel'])
    X = np.array(start['X'])
    e_ref = start['rel']
    xi1_0 = float(np.linalg.norm(X[i5] - X[h]) - np.linalg.norm(X[h] - X[iow]))
    r0 = float(np.linalg.norm(X[i6] - X[isg]))
    XI_END, RCS_END = 1.0, 2.70
    s = res.setdefault('stepwise', dict(start_xi=start['xi'], start_rel=e_ref,
                                        xi1_start=xi1_0, rcs_start=r0, points={}))
    targets = [round(v, 2) for v in np.arange(round(xi1_0 * 5) / 5, XI_END + 0.01, 0.2)]
    print(f'  starting from the relay point xi {start["xi"]:+.2f} ({e_ref:+.2f} kcal/mol), '
          f'Glu58 protonated; driving d(C5-H) - d(H-O327) from {xi1_0:+.2f}', flush=True)
    for xi in targets:
        k_ = str(xi)
        if k_ in s['points']:
            if s['points'][k_].get('X'):
                X = np.array(s['points'][k_]['X'])
            continue
        frac = min(max((xi - xi1_0) / (XI_END - xi1_0), 0.0), 1.0)
        rcs = r0 + frac * (RCS_END - r0)
        t0 = time.time()
        X, E, ok = _relax_safe(p, X, [(i5, h, h, iow, float(xi)), (i6, isg, float(rcs))])
        g = geom(X, lab, h)
        d = lambda a, b: float(np.linalg.norm(X[a] - X[b]))
        s['points'][k_] = dict(xi1=xi, rcs_target=rcs, E=E,
                               rel=None if E is None else float((E - p['E']) * K),
                               converged=ok, X=X.tolist() if ok else None,
                               C5_H=d(i5, h), H_O327=d(h, iow),
                               H2_Glu58=d(h2, lab.index('glu58:OE2')), **g)
        if ok:
            # the proton must be bonded to C5 or to the acceptor; a completed
            # transfer leaves C5-H long and that is the reaction, not a fault
            sanity.record(res, 'proton_bonded',
                          min(s['points'][k_]['C5_H'], s['points'][k_]['H_O327']),
                          label=f'stepwise xi1 {xi}')
        save(res)
        v = s['points'][k_]
        print(f"  stepwise xi1 {xi:+5.2f} rCS->{rcs:.2f}  "
              + ('failed' if E is None else f"{v['rel']:+8.2f} kcal/mol")
              + f"  C5-H {v['C5_H']:.2f} H..O {v['H_O327']:.2f} C6-S {g['C6-SG']:.2f}"
                f"  ({time.time() - t0:.0f}s)", flush=True)
    good = [v for v in s['points'].values() if v['rel'] is not None]
    if good:
        s['barrier'] = max(v['rel'] for v in good)
        sanity.record(res, 'reaction_barrier', s['barrier'], label='stepwise barrier')
        # This profile is meant to rise to a barrier and fall to the product, so the
        # descent is the reaction itself. largest_drop reported that 13 kcal/mol
        # descent as a fault; the right test is a reversal INSIDE a monotonic run.
        seq = [v['rel'] for v in sorted(good, key=lambda v: v['xi1'])]
        sanity.record(res, 'scan_drop', sanity.segment_drop(seq), label='stepwise profile')
        sanity.record(res, 'profile_segments', sanity.monotonic_segments(seq),
                      label='stepwise profile')
        moved = [v for v in good if v['C5_H'] > 1.3]
        print(f"stepwise: highest {s['barrier']:+.2f} kcal/mol, C5-H reached "
              f"{max(v['C5_H'] for v in good):.2f} A at {len(moved)} points "
              f"(FdUMP's only route costs {res['cleave']['F']['barrier']:+.2f})", flush=True)
    save(res)


def stage_concerted(res):
    """The elimination as the literature describes it: the C5 proton and the
    Cys146 thiolate leaving together.

    Driving the protons alone is not enough. Along the relay scan C6-S barely
    moved (1.86 -> 1.90 A) while C5-H stretched to 1.23, so the coordinate was
    building a carbanion at C5 with the thiolate still bonded, and the energy
    reached +24.8 kcal/mol with the proton only a quarter transferred. A relaxed
    scan at fixed proton position finds the local minimum, which keeps the C-S
    bond intact; the bond has to be driven too.

    Here r(C6-S) follows a ramp tied to the relay coordinate, so the two leave in
    step. If the concerted picture is right this barrier falls well below the
    25.10 kcal/mol that C6-S cleavage alone costs FdUMP.
    """
    p = res['prep']['H']; lab, h = p['labels'], p['hydrogen']
    X0 = np.array(p['X'])
    hops, h2 = relay_hops(p, X0)
    i6, isg = lab.index('nucleotide:C6'), lab.index('cys146:SG')
    i5 = lab.index('nucleotide:C5')
    xi0 = sum(float(np.linalg.norm(X0[d] - X0[hy]) - np.linalg.norm(X0[hy] - X0[a]))
              for d, hy, a in hops)
    r0 = float(np.linalg.norm(X0[i6] - X0[isg]))
    XI_END, RCS_END = 1.5, 2.70
    s = res.setdefault('concerted', dict(hops=[list(t) for t in hops], water_h=h2,
                                         xi_reactant=xi0, rcs_reactant=r0,
                                         rcs_end=RCS_END, points={}))
    targets = [round(v, 2) for v in np.arange(round(xi0 * 2) / 2, XI_END + 0.01, 0.2)]
    X = X0.copy()
    for xi in targets:
        k_ = str(xi)
        if k_ in s['points']:
            if s['points'][k_].get('X'):
                X = np.array(s['points'][k_]['X'])
            continue
        frac = min(max((xi - xi0) / (XI_END - xi0), 0.0), 1.0)
        rcs = r0 + frac * (RCS_END - r0)
        t0 = time.time()
        X, E, ok = _relax_safe(p, X, [('relay', hops, float(xi)), (i6, isg, float(rcs))])
        g = geom(X, lab, h)
        d = lambda a, b: float(np.linalg.norm(X[a] - X[b]))
        s['points'][k_] = dict(xi=xi, rcs_target=rcs, E=E,
                               rel=None if E is None else float((E - p['E']) * K),
                               converged=ok, X=X.tolist() if ok else None,
                               C5_H=d(i5, h), H_O327=d(h, lab.index('wat327:O')),
                               O327_H2=d(lab.index('wat327:O'), h2),
                               H2_Glu58=d(h2, lab.index('glu58:OE2')), **g)
        save(res)
        v = s['points'][k_]
        if ok:
            sanity.record(res, 'bond_CS', g['C6-SG'], label=f'concerted xi {xi}')
            sanity.record(res, 'proton_bonded', min(v['C5_H'], v['H_O327']),
                          label=f'concerted xi {xi}')
        print(f"  concerted xi {xi:+5.2f} rCS->{rcs:.2f}  "
              + ('failed' if E is None else f"{v['rel']:+8.2f} kcal/mol")
              + f"  C5-H {v['C5_H']:.2f} H..O {v['H_O327']:.2f} C6-S {g['C6-SG']:.2f}"
                f"  ({time.time() - t0:.0f}s)", flush=True)
    _backward(res, p, s, hops, lab, h, h2, label='concerted', extra=(i6, isg))
    good = [v for v in s['points'].values() if v['rel'] is not None]
    if good:
        s['barrier'] = max(v['rel'] for v in good)
        sanity.record(res, 'reaction_barrier', s['barrier'], label='concerted barrier')
        seq = [v['rel'] for v in sorted(good, key=lambda v: v['xi'])]
        sanity.record(res, 'scan_drop', sanity.largest_drop(seq), label='concerted profile')
        print(f"concerted: highest point {s['barrier']:+.2f} kcal/mol "
              f"(FdUMP's cleavage-only route costs {res['cleave']['F']['barrier']:+.2f})", flush=True)
    save(res)


def stage_saddle(res):
    """Refine the relay maximum into a genuine transition state and prove it with
    frequencies: a restrained scan point is not a stationary point."""
    from chemoqbit.enzyme import tsopt as T
    from chemoqbit.enzyme.xtb import energy_grad
    p = res['prep']['H']; sym = p['symbols']; lab = p['labels']; h = p['hydrogen']
    frozen = p['frozen']
    hops = [tuple(t) for t in res['relay']['hops']]
    good = [v for v in res['relay']['points'].values() if v.get('rel') is not None and v.get('X')]
    assert good, 'the relay scan has no usable points'
    top = max(good, key=lambda v: v['rel'])
    X = np.array(top['X'])
    print(f"  refining the relay maximum: xi {top['xi']:+.2f}, {top['rel']:+.2f} kcal/mol", flush=True)
    fn = lambda Y: (lambda r: (float(r[0]), r[1]))(energy_grad(sym, Y, 0))
    gfn = lambda Y: energy_grad(sym, Y, 0)[1]
    free = [i for i in range(len(sym)) if i not in set(frozen)]
    t0 = time.time()
    Xn, E, info = T.prfo_follow(X, fn, relay_guide(X, hops), grad_fn=gfn, frozen=frozen,
                                trust=0.06, gtol=3e-4, maxiter=60, neg=0.02, log=lambda m: None)
    nu = T.frequencies(np.array(Xn), T.fd_hessian(np.array(Xn), gfn, free=free),
                       [T.AMU[s_] for s_ in sym], free=free)
    nim = int((nu < -50).sum())
    g = geom(np.array(Xn), lab, h)
    res['saddle'] = dict(E=float(E), rel=float((E - p['E']) * K), X=np.array(Xn).tolist(),
                         n_imaginary=nim, nu_imag=float(nu.min()) if nim else None,
                         followed_eig=info.get('followed_eig'), seconds=time.time() - t0, **g)
    sanity.record(res, 'imag_freq_ts', nim, label='relay saddle')
    if nim:
        sanity.record(res, 'imag_wavenumber', abs(nu.min()), label='relay saddle')
    sanity.record(res, 'reaction_barrier', res['saddle']['rel'], label='relay barrier')
    save(res)
    print(f"relay saddle: {res['saddle']['rel']:+.2f} kcal/mol, imaginary modes {nim}"
          + (f" ({abs(nu.min()):.0f}i cm-1)" if nim else '')
          + f", C6-S {g['C6-SG']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  FdUMP's only route costs {res['cleave']['F']['barrier']:+.2f} kcal/mol", flush=True)


def stage_band(res):
    """2D band over the approach and the proton transfer, for the best acceptor."""
    p = res['prep']['H']; lab, h = p['labels'], p['hydrogen']
    i5 = lab.index('nucleotide:C5')
    name = res['best_acceptor']
    ja = lab.index(A.ACCEPTORS[name])
    appr = {x['r']: x['X'] for x in res['acceptor'][name]['scan'] if x['X']}
    s = res.setdefault('band', dict(acceptor=name, xi=XIR.tolist(), points={}))
    for r in sorted(appr, reverse=True):
        X = np.array(appr[r])                      # each row starts from the approach scan
        for xi in band(r):
            kk = f'{r}_{xi}'
            if kk in s['points']:
                if s['points'][kk].get('X'):
                    X = np.array(s['points'][kk]['X'])
                continue
            t0 = time.time()
            X, E, ok = _relax_safe(p, X, [(i5, ja, float(r)), (i5, h, ja, h, float(xi))])
            s['points'][kk] = dict(E=E, rel=None if E is None else float((E - p['E']) * K),
                                   converged=ok, X=X.tolist() if ok else None, **geom(X, lab, h))
            save(res)
            print(f"  band r {r:.2f} xi {xi:+.1f}  "
                  + ('failed' if E is None else f"{(E - p['E']) * K:+8.2f} kcal/mol")
                  + f"  C5-H {s['points'][kk]['C5-H']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    good = [v['rel'] for v in s['points'].values() if v['rel'] is not None]
    s['min_barrier_on_band'] = min(good) if good else None
    save(res)


STAGES = dict(prep=stage_prep, acceptor=stage_acceptor, cleave=stage_cleave,
              relay=stage_relay, stepwise=stage_stepwise, concerted=stage_concerted,
              saddle=stage_saddle, band=stage_band)


def main(names):
    res = load()
    for n in names:
        print(f'=== {n}', flush=True)
        STAGES[n](res)
        res['_stages_done'] = sorted(set(res.get('_stages_done', [])) | {n})
        save(res)


if __name__ == '__main__':
    main(sys.argv[1:] or ['prep', 'acceptor', 'cleave', 'relay', 'stepwise', 'saddle'])
