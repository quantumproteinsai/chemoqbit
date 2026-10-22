"""Two-dimensional surface for the thymidylate synthase hydride transfer:
the hydride coordinate xi = d(C6-H) - d(H-C11) and the C6(dUMP)-S(Cys146)
distance r, to test whether C-S cleavage - concerted but very asynchronous
with the hydride transfer in QM/MM studies - removes the excess barrier and
endothermicity found along the one-dimensional path (the default model).

Stage A  GFN2-xTB constrained optimisations on the (xi, r) grid. Each xi
         column starts from the matching point of the backfilled 1D scan
         (r ~ 1.87 A) and stretches C6-S step by step, so neighbours stay
         continuous.
Stage B  ONIOM(wB97X-D3(BJ)/def2-SVP : GFN2-xTB) energies at every grid
         point, mechanical embedding, with a fragment enlarged to contain
         C6 of the nucleotide and SG of Cys146 (22 atoms with links).
Summary  surfaces relative to the reactant corner, and the minimax path from
         the reactant corner to the product corner: its highest point is the
         lowest barrier the grid allows, and the saddle estimate for stage C.

Resumable: results/ts_cs2d.json, written atomically after every point.

    python scripts/ts_cs2d.py
"""
import json, sys, time, heapq
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'results' / 'ts_cs2d.json'
XI = [round(-1.4 + 0.2 * i, 1) for i in range(15)]           # -1.4 ... +1.4
RCS = [round(1.85 + 0.2 * j, 2) for j in range(9)]            # 1.85 ... 3.45
EXTRA = ('nucleotide:C4', 'nucleotide:O4', 'nucleotide:C6', 'cys146:SG')
XC, BASIS = 'wb97x-d3bj', 'def2-svp'


def log(*a):
    print(*a, flush=True)


def save(res):
    tmp = OUT.with_suffix('.json.tmp'); tmp.write_text(json.dumps(res, indent=1)); tmp.replace(OUT)


def key(xi, r):
    return f'{xi:+.1f}_{r:.2f}'


def main():
    import ts_cluster_nuc as S
    from chemoqbit.enzyme.frozen import optimise_frozen
    from chemoqbit.enzyme.xtb import energy_grad
    from chemoqbit.enzyme.oniom import fragment
    from pyscf import gto, dft
    K = S.HARTREE_KCAL
    r0 = json.loads((ROOT / 'results' / 'ts_cluster_run.json').read_text())
    p = r0['prep']; sym, k, frozen, q = p['symbols'], p['key'], p['frozen'], p['charge']
    rest = [tuple(c) for c in p.get('restraints', [])]
    atoms, cuts = S.active_region(r0, extra=EXTRA)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res.setdefault('config', dict(xi=XI, rcs=RCS, fragment_atoms=atoms, cuts=[list(c) for c in cuts],
                                  xc=XC, basis=BASIS, charge=q))
    pts = res.setdefault('points', {})
    scan = {round(s['xi'], 1): s for s in r0['scan']}

    # ---- stage A: xTB grid, column by column
    for xi in XI:
        X = np.array(scan[xi]['coords'])
        prev = None
        for rc in RCS:
            kk = key(xi, rc)
            if kk in pts:
                X = np.array(pts[kk]['coords']); prev = pts[kk]; continue
            t0 = time.time()
            cons = rest + [(k['Cd'], k['Hd'], k['Hd'], k['C11'], float(xi)), (k['C6u'], k['S'], float(rc))]
            X, E = optimise_frozen(sym, X, q, frozen=frozen, constraints=cons)
            d = lambda a, b: float(np.linalg.norm(X[k[a]] - X[k[b]]))
            note = ''
            if prev is not None and abs(E - prev['E_xtb']) * K > 12:
                note = f'jump {(E - prev["E_xtb"]) * K:+.1f} kcal/mol'
            pts[kk] = dict(xi=xi, rcs=rc, E_xtb=float(E), coords=X.tolist(), CdH=d('Cd', 'Hd'), HC11=d('Hd', 'C11'),
                           CS=d('C6u', 'S'), N5C11=d('N5', 'C11'), note=note)
            prev = pts[kk]; save(res)
            log(f'xtb xi={xi:+.1f} r={rc:.2f}: E={E:.6f} C6-S={pts[kk]["CS"]:.2f} {note} ({time.time() - t0:.0f}s)')

    # ---- stage B: ONIOM(DFT:xTB) energies, mechanical embedding
    dm = None
    for kk, v in pts.items():
        if 'E_oniom' in v:
            continue
        t0 = time.time()
        X = np.array(v['coords'])
        fs, fX = fragment(sym, X, atoms, cuts)
        mol = gto.M(atom=[(s, x) for s, x in zip(fs, fX)], basis=BASIS, charge=0, unit='Angstrom', verbose=0)
        mf = dft.RKS(mol, xc=XC); mf.conv_tol = 1e-9
        e_hi = float(mf.kernel(dm0=dm)); dm = mf.make_rdm1() if mf.converged else None
        e_lf = float(energy_grad(fs, fX, 0, solvent=None)[0])
        v.update(E_high=e_hi, E_xtb_frag=e_lf, E_oniom=v['E_xtb'] + e_hi - e_lf, high_converged=bool(mf.converged))
        save(res)
        log(f'oniom {kk}: E={v["E_oniom"]:.6f} ({time.time() - t0:.0f}s)')

    # ---- summary: surfaces and the minimax path
    E0x, E0o = pts[key(XI[0], RCS[0])]['E_xtb'], pts[key(XI[0], RCS[0])]['E_oniom']
    grid = {lev: [[(pts[key(xi, rc)][f'E_{lev}'] - (E0x if lev == 'xtb' else E0o)) * K for rc in RCS] for xi in XI]
            for lev in ('xtb', 'oniom')}
    summary = {}
    for lev, G in grid.items():
        G = np.array(G)
        nx, nr = G.shape
        best = {(0, 0): G[0, 0]}; heap = [(G[0, 0], 0, 0)]; parent = {}
        while heap:                                    # minimise the highest energy along the path
            h, i, j = heapq.heappop(heap)
            if h > best.get((i, j), np.inf):
                continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < nr:
                    hb = max(h, G[a, b])
                    if hb < best.get((a, b), np.inf):
                        best[(a, b)] = hb; parent[(a, b)] = (i, j); heapq.heappush(heap, (hb, a, b))
        prod = min(((nx - 1, j) for j in range(nr)), key=lambda c: G[c])
        path, c = [prod], prod
        while c != (0, 0):
            c = parent[c]; path.append(c)
        path = path[::-1]
        top = max(path, key=lambda c: G[c])
        summary[lev] = dict(barrier=float(G[top]), saddle=dict(xi=XI[top[0]], rcs=RCS[top[1]]),
                            product=dict(xi=XI[prod[0]], rcs=RCS[prod[1]], E=float(G[prod])),
                            product_r185=float(G[nx - 1, 0]),
                            path=[dict(xi=XI[i], rcs=RCS[j], E=float(G[i, j])) for i, j in path], grid=G.tolist())
        log(f"{lev}: minimax barrier {G[top]:.2f} kcal/mol at xi={XI[top[0]]:+.1f}, r(C6-S)={RCS[top[1]]:.2f}; "
            f"product corner {G[prod]:+.2f} at r={RCS[prod[1]]:.2f} (C-S intact: {G[nx - 1, 0]:+.2f})")
    res['summary'] = summary; save(res)


if __name__ == '__main__':
    main()
