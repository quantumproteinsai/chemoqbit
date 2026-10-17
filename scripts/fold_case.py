"""Lattice folding with QAOA on the qsim-pynq emulator -> results/fold.json

For each peptide: exact ground states, RMSD to experiment (chignolin),
QAOA ground-state probability vs depth p (INTERP warm start, Zhou et al.,
PRX 10, 021067 (2020)), fixed-point check, gate-model cost of one cost layer.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit.fold.lattice import ground_states, n_qubits, decode_turns, coords, analyse, pair_energies, rmsd_to
from chemoqbit.fold.qaoa import cost_vector, optimise, gate_model_cost, qaoa_state, fixed_qaoa_state

PEPTIDES = [('angiotensin II', 'DRVYIHPF', 10),
            ('chignolin', 'GYDPETGTWG', 10),
            ('head activator neuropeptide', 'QPPGGSKVILF', 8)]
OUT = ROOT / 'results' / 'fold.json'
res = json.loads(OUT.read_text()) if OUT.exists() else {}


def interp(x, p):
    """INTERP: angles for depth p+1 from the optimum at depth p."""
    g, b = x[:p], x[p:]
    def up(v):
        w = np.zeros(p + 1)
        for i in range(p + 1):
            lo = v[i - 1] if i - 1 >= 0 else 0.0
            hi = v[i] if i < p else 0.0
            w[i] = (i / p) * lo + ((p - i) / p) * hi
        return w
    return np.r_[up(g), up(b)]


ref = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                for l in open(ROOT / 'chemoqbit/fold/data/1uao_ca.pdb') if l.startswith('ATOM')])

for name, seq, pmax in PEPTIDES:
    if name in res and len(res[name]['qaoa']) >= pmax:
        print(name, 'done'); continue
    N, n = len(seq), n_qubits(len(seq))
    emin, gs, E, st = ground_states(seq)
    C, scale = cost_vector(E)
    tr = decode_turns(int(gs[0]), N)
    _, _, contacts = analyse(tr, pair_energies(seq))
    r = res.setdefault(name, dict(sequence=seq, N=N, qubits=n, states=int(2 ** n),
                                  valid=int(np.sum(st == 0)), e_min_RT=float(emin),
                                  degeneracy=int(len(gs)), contacts=[list(c) for c in contacts],
                                  p_random=float(len(gs) / 2 ** n),
                                  gate_model=gate_model_cost(C), qaoa=[]))
    if seq == 'GYDPETGTWG':
        valid = np.flatnonzero(st == 0)
        rm = np.array([rmsd_to(coords(decode_turns(int(b), N)), ref) for b in valid])
        r['rmsd'] = dict(ground_states=[rmsd_to(coords(decode_turns(int(b), N)), ref) for b in gs],
                         best_possible=float(rm.min()), median=float(np.median(rm)),
                         rank_of_best_gs=int(np.sum(rm < min(rm[np.isin(valid, gs)]))) + 1,
                         n_valid=int(len(valid)))
    x = None
    for p in range(1, pmax + 1):
        if p <= len(r['qaoa']):
            x = np.array(r['qaoa'][p - 1]['x']); continue
        t0 = time.time()
        x0 = interp(x, p - 1) if x is not None else None
        o = optimise(C, n, p, gs, objective='energy', x0=x0, restarts=1 if x0 is not None else 4)
        x = o['x']
        psi_fx = fixed_qaoa_state(C, n, x[:p], x[p:])
        pr = np.abs(psi_fx) ** 2
        pr /= pr.sum()
        o['p_gs_fixed'] = float(pr[gs].sum())
        o['x'] = [float(v) for v in x]
        o['seconds'] = time.time() - t0
        r['qaoa'].append(o)
        OUT.write_text(json.dumps(res, indent=1))
        print(f"{name:28s} n={n:2d} p={p:2d} P_gs={o['p_gs']:.3f} fixed={o['p_gs_fixed']:.3f} "
              f"(random {r['p_random']:.4f}) {o['seconds']:.0f}s", flush=True)
