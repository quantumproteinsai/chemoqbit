"""Can QAOA be rescued at 14-16 qubits? -> results/fold_variants.json

Variants, each with INTERP warm starts over depth p:
  x_energy : standard X mixer, minimise <C>              (baseline, as fold_case)
  x_cvar   : X mixer, minimise CVaR_0.1 (Barkoutsos et al., Quantum 4, 256 (2020))
  x_oracle : X mixer, maximise P(ground) directly -- needs the answer, so an
             upper bound on what these circuits can express, not a method
  grover   : Grover-mixer QAOA restricted to valid folds (Baertschi & Eidenbenz,
             IEEE QCE 2020): start in the uniform superposition |s> of valid
             folds, mixer exp(-i b |s><s|), cost exp(-i g C) -- minimise <C>
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit.fold.lattice import ground_states, n_qubits
from chemoqbit.fold.qaoa import cost_vector, apply_mixer

OUT = ROOT / 'results' / 'fold_variants.json'
res = json.loads(OUT.read_text()) if OUT.exists() else {}
PEPS = [('chignolin', 'GYDPETGTWG', 10), ('head activator neuropeptide', 'QPPGGSKVILF', 8)]


def interp(x, p):
    g, b = x[:p], x[p:]
    up = lambda v: np.array([(i / p) * (v[i - 1] if i >= 1 else 0) + ((p - i) / p) * (v[i] if i < p else 0)
                             for i in range(p + 1)])
    return np.r_[up(g), up(b)]


def make(seq):
    emin, gs, E, st = ground_states(seq)
    C, _ = cost_vector(E)
    n = n_qubits(len(seq))
    valid = st == 0
    gsm = np.zeros(len(C), bool); gsm[gs] = True
    return C, n, valid, gsm


def state_x(C, n, x, p):
    psi = np.full(2 ** n, 2 ** (-n / 2), complex)
    for g, b in zip(x[:p], x[p:]):
        psi = apply_mixer(psi * np.exp(-1j * g * C), n, b)
    return psi


def state_grover(C, valid, x, p):
    s = valid / np.sqrt(valid.sum())
    psi = s.astype(complex)
    for g, b in zip(x[:p], x[p:]):
        psi = psi * np.exp(-1j * g * C)
        psi = psi - (1 - np.exp(-1j * b)) * np.vdot(s, psi) * s
    return psi


def cvar(pr, C, alpha=0.1):
    o = np.argsort(C)
    c, w = C[o], pr[o]
    cum = np.cumsum(w)
    k = np.searchsorted(cum, alpha)
    return float((np.sum(c[:k] * w[:k]) + c[k] * (alpha - (cum[k - 1] if k else 0))) / alpha)


for name, seq, pmax in PEPS:
    C, n, valid, gsm = make(seq)
    r = res.setdefault(name, dict(p_random=float(gsm.sum() / 2 ** n),
                                  p_random_valid=float(gsm.sum() / valid.sum())))
    for var in ('x_cvar', 'x_oracle', 'grover'):
        rows = r.setdefault(var, [])
        x = np.array(rows[-1]['x']) if rows else None
        for p in range(len(rows) + 1, pmax + 1):
            t0 = time.time()
            if var == 'grover':
                st = lambda x: state_grover(C, valid, x, p)
            else:
                st = lambda x: state_x(C, n, x, p)
            if var == 'x_cvar':
                f = lambda x: cvar(np.abs(st(x)) ** 2, C)
            elif var == 'x_oracle':
                f = lambda x: -float((np.abs(st(x)) ** 2)[gsm].sum())
            else:
                f = lambda x: float((np.abs(st(x)) ** 2) @ C)
            starts = [interp(x, p - 1)] if x is not None else \
                [np.r_[np.linspace(0.1, 1, p) * 2, np.linspace(1, 0.1, p) * 0.6] + d
                 for d in (0, 0.2, -0.2, 0.5)]
            best = min((minimize(f, s0, method='L-BFGS-B', options=dict(maxiter=300)) for s0 in starts),
                       key=lambda o: o.fun)
            x = best.x
            pg = float((np.abs(st(x)) ** 2)[gsm].sum())
            rows.append(dict(p=p, p_gs=pg, x=[float(v) for v in x], seconds=time.time() - t0))
            OUT.write_text(json.dumps(res, indent=1))
            print(f'{name:28s} {var:8s} p={p:2d} P_gs={pg:.4f} ({time.time() - t0:.0f}s)', flush=True)
