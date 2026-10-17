"""QAOA for lattice folding on a state-vector machine.

Cost layer  U_C(g) = exp(-i g C)   with C the diagonal cost table
Mixer       U_B(b) = prod_q exp(-i b X_q) = prod_q RX_q(2b)

On qsim-pynq the cost layer is ONE instruction, DIAG: the fabric streams the
state vector past a cost table held in block RAM and multiplies amplitude x
by exp(-i g C[x]). On gate-based hardware the same layer must be compiled to
exp(-i g c_S Z_S) for every Z-string S in the Walsh expansion of C, which is
counted by ``gate_model_cost``.

Penalty-free convention: invalid bitstrings (backtracking or overlapping
chains) get cost 0, the same as an extended chain with no contacts. Every
valid compact fold has negative cost, so the ground state is unchanged and
no penalty weight needs tuning.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def cost_vector(E_table, invalid=0.0):
    C = np.where(np.isnan(E_table), invalid, E_table)
    scale = np.max(np.abs(C))
    return C / scale, scale            # normalised to [-1, 1]


def apply_mixer(psi, n, beta):
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape([2] * n)
    for q in range(n):
        a = np.take(psi, 0, axis=q)
        b = np.take(psi, 1, axis=q)
        psi = np.stack([c * a + s * b, s * a + c * b], axis=q)
    return psi.reshape(-1)


def qaoa_state(C, n, gammas, betas):
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * C)
        psi = apply_mixer(psi, n, b)
    return psi


def optimise(C, n, p, gs_idx, objective='energy', x0=None, restarts=4, seed=0):
    """Optimise 2p angles. objective: 'energy' (<C>) or 'gs' (-P_ground)."""
    gs_mask = np.zeros(len(C), bool)
    gs_mask[gs_idx] = True

    def f(x):
        psi = qaoa_state(C, n, x[:p], x[p:])
        pr = np.abs(psi) ** 2
        return float(pr @ C) if objective == 'energy' else -float(pr[gs_mask].sum())

    rng = np.random.default_rng(seed)
    starts = []
    if x0 is not None:
        starts.append(np.asarray(x0))
    # linear ramp (annealing-like) start, then random perturbations of it
    T = 0.75 * p
    ramp = np.r_[np.linspace(1, p, p) / p * T * np.pi / p * 2,
                 (1 - np.linspace(1, p, p) / p + 1 / (2 * p)) * T * np.pi / p / 2]
    starts.append(ramp)
    for _ in range(restarts):
        starts.append(ramp + rng.normal(scale=0.15, size=2 * p))
    best = None
    for s0 in starts:
        r = minimize(f, s0, method='L-BFGS-B', options=dict(maxiter=400))
        if best is None or r.fun < best.fun:
            best = r
    psi = qaoa_state(C, n, best.x[:p], best.x[p:])
    pr = np.abs(psi) ** 2
    return dict(x=best.x, p=p, energy=float(pr @ C), p_gs=float(pr[gs_mask].sum()),
                nfev=int(best.nfev))


def walsh_coefficients(C):
    """C(x) = sum_S c_S prod_{q in S} z_q  with z = +-1 (fast Walsh-Hadamard)."""
    a = C.astype(float).copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), 2 * h):
            x, y = a[i:i + h].copy(), a[i + h:i + 2 * h].copy()
            a[i:i + h], a[i + h:i + 2 * h] = x + y, x - y
        h *= 2
    return a / len(a)


def gate_model_cost(C, tol=1e-12):
    """Gates for ONE cost layer on a gate-based machine with a CNOT ladder per
    Z-string: 2(w-1) CNOTs + 1 RZ for each nonzero term of weight w >= 1."""
    c = walsh_coefficients(C)
    n = int(np.log2(len(C)))
    idx = np.flatnonzero(np.abs(c) > tol)
    w = np.array([bin(int(i)).count('1') for i in idx])
    w = w[w > 0]
    return dict(terms=int(len(w)), max_weight=int(w.max()) if len(w) else 0,
                cnots=int(np.sum(2 * (w - 1))), rz=int(len(w)),
                weight_hist={int(k): int(v) for k, v in zip(*np.unique(w, return_counts=True))},
                n_qubits=n)


# ------------------------------------------------------- FPGA fixed point
def fixed_qaoa_state(C, n, gammas, betas, frac_bits=16, table_bits=16, angle_bits=16):
    """Bit-accurate emulation: 18-bit amplitudes, cost table quantised to
    `table_bits`, phases g*C[x] quantised to the 16-bit angle grid, cos/sin
    and RX coefficients rounded to `frac_bits`, round-to-nearest after every
    instruction."""
    s = 2.0 ** frac_bits
    lim = 2 ** (frac_bits + 1) - 1

    def q(z):
        re = np.clip(np.rint(z.real * s), -lim, lim)
        im = np.clip(np.rint(z.imag * s), -lim, lim)
        return (re + 1j * im) / s

    qc = lambda v: np.rint(v * s) / s
    Cq = np.rint(C * 2 ** (table_bits - 1)) / 2 ** (table_bits - 1)
    lsb = np.pi / 2 ** (angle_bits - 1)
    psi = q(np.full(2 ** n, 2 ** (-n / 2), dtype=complex))
    for g, b in zip(gammas, betas):
        ph = np.rint(((g * Cq + np.pi) % (2 * np.pi) - np.pi) / lsb) * lsb
        psi = q(psi * (qc(np.cos(ph)) - 1j * qc(np.sin(ph))))
        bq = np.rint(b / lsb) * lsb
        c, sn = qc(np.cos(bq)), qc(np.sin(bq))
        psi = psi.reshape([2] * n)
        for k in range(n):
            a0, a1 = np.take(psi, 0, axis=k), np.take(psi, 1, axis=k)
            psi = q(np.stack([c * a0 - 1j * sn * a1, -1j * sn * a0 + c * a1], axis=k))
        psi = psi.reshape(-1)
    return psi


def qaoa_program(n, gammas, betas, slot=0):
    """QAOA as qsim-pynq instructions: H on all qubits, then per layer one
    DIAG (cost table in `slot`) and n RX(2 beta)."""
    g = [('H', q) for q in range(n)]
    for gm, b in zip(gammas, betas):
        g.append(('DIAG', slot, gm))
        g += [('RX', q, 2 * b) for q in range(n)]
    return g
