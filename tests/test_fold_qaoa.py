import numpy as np
from chemoqbit.fold.lattice import ground_states, n_qubits
from chemoqbit.fold.qaoa import (cost_vector, qaoa_state, fixed_qaoa_state, walsh_coefficients,
                                 gate_model_cost, qaoa_program)
from chemoqbit.isa import encode, decode


def _problem():
    emin, gs, E, st = ground_states('DRVYIHPF')
    C, _ = cost_vector(E)
    return C, n_qubits(8), gs


def test_walsh_expansion_reconstructs_cost():
    C, n, _ = _problem()
    c = walsh_coefficients(C)
    x = np.arange(2 ** n)
    rec = np.zeros_like(C)
    for S in np.flatnonzero(np.abs(c) > 1e-14):
        par = np.array([bin(int(v) & int(S)).count('1') & 1 for v in x])
        rec += c[S] * (1 - 2 * par)
    assert np.allclose(rec, C)


def test_qaoa_program_via_isa_matches_direct_state():
    C, n, _ = _problem()
    gam, bet = [0.4, 0.9], [0.6, 0.3]
    prog = decode(encode(qaoa_program(n, gam, bet)))
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    for g in prog:
        if g[0] == 'H':
            pass
        if g[0] == 'DIAG':
            psi = psi * np.exp(-1j * g[2] * C)
        else:
            from chemoqbit.circuit import FloatSim
            psi = FloatSim(n)._apply(psi, g) if g[0] != 'DIAG' else psi
    ref = qaoa_state(C, n, gam, bet)
    assert abs(abs(np.vdot(ref, psi)) - 1) < 1e-6


def test_fixed_point_qaoa_close_to_float():
    C, n, gs = _problem()
    gam, bet = [0.4, 0.9, 1.2], [0.6, 0.3, 0.2]
    a = qaoa_state(C, n, gam, bet)
    b = fixed_qaoa_state(C, n, gam, bet)
    b /= np.linalg.norm(b)
    assert 1 - abs(np.vdot(a, b)) ** 2 < 1e-6
