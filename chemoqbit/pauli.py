"""Minimal Pauli-string algebra and the Jordan-Wigner transform.

A Pauli operator is a dict {pauli_string: complex_coeff}, where pauli_string
is a tuple of length n over {'I','X','Y','Z'} and qubit 0 is the leftmost
character. Spin orbitals are interleaved: 2p = alpha, 2p+1 = beta.

Kept deliberately small and dependency-free so that every step can be read
(and reproduced) in a textbook chapter.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# single-qubit product table: (a, b) -> (phase, c) with a*b = phase*c
_MUL = {
    ('I', 'I'): (1, 'I'), ('I', 'X'): (1, 'X'), ('I', 'Y'): (1, 'Y'), ('I', 'Z'): (1, 'Z'),
    ('X', 'I'): (1, 'X'), ('X', 'X'): (1, 'I'), ('X', 'Y'): (1j, 'Z'), ('X', 'Z'): (-1j, 'Y'),
    ('Y', 'I'): (1, 'Y'), ('Y', 'X'): (-1j, 'Z'), ('Y', 'Y'): (1, 'I'), ('Y', 'Z'): (1j, 'X'),
    ('Z', 'I'): (1, 'Z'), ('Z', 'X'): (1j, 'Y'), ('Z', 'Y'): (-1j, 'X'), ('Z', 'Z'): (1, 'I'),
}


def mul_strings(a, b):
    phase = 1
    out = []
    for x, y in zip(a, b):
        p, c = _MUL[(x, y)]
        phase *= p
        out.append(c)
    return phase, tuple(out)


def op_mul(A, B, tol=1e-14):
    C = {}
    for sa, ca in A.items():
        for sb, cb in B.items():
            ph, s = mul_strings(sa, sb)
            C[s] = C.get(s, 0) + ph * ca * cb
    return {s: c for s, c in C.items() if abs(c) > tol}


def op_add(*ops, tol=1e-14):
    C = {}
    for A in ops:
        for s, c in A.items():
            C[s] = C.get(s, 0) + c
    return {s: c for s, c in C.items() if abs(c) > tol}


def op_scale(A, k):
    return {s: k * c for s, c in A.items()}


def jw_ladder(p, n, dagger):
    """Jordan-Wigner image of a_p^dagger (dagger=True) or a_p."""
    zs = ('Z',) * p
    rest = ('I',) * (n - p - 1)
    sx = zs + ('X',) + rest
    sy = zs + ('Y',) + rest
    # a^dag = (X - iY)/2 ,  a = (X + iY)/2
    return {sx: 0.5, sy: (-0.5j if dagger else 0.5j)}


def fermion_term(indices, n):
    """Product of ladder ops, e.g. [(p,1),(q,0)] -> a_p^dag a_q."""
    out = {('I',) * n: 1.0}
    for idx, dag in indices:
        out = op_mul(out, jw_ladder(idx, n, bool(dag)))
    return out


def molecular_qubit_hamiltonian(ecore, h1, h2, tol=1e-10):
    """Spatial-orbital integrals (h2 in chemists' notation (pq|rs)) -> Pauli op.

    H = E_core + sum_pq h_pq a+_p a_q + 1/2 sum_pqrs (pq|rs) a+_p a+_r a_s a_q
    """
    norb = h1.shape[0]
    n = 2 * norb
    H = {('I',) * n: complex(ecore)}
    # build spin-orbital integrals
    for p in range(norb):
        for q in range(norb):
            if abs(h1[p, q]) < tol:
                continue
            for s in (0, 1):
                H = op_add(H, op_scale(fermion_term([(2*p+s, 1), (2*q+s, 0)], n), h1[p, q]))
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s_ in range(norb):
                    v = h2[p, q, r, s_]
                    if abs(v) < tol:
                        continue
                    for a in (0, 1):
                        for b in (0, 1):
                            P, Q, R, S = 2*p+a, 2*q+a, 2*r+b, 2*s_+b
                            if P == R or Q == S:
                                continue
                            term = fermion_term([(P, 1), (R, 1), (S, 0), (Q, 0)], n)
                            H = op_add(H, op_scale(term, 0.5 * v))
    # Hamiltonian is Hermitian: coefficients must be real
    imag = max((abs(c.imag) for c in H.values()), default=0)
    assert imag < 1e-9, f"non-Hermitian residue {imag}"
    return {s: c.real for s, c in H.items() if abs(c) > tol}


_PM = {
    'I': sp.identity(2, format='csr', dtype=complex),
    'X': sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
    'Y': sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
    'Z': sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex)),
}


def to_sparse(op, n):
    """Pauli op -> sparse 2^n matrix. Qubit 0 is the most significant bit."""
    dim = 2 ** n
    M = sp.csr_matrix((dim, dim), dtype=complex)
    for s, c in op.items():
        m = sp.identity(1, format='csr', dtype=complex)
        for ch in s:
            m = sp.kron(m, _PM[ch], format='csr')
        M = M + c * m
    return M


def qubitwise_commuting_groups(op):
    """Greedy grouping of Pauli strings into qubit-wise commuting sets
    (one measurement basis per group on hardware)."""
    groups = []
    for s in sorted(op, key=lambda s: -abs(op[s])):
        if all(ch == 'I' for ch in s):
            continue
        for g in groups:
            if all(all(a == b or a == 'I' or b == 'I' for a, b in zip(s, t)) for t in g):
                g.append(s)
                break
        else:
            groups.append([s])
    return groups
