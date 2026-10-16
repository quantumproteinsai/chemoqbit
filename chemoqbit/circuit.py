"""UCCSD ansatz, gate-level compilation and two state-vector back-ends.

Back-ends
---------
* ``FloatSim``  : complex128 reference.
* ``FixedSim``  : emulates the qsim-pynq datapath -- amplitudes held as signed
  integers with ``frac_bits`` fractional bits (default 17 -> 18-bit words that
  fit a Xilinx DSP48E1 18x25 multiplier), gate coefficients quantised to
  ``coef_bits``, round-to-nearest and saturation after every gate. CNOT is a
  pure permutation of addresses and is therefore exact in hardware.

Qubit 0 is the most significant bit of the basis index (same convention as
``pauli.to_sparse``).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .pauli import fermion_term, op_add, op_scale


# ---------------------------------------------------------------- excitations
def uccsd_excitations(n_qubits, n_elec):
    """Spin-conserving singles and doubles out of the HF determinant
    (interleaved spin orbitals; the first n_elec are occupied)."""
    occ = list(range(n_elec))
    vir = list(range(n_elec, n_qubits))
    singles = [(i, a) for i in occ for a in vir if i % 2 == a % 2]
    doubles = []
    for x, i in enumerate(occ):
        for j in occ[x+1:]:
            for y, a in enumerate(vir):
                for b in vir[y+1:]:
                    if (i % 2 + j % 2) == (a % 2 + b % 2):
                        doubles.append((i, j, a, b))
    return singles, doubles


def excitation_generator(exc, n):
    """Anti-Hermitian generator G = T - T^dag as {pauli: real c} with G = i*sum c P."""
    if len(exc) == 2:
        i, a = exc
        T = fermion_term([(a, 1), (i, 0)], n)
        Td = fermion_term([(i, 1), (a, 0)], n)
    else:
        i, j, a, b = exc
        T = fermion_term([(a, 1), (b, 1), (j, 0), (i, 0)], n)
        Td = fermion_term([(i, 1), (j, 1), (b, 0), (a, 0)], n)
    G = op_add(T, op_scale(Td, -1))
    out = {}
    for s, c in G.items():
        assert abs(c.real) < 1e-12, "generator must be purely imaginary"
        out[s] = c.imag
    return out


# ----------------------------------------------------------- pauli rotations
_GEN_CACHE = {}
def _pauli_masks(s):
    """(flip_mask, z_mask, n_y) for fast application of a Pauli string."""
    n = len(s)
    flip = zmask = 0
    ny = 0
    for q, ch in enumerate(s):
        bit = 1 << (n - 1 - q)
        if ch in 'XY':
            flip |= bit
        if ch in 'ZY':
            zmask |= bit
        if ch == 'Y':
            ny += 1
    return flip, zmask, ny


def apply_pauli(psi, masks, idx):
    """P|psi> for P given as (flip, zmask, ny) masks, or as a precomputed
    (perm, phase) pair from ``pauli_gather`` (much faster in loops)."""
    if len(masks) == 2:
        perm, ph = masks
        return psi[perm] * ph
    flip, zmask, ny = masks
    # P|x> = i^ny (-1)^{popcount(x & zmask)} |x ^ flip>   (Y = iXZ)
    par = _popcount_parity(idx & zmask)
    out = np.empty_like(psi)
    out[idx ^ flip] = psi * ((1j) ** ny) * (1 - 2 * par)
    return out


def pauli_gather(masks, n):
    """Precompute out = psi[perm] * phase  (perm = x ^ flip is an involution)."""
    flip, zmask, ny = masks
    idx = np.arange(2 ** n)
    perm = idx ^ flip
    ph = ((1j) ** ny) * (1 - 2 * _popcount_parity(perm & zmask))
    return perm, ph.astype(complex)


def _popcount_parity(v):
    v = v.copy()
    p = np.zeros_like(v)
    while np.any(v):
        p ^= v & 1
        v >>= 1
    return p


@dataclass
class Ansatz:
    n: int
    n_elec: int
    excitations: list
    rotations: list  # per parameter: list of (masks, c, string)

    @classmethod
    def uccsd(cls, n, n_elec):
        s, d = uccsd_excitations(n, n_elec)
        return cls.from_excitations(n, n_elec, s + d)

    @classmethod
    def from_excitations(cls, n, n_elec, exc):
        rots = []
        for e in exc:
            key = (n, tuple(e))
            if key not in _GEN_CACHE:
                G = excitation_generator(e, n)
                _GEN_CACHE[key] = [(pauli_gather(_pauli_masks(p), n), c, p) for p, c in G.items()]
            rots.append(_GEN_CACHE[key])
        return cls(n, n_elec, list(exc), rots)

    # --------------------------------------------- analytic (adjoint) gradient
    def energy_and_grad(self, theta, H):
        """E and dE/dtheta in one forward + one backward sweep.

        U_k = exp(theta_k G_k), G_k = i*Gt_k with Gt_k = sum_j c_j P_j Hermitian
        (its Pauli terms commute), so dU_k/dtheta_k = i Gt_k U_k and
        dE/dtheta_k = 2 Re < lambda_k | i Gt_k | psi_k >.
        """
        idx = np.arange(2 ** self.n)
        psi = self.state(theta)
        lam = H @ psi
        E = float(np.real(np.vdot(psi, lam)))
        grad = np.zeros(len(theta))
        for k in range(len(theta) - 1, -1, -1):
            rot = self.rotations[k]
            gpsi = np.zeros_like(psi)
            for masks, c, _ in rot:
                gpsi += c * apply_pauli(psi, masks, idx)
            grad[k] = 2 * np.real(np.vdot(lam, 1j * gpsi))
            # undo U_k on both vectors: U_k^dag = prod exp(-i th c P)
            th = theta[k]
            for masks, c, _ in rot:
                t = th * c
                psi = np.cos(t) * psi - 1j * np.sin(t) * apply_pauli(psi, masks, idx)
                lam = np.cos(t) * lam - 1j * np.sin(t) * apply_pauli(lam, masks, idx)
        return E, grad

    @property
    def n_params(self):
        return len(self.excitations)

    def hf_state(self):
        psi = np.zeros(2 ** self.n, dtype=complex)
        idx = sum(1 << (self.n - 1 - q) for q in range(self.n_elec))
        psi[idx] = 1.0
        return psi

    def state(self, theta):
        """exp(theta_k G_k) ... applied to |HF> (exact, float64)."""
        psi = self.hf_state()
        idx = np.arange(2 ** self.n)
        for th, rot in zip(theta, self.rotations):
            if th == 0.0:
                continue
            for masks, c, _ in rot:
                # exp(i th c P) = cos + i sin P
                t = th * c
                psi = np.cos(t) * psi + 1j * np.sin(t) * apply_pauli(psi, masks, idx)
        return psi

    # ---------------------------------------------------- gate compilation
    def gates(self, theta):
        """Compile to {H, RX, RZ, CNOT}: list of tuples."""
        g = []
        for th, rot in zip(theta, self.rotations):
            for _, c, s in rot:
                g += pauli_exp_gates(s, -th * c)  # exp(i th c P) = exp(-i t P), t=-th c
        return g


def pauli_exp_gates(s, t):
    """exp(-i t P) via basis change + CNOT ladder + RZ(2t)."""
    act = [q for q, ch in enumerate(s) if ch != 'I']
    if not act:
        return []  # global phase
    pre, post = [], []
    for q in act:
        if s[q] == 'X':
            pre.append(('H', q)); post.append(('H', q))
        elif s[q] == 'Y':
            pre.append(('RX', q, np.pi / 2)); post.append(('RX', q, -np.pi / 2))
    ladder = [('CNOT', act[k], act[k+1]) for k in range(len(act) - 1)]
    return pre + ladder + [('RZ', act[-1], 2 * t)] + ladder[::-1] + post[::-1]


def gate_counts(gates):
    out = {}
    for g in gates:
        out[g[0]] = out.get(g[0], 0) + 1
    return out


# ------------------------------------------------------------- simulators
class FloatSim:
    def __init__(self, n):
        self.n = n

    def run(self, psi, gates):
        psi = psi.astype(complex).copy()
        for g in gates:
            psi = self._apply(psi, g)
        return psi

    def _one(self, psi, q, U):
        s = psi.reshape(2 ** q, 2, -1)
        return np.einsum('ab,ibj->iaj', U, s).reshape(-1)

    def _apply(self, psi, g):
        if g[0] == 'CNOT':
            c, t = g[1], g[2]
            s = psi.reshape([2] * self.n).copy()
            idx = [slice(None)] * self.n
            idx[c] = 1
            sub = s[tuple(idx)]
            tt = t - 1 if t > c else t
            s[tuple(idx)] = np.flip(sub, axis=tt)
            return s.reshape(-1)
        return self._one(psi, g[1], gate_matrix(g))


def gate_matrix(g):
    if g[0] == 'H':
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    if g[0] == 'RX':
        a = g[2] / 2
        return np.array([[np.cos(a), -1j*np.sin(a)], [-1j*np.sin(a), np.cos(a)]])
    if g[0] == 'RZ':
        a = g[2] / 2
        return np.diag([np.exp(-1j*a), np.exp(1j*a)])
    raise ValueError(g)


class FixedSim(FloatSim):
    """Bit-accurate emulation of a fixed-point state-vector datapath."""

    def __init__(self, n, frac_bits=17, word_bits=18, coef_bits=17):
        super().__init__(n)
        self.F, self.W, self.C = frac_bits, word_bits, coef_bits
        self.lim = 2 ** (word_bits - 1) - 1

    def q(self, x):
        """round re/im parts to the amplitude grid, with saturation."""
        s = 2.0 ** self.F
        re = np.clip(np.rint(x.real * s), -self.lim, self.lim)
        im = np.clip(np.rint(x.imag * s), -self.lim, self.lim)
        return (re + 1j * im) / s

    def qc(self, U):
        s = 2.0 ** self.C
        return (np.rint(U.real * s) + 1j * np.rint(U.imag * s)) / s

    def run(self, psi, gates):
        psi = self.q(psi)
        for g in gates:
            if g[0] == 'CNOT':
                psi = FloatSim._apply(self, psi, g)          # exact permutation
            else:
                psi = self.q(self._one(psi, g[1], self.qc(gate_matrix(g))))
        return psi
