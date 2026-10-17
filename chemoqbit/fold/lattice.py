"""Coarse-grained lattice protein on the tetrahedral (diamond) lattice.

One bead per residue (Calpha). Bonds alternate between the two sublattices:
from an even bead the bond vector is +t[k], from an odd bead -t[k], with

    t = (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)

so every bond angle is the tetrahedral 109.47 deg. Turn k_i in {0,1,2,3} is
the bond from bead i to bead i+1. A chain doubles back when two consecutive
turns are equal, so valid chains have k_i != k_{i-1}.

Qubit encoding (dense, 2 qubits per turn, as in Robert et al., npj Quantum
Inf. 7, 38 (2021)): turns 0 and 1 are fixed (k0=1, k1=0) to remove the
rotational symmetry, leaving 2(N-3) qubits for N residues. Unlike that
paper, no interaction qubits are used: the energy of every bitstring is
evaluated directly, giving a diagonal cost table of length 2^(2(N-3)).
That table is exactly what a state-vector FPGA can hold and apply in one
pass (see chemoqbit.fold.qaoa).

Energy = sum over non-bonded lattice-neighbour pairs of the Miyazawa-Jernigan
contact energy (RT units). Lattice neighbours sit at squared distance 3.
On the bipartite diamond lattice, |i-j| must be odd and >= 5 for a contact.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

T = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.int64)
FIXED = (1, 0)
_DATA = Path(__file__).parent / 'data' / 'mj_matrix.txt'


def load_mj():
    lines = _DATA.read_text().split('\n')
    aa = lines[0].split()
    M = np.array([[float(v) for v in ln.split()] for ln in lines[1:] if ln.strip()])
    M = np.triu(M) + np.triu(M, 1).T
    return aa, M


def pair_energies(seq):
    aa, M = load_mj()
    idx = [aa.index(c) for c in seq]
    return M[np.ix_(idx, idx)]


def n_qubits(N):
    return 2 * (N - 3)


def decode_turns(bits_int, N):
    """integer bitstring (qubit 0 = MSB) -> full turn list of length N-1."""
    nq = n_qubits(N)
    free = [(bits_int >> (nq - 2 - 2 * j)) & 3 for j in range(N - 3)]
    return list(FIXED) + free


def coords(turns):
    X = np.zeros((len(turns) + 1, 3), dtype=np.int64)
    for i, k in enumerate(turns):
        X[i + 1] = X[i] + (T[k] if i % 2 == 0 else -T[k])
    return X


def analyse(turns, E_pair):
    """returns (status, energy, contacts). status: 'valid', 'backtrack', 'overlap'."""
    if any(turns[i] == turns[i - 1] for i in range(1, len(turns))):
        return 'backtrack', None, []
    X = coords(turns)
    if len({tuple(x) for x in X}) < len(X):
        return 'overlap', None, []
    N = len(X)
    contacts, E = [], 0.0
    for i in range(N):
        for j in range(i + 5, N, 2):
            if int(np.sum((X[i] - X[j]) ** 2)) == 3:
                contacts.append((i, j))
                E += E_pair[i, j]
    return 'valid', E, contacts


def cost_table(seq):
    """Diagonal cost over all 2^n bitstrings, vectorised.

    Returns E (float, NaN for invalid strings) and a status array
    (0 valid, 1 backtrack, 2 overlap).
    """
    N = len(seq)
    nq = n_qubits(N)
    dim = 1 << nq
    x = np.arange(dim, dtype=np.int64)
    turns = np.empty((dim, N - 1), dtype=np.int64)
    turns[:, 0], turns[:, 1] = FIXED
    for j in range(N - 3):
        turns[:, 2 + j] = (x >> (nq - 2 - 2 * j)) & 3
    status = np.zeros(dim, dtype=np.int8)
    back = np.any(turns[:, 1:] == turns[:, :-1], axis=1)
    status[back] = 1
    sign = np.where(np.arange(N - 1) % 2 == 0, 1, -1)
    steps = T[turns] * sign[None, :, None]
    X = np.concatenate([np.zeros((dim, 1, 3), dtype=np.int64), np.cumsum(steps, axis=1)], axis=1)
    overlap = np.zeros(dim, dtype=bool)
    for i in range(N):
        for j in range(i + 2, N, 2):     # same sublattice only can coincide
            overlap |= np.all(X[:, i] == X[:, j], axis=1)
    status[(~back) & overlap] = 2
    Ep = pair_energies(seq)
    E = np.zeros(dim)
    for i in range(N):
        for j in range(i + 5, N, 2):
            d2 = np.sum((X[:, i] - X[:, j]) ** 2, axis=1)
            E += np.where(d2 == 3, Ep[i, j], 0.0)
    E[status != 0] = np.nan
    return E, status


def ground_states(seq):
    E, st = cost_table(seq)
    emin = np.nanmin(E)
    gs = np.flatnonzero(np.isclose(E, emin))
    return emin, gs, E, st


def rmsd_to(X_lattice, X_ref, bond=3.8):
    """Kabsch RMSD after scaling lattice bonds (sqrt 3) to `bond` angstrom;
    also tries the mirror image (lattice folds are chiral-ambiguous)."""
    A = X_lattice * (bond / np.sqrt(3.0))
    best = np.inf
    for mirror in (1, -1):
        P = A * np.array([1, 1, mirror])
        P = P - P.mean(0)
        Q = X_ref - X_ref.mean(0)
        H = P.T @ Q
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        best = min(best, float(np.sqrt(np.mean(np.sum((P @ R.T - Q) ** 2, axis=1)))))
    return best
