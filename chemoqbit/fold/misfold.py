"""Misfolding and its rescue on the diamond lattice.

The contacts of a conformation do not depend on the sequence, so they are
enumerated once per chain length (``Ensemble``) and the energy of any
sequence is a matrix product, E = A @ e(seq), with A[c, p] = 1 when
conformation c makes contact p. This makes exhaustive single- and
double-mutant scans cheap.

Native state: the conformations whose contact map equals the wild type's
ground-state map. Its population at temperature T (energies in RT units of
the Miyazawa-Jernigan matrix) gives the folding free energy
dG = -T ln(P_N / (1 - P_N)); dG < 0 means folded.

Pharmacological chaperone: a ligand of one bead (an empty lattice site) or
two bonded beads that gains -eps per contact with a chosen set of 'pocket'
residues, those around its best placement in the native fold. Each
conformation c binds with energy B_c = -eps * n_c, n_c the largest number
of pocket residues one placement in c touches. At ligand activity x
(concentration over the reference dissociation constant) the conformation's
weight becomes exp(-E_c/T) (1 + x exp(-B_c/T)), i.e. an effective diagonal
energy E_c - T ln(1 + x exp(-B_c/T)) that QAOA can use unchanged.
"""
from __future__ import annotations

import numpy as np

from .lattice import T as TV, FIXED, n_qubits, pair_energies


class Ensemble:
    """All self-avoiding conformations of an N-bead chain, with their contacts
    and coordinates. ``index`` maps each valid conformation to its bitstring."""

    def __init__(self, N):
        self.N, self.nq = N, n_qubits(N)
        dim = 1 << self.nq
        x = np.arange(dim, dtype=np.int64)
        turns = np.empty((dim, N - 1), dtype=np.int64)
        turns[:, 0], turns[:, 1] = FIXED
        for j in range(N - 3):
            turns[:, 2 + j] = (x >> (self.nq - 2 - 2 * j)) & 3
        ok = ~np.any(turns[:, 1:] == turns[:, :-1], axis=1)
        sign = np.where(np.arange(N - 1) % 2 == 0, 1, -1)
        X = np.concatenate([np.zeros((dim, 1, 3), dtype=np.int64),
                            np.cumsum(TV[turns] * sign[None, :, None], axis=1)], axis=1)
        for i in range(N):
            for j in range(i + 2, N, 2):
                ok &= ~np.all(X[:, i] == X[:, j], axis=1)
        self.index = np.flatnonzero(ok)
        self.X = X[ok]
        self.pairs = [(i, j) for i in range(N) for j in range(i + 5, N, 2)]
        self.A = np.stack([np.sum((self.X[:, i] - self.X[:, j]) ** 2, axis=1) == 3
                           for i, j in self.pairs], axis=1).astype(np.float64)

    def energies(self, seq):
        Ep = pair_energies(seq)
        return self.A @ np.array([Ep[i, j] for i, j in self.pairs])

    def contact_key(self, c):
        return tuple(self.pairs[p] for p in np.flatnonzero(self.A[c]))

    def native_mask(self, key):
        want = np.array([p in key for p in self.pairs], dtype=np.float64)
        return np.all(self.A == want, axis=1)

    def full_table(self, E_valid):
        """Back to a cost vector over all 2^nq bitstrings (NaN = invalid)."""
        out = np.full(1 << self.nq, np.nan)
        out[self.index] = E_valid
        return out


def native_key(ens, seq):
    E = ens.energies(seq)
    gs = np.flatnonzero(np.isclose(E, E.min()))
    keys = {ens.contact_key(c) for c in gs}
    return (keys.pop() if len(keys) == 1 else None), E


def population(E, native, T, extra=None):
    """P_native and dG_fold = -T ln(P_N/(1-P_N)). ``extra``: per-conformation
    free-energy shift (e.g. ligand binding), added to E."""
    G = E if extra is None else E + extra
    w = np.exp(-(G - G.min()) / T)
    pn = float(w[native].sum() / w.sum())
    return pn, float(-T * np.log(pn / (1 - pn))) if 0 < pn < 1 else (np.inf if pn == 0 else -np.inf)


def ground_is_native(E, native, tol=1e-9):
    gs = np.isclose(E, E.min(), atol=tol)
    return bool(np.all(native[gs])), int(gs.sum())


def mutate(seq, pos, aa):
    return seq[:pos] + aa + seq[pos + 1:]


# --------------------------------------------------------------- chaperone ligand
def _neighbour_sites(x, parity):
    """the four lattice sites bonded to a site of sublattice `parity` (0 even)."""
    return x + (TV if parity == 0 else -TV)


def placements(ens, c, beads=1, residues=None):
    """Residue sets contacted by a ligand of 1 bead (one empty site) or 2 beads
    (two bonded empty sites) in conformation c, one tuple per placement that
    touches >= 1 residue. ``residues`` restricts the residues counted."""
    X = ens.X[c]
    occ = {tuple(v) for v in X}
    adj = {}
    for i, x in enumerate(X):
        if residues is not None and i not in residues:
            continue
        for s in _neighbour_sites(x, i % 2):
            s = tuple(int(v) for v in s)
            if s not in occ:
                adj.setdefault(s, set()).add(i)
    if beads == 1:
        return [tuple(sorted(r)) for r in adj.values()]
    out = []
    for s1, r1 in adj.items():
        for s2 in _neighbour_sites(np.array(s1), s1[0] % 2):
            s2 = tuple(int(v) for v in s2)
            if s2 not in occ:
                out.append(tuple(sorted(r1 | adj.get(s2, set()))))
    return out


def best_pocket(ens, native, beads=1):
    """The pocket every native conformation offers: among the residue sets
    (and their subsets) touched by a ligand placement in some native
    conformation, those that all native conformations present in full; the
    largest, then the one fewest conformations present (most selective).
    Returns (pocket, n_c) or (None, None)."""
    from itertools import combinations
    cand = set()
    for c in np.flatnonzero(native):
        for r in placements(ens, c, beads):
            cand |= {s for k in range(1, len(r) + 1) for s in combinations(r, k)}
    best = None
    for r in sorted(cand, key=lambda r: (-len(r), r)):
        if best is not None and len(r) < len(best[0]):
            break
        n = pocket_contacts(ens, r, beads)
        if np.all(n[native] == len(r)):
            score = int(np.sum(n == len(r)))
            if best is None or score < best[2]:
                best = (r, n, score)
    return (best[0], best[1]) if best else (None, None)


def pocket_contacts(ens, pocket, beads=1):
    """n_c: the largest number of pocket residues one ligand placement touches, per conformation."""
    pocket = set(pocket)
    return np.array([max((len(r) for r in placements(ens, c, beads, residues=pocket)), default=0)
                     for c in range(len(ens.X))], dtype=np.int64)


def ligand_shift(n, eps, x, T):
    """free-energy shift per conformation from a ligand at activity x."""
    return -T * np.log1p(x * np.exp(eps * n / T))
