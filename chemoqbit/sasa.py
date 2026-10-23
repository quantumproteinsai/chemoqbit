"""Solvent-accessible surface area by the Shrake-Rupley construction.

Needed to decide which pair of beta-sheets in an amyloid microcrystal is the
steric zipper. Sawaya et al. (Nature 447, 453 (2007)) define one as a pair of
sheets whose facing side chains interdigitate in a DRY interface, so the test
is quantitative: the pair that buries the most side-chain area with no water
between the sheets. Counting close contacts, as a first attempt here did, does
not distinguish that from an edge-on packing contact.

Heavy atoms only, with the usual Bondi radii and a 1.4 A water probe.
"""
from __future__ import annotations

import numpy as np

RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 'F': 1.47, 'H': 1.20}
PROBE = 1.4


def _sphere(n):
    """n roughly uniform points on the unit sphere (golden spiral)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


def sasa(sym, X, n_points=256, probe=PROBE):
    """Per-atom solvent-accessible area in A^2."""
    X = np.asarray(X, float)
    r = np.array([RADII.get(s, 1.70) for s in sym]) + probe
    pts = _sphere(n_points)
    out = np.zeros(len(X))
    # neighbour list: an atom can only be occluded by one within r_i + r_j
    rmax = r.max()
    for i in range(len(X)):
        d = np.linalg.norm(X - X[i], axis=1)
        nb = np.flatnonzero((d < r[i] + rmax) & (np.arange(len(X)) != i))
        if len(nb) == 0:
            out[i] = 4 * np.pi * r[i] ** 2
            continue
        test = X[i] + r[i] * pts
        acc = np.ones(len(pts), bool)
        for j in nb:
            acc &= np.linalg.norm(test - X[j], axis=1) >= r[j]
            if not acc.any():
                break
        out[i] = 4 * np.pi * r[i] ** 2 * acc.sum() / len(pts)
    return out


def buried(sym_a, Xa, sym_b, Xb, **kw):
    """Area buried when the two groups come together, total in A^2.
    buried = SASA(A) + SASA(B) - SASA(A+B), summed over both."""
    a = sasa(sym_a, Xa, **kw).sum()
    b = sasa(sym_b, Xb, **kw).sum()
    both = sasa(list(sym_a) + list(sym_b), np.vstack([Xa, Xb]), **kw).sum()
    return a + b - both


def buried_per_atom(sym_a, Xa, sym_b, Xb, **kw):
    """Area each atom of A loses to B (A^2), for splitting into side chain and backbone."""
    a = sasa(sym_a, Xa, **kw)
    both = sasa(list(sym_a) + list(sym_b), np.vstack([Xa, Xb]), **kw)[:len(sym_a)]
    return a - both
