"""Two-layer ONIOM energy and gradient with mechanical embedding:

    E = E_xTB(cluster) + E_high(fragment) - E_xTB(fragment)

The fragment is a set of cluster atoms plus link hydrogens, one per cut bond
(a inside, b outside), placed at a fixed distance d from a along a->b:

    X_H = X_a + d u,   u = (X_b - X_a) / |X_b - X_a|

so the link atom's gradient is passed back to a and b through the Jacobian
dX_H/dX_a = I - (d/|v|)(I - u u^T),  dX_H/dX_b = (d/|v|)(I - u u^T).
Both fragment terms are in the gas phase; solvation (ALPB) and the
fragment-environment interaction enter through the cluster xTB term only.
Units: Angstrom, Eh, Eh/Angstrom.
"""
from __future__ import annotations

import numpy as np

from .xtb import energy_grad

BOHR = 0.529177210903
CAP = {'C': 1.09, 'N': 1.01, 'O': 0.97, 'S': 1.34}


def link_positions(sym, X, cuts):
    out = []
    for a, b in cuts:
        v = X[b] - X[a]
        out.append(X[a] + CAP[sym[a]] * v / np.linalg.norm(v))
    return np.array(out).reshape(-1, 3)


def fragment(sym, X, atoms, cuts):
    """symbols and coordinates of the fragment: `atoms` in order, then the links"""
    fsym = [sym[i] for i in atoms] + ['H'] * len(cuts)
    return fsym, np.vstack([X[atoms], link_positions(sym, X, cuts)])


def scatter_gradient(sym, X, atoms, cuts, gf):
    """fragment gradient (atoms, then links) -> cluster gradient"""
    G = np.zeros_like(X)
    G[atoms] += gf[:len(atoms)]
    for (a, b), gh in zip(cuts, gf[len(atoms):]):
        v = X[b] - X[a]; nv = np.linalg.norm(v); u = v / nv
        P = (np.eye(3) - np.outer(u, u)) * (CAP[sym[a]] / nv)
        G[a] += gh - P @ gh
        G[b] += P @ gh
    return G


class HighLevel:
    """Restricted DFT (or HF) energy and gradient of a closed-shell fragment with
    PySCF, restarting each SCF from the previous density (the geometry changes
    little between calls in an optimisation or a finite-difference Hessian)."""

    def __init__(self, xc='wb97x-d3bj', basis='def2-svp', charge=0, conv_tol=1e-10):
        self.xc, self.basis, self.charge, self.conv_tol = xc, basis, charge, conv_tol
        self.dm = None
        self.calls = 0

    def __call__(self, fsym, fX):
        from pyscf import gto, dft, scf
        mol = gto.M(atom=[(s, x) for s, x in zip(fsym, fX)], basis=self.basis,
                    charge=self.charge, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol) if self.xc.lower() == 'hf' else dft.RKS(mol, xc=self.xc)
        mf.conv_tol = self.conv_tol
        e = mf.kernel(dm0=self.dm)
        if not mf.converged:
            e = mf.newton().kernel(mf.mo_coeff, mf.mo_occ)
        g = mf.nuc_grad_method().kernel() / BOHR
        self.dm = mf.make_rdm1()
        self.calls += 1
        return float(e), np.asarray(g)


def oniom_fn(sym, charge, atoms, cuts, high, extra=None):
    """fn(X) -> (E, G) for the cluster; `extra` adds a (E, G) term, e.g. restraints"""
    atoms = list(atoms)

    def fn(X):
        X = np.asarray(X, float)
        E_lo, G_lo = energy_grad(sym, X, charge)
        fsym, fX = fragment(sym, X, atoms, cuts)
        e_hi, g_hi = high(fsym, fX)
        e_lf, g_lf = energy_grad(fsym, fX, 0, solvent=None)
        E = E_lo + e_hi - e_lf
        G = G_lo + scatter_gradient(sym, X, atoms, cuts, g_hi - g_lf)
        if extra is not None:
            Ep, Gp = extra(X)
            E, G = E + Ep, G + Gp
        return E, G
    return fn
