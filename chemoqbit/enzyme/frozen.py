"""Optimisation with a frozen subset of atoms (cluster-model caps)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .xtb import energy_grad


def penalty(Y, constraints, k=20.0):
    """Harmonic restraint energy (Eh) and gradient (Eh/A). Forms:
    (i, j, d0)             d_ij = d0
    (i, j, l, m, x0)       d_ij - d_lm = x0
    ('min', i, j, d0)      d_ij >= d0 (flat-bottom: no force once d_ij >= d0)"""
    E, G = 0.0, np.zeros_like(Y)
    for c in constraints:
        if c[0] == 'min':
            _, i, j, d0 = c
            u = Y[i] - Y[j]; d = np.linalg.norm(u)
            if d < d0:
                E += k * (d - d0) ** 2
                g = 2 * k * (d - d0) * u / d
                G[i] += g; G[j] -= g
        elif len(c) == 3:
            i, j, d0 = c
            u = Y[i] - Y[j]; d = np.linalg.norm(u)
            E += k * (d - d0) ** 2
            g = 2 * k * (d - d0) * u / d
            G[i] += g; G[j] -= g
        else:
            i, j, l, m, x0 = c
            u1 = Y[i] - Y[j]; d1 = np.linalg.norm(u1)
            u2 = Y[l] - Y[m]; d2 = np.linalg.norm(u2)
            ff = 2 * k * ((d1 - d2) - x0)
            E += k * ((d1 - d2) - x0) ** 2
            G[i] += ff * u1 / d1; G[j] -= ff * u1 / d1
            G[l] -= ff * u2 / d2; G[m] += ff * u2 / d2
    return E, G


def optimise_frozen(sym, X, charge=0, frozen=(), constraints=(), k=20.0,
                    gtol=3e-4, maxiter=3000):
    """Relax all atoms except `frozen` (indices). `constraints` as in `penalty`.
    Returns (X, E_xtb) with the restraint energy excluded."""
    X = np.array(X, float)
    mask = np.ones(len(X), bool)
    mask[list(frozen)] = False
    free = np.flatnonzero(mask)

    last = {}

    def f(v):
        Y = X.copy(); Y[free] = v.reshape(-1, 3)
        try:
            E, G = energy_grad(sym, Y, charge)
        except RuntimeError:
            # the xTB SCF failed at a trial geometry, typically an overlong first
            # line-search step against a stiff restraint: report a high energy so
            # the line search backs off, instead of aborting the optimisation
            if not last:
                raise
            return last['f'] + 1.0, last['g']
        Ep, Gp = penalty(Y, constraints, k)
        last['f'], last['g'] = E + Ep, (G + Gp)[free].ravel()
        return last['f'], last['g']

    r = minimize(f, X[free].ravel(), jac=True, method='L-BFGS-B',
                 options=dict(maxiter=maxiter, gtol=gtol))
    X[free] = r.x.reshape(-1, 3)
    return X, float(energy_grad(sym, X, charge)[0])
