"""Optimisation with a frozen subset of atoms (cluster-model caps)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .xtb import energy_grad


def penalty(Y, constraints, k=20.0):
    """Harmonic restraint energy (Eh) and gradient (Eh/A). Forms:
    (i, j, d0)             d_ij = d0
    (i, j, l, m, x0)       d_ij - d_lm = x0
    ('min', i, j, d0)      d_ij >= d0 (flat-bottom: no force once d_ij >= d0)
    ('tor', i, j, k, l, p0) dihedral(i,j,k,l) = p0 degrees (periodic, k in Eh/rad^2)
    ('relay', [(d,h,a), ...], x0)  sum over hops of (d_dh - d_ha) = x0, so several
                           protons move together: a Grotthuss relay along a hydrogen
                           bond chain, which driving one hop alone cannot describe
                           (it builds an unstabilised hydronium instead)"""
    E, G = 0.0, np.zeros_like(Y)
    for c in constraints:
        if c[0] == 'min':
            _, i, j, d0 = c
            u = Y[i] - Y[j]; d = np.linalg.norm(u)
            if d < d0:
                E += k * (d - d0) ** 2
                g = 2 * k * (d - d0) * u / d
                G[i] += g; G[j] -= g
        elif c[0] == 'relay':
            _, hops, x0 = c
            tot, terms = 0.0, []
            for dn, hy, ac in hops:
                u1 = Y[dn] - Y[hy]; d1 = np.linalg.norm(u1)
                u2 = Y[hy] - Y[ac]; d2 = np.linalg.norm(u2)
                tot += d1 - d2
                terms.append((dn, hy, ac, u1, d1, u2, d2))
            dev = tot - x0
            E += k * dev ** 2
            ff = 2 * k * dev
            for dn, hy, ac, u1, d1, u2, d2 in terms:
                G[dn] += ff * u1 / d1; G[hy] -= ff * u1 / d1
                G[hy] -= ff * u2 / d2; G[ac] += ff * u2 / d2
        elif c[0] == 'tor':
            _, i, j, k_, l_, p0 = c
            b1, b2, b3 = Y[j] - Y[i], Y[k_] - Y[j], Y[l_] - Y[k_]
            n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
            l2 = np.linalg.norm(b2)
            phi = np.arctan2(np.dot(np.cross(n1, n2), b2 / l2), np.dot(n1, n2))
            dp = (phi - np.radians(p0) + np.pi) % (2 * np.pi) - np.pi      # shortest way round
            E += k * dp ** 2
            ff = 2 * k * dp
            s1, s2 = np.dot(n1, n1), np.dot(n2, n2)
            gi = -l2 / s1 * n1
            gl = l2 / s2 * n2
            pp, qq = np.dot(b1, b2) / l2 ** 2, np.dot(b3, b2) / l2 ** 2
            gj = -(1 + pp) * gi + qq * gl
            gk = pp * gi - (1 + qq) * gl
            G[i] += ff * gi; G[j] += ff * gj; G[k_] += ff * gk; G[l_] += ff * gl
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
                    gtol=3e-4, maxiter=3000, solvent='chloroform'):
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
            E, G = energy_grad(sym, Y, charge, solvent=solvent)
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
    return X, float(energy_grad(sym, X, charge, solvent=solvent)[0])
