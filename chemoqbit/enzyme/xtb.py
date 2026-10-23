"""GFN2-xTB (tblite) optimiser with distance constraints, for reaction scans.

Implicit solvent: ALPB chloroform (eps 4.8), the closest named solvent to the
eps = 4 conventionally used for enzyme active-site clusters.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from tblite.interface import Calculator

BOHR = 0.529177210903
Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Zn': 30, 'Cu': 29}


def energy_grad(sym, X, charge=0, solvent='chloroform'):
    calc = Calculator('GFN2-xTB', np.array([Z[s] for s in sym]), X / BOHR, charge=charge)
    calc.set('verbosity', 0)
    if solvent:
        calc.add('alpb-solvation', solvent)
    r = calc.singlepoint()
    return r.get('energy'), r.get('gradient') / BOHR      # Eh, Eh/Angstrom


def charges(sym, X, charge=0, solvent='chloroform'):
    """GFN2-xTB atomic partial charges (e), e.g. for electrostatic embedding"""
    calc = Calculator('GFN2-xTB', np.array([Z[s] for s in sym]), X / BOHR, charge=charge)
    calc.set('verbosity', 0)
    if solvent:
        calc.add('alpb-solvation', solvent)
    return calc.singlepoint().get('charges')


def optimise(sym, X, charge=0, constraints=(), k=5.0, gtol=2e-4, maxiter=2000):
    """constraints: [(i, j, d0)] distances or [(i, j, l, m, x0)] for d_ij - d_lm = x0;
    harmonic, k in Eh/A^2.
    Returns (X_opt, E_xtb_without_restraint)."""
    shape = X.shape

    def f(x):
        Y = x.reshape(shape)
        E, G = energy_grad(sym, Y, charge)
        for c in constraints:
            if len(c) == 3:                      # distance i-j = d0
                i, j, d0 = c
                v = Y[i] - Y[j]; d = np.linalg.norm(v)
                E += k * (d - d0) ** 2
                g = 2 * k * (d - d0) * v / d
                G[i] += g; G[j] -= g
            else:                                # d(i-j) - d(l-m) = x0 (antisymmetric stretch)
                i, j, l, m, x0 = c
                v1 = Y[i] - Y[j]; d1 = np.linalg.norm(v1)
                v2 = Y[l] - Y[m]; d2 = np.linalg.norm(v2)
                x = d1 - d2
                E += k * (x - x0) ** 2
                f = 2 * k * (x - x0)
                G[i] += f * v1 / d1; G[j] -= f * v1 / d1
                G[l] -= f * v2 / d2; G[m] += f * v2 / d2
        return E, G.ravel()

    r = minimize(f, X.ravel(), jac=True, method='L-BFGS-B',
                 options=dict(maxiter=maxiter, gtol=gtol))
    Y = r.x.reshape(shape)
    return Y, energy_grad(sym, Y, charge)[0]
