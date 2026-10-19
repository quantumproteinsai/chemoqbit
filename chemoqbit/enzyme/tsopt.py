"""Saddle-point and minimum optimisation with geomeTRIC on any energy/gradient
function (here GFN2-xTB), finite-difference Hessians, and harmonic analysis
with isotopic substitution for kinetic isotope effects.
"""
from __future__ import annotations

import os, tempfile
import numpy as np
import geometric
import geometric.molecule
from geometric.engine import Engine

BOHR = 0.529177210903
AMU = {'H': 1.00782503, 'D': 2.01410178, 'T': 3.01604928, 'C': 12.0, 'N': 14.00307401,
       'O': 15.99491462, 'F': 18.99840316, 'S': 31.97207117}
HARTREE_J = 4.3597447222071e-18
AMU_KG = 1.66053906660e-27
C_CM = 2.99792458e10
KB = 1.380649e-23
H_PL = 6.62607015e-34


class FnEngine(Engine):
    def __init__(self, molecule, fn):
        super().__init__(molecule)
        self.fn = fn

    def calc_new(self, coords, dirname):
        X = coords.reshape(-1, 3) * BOHR          # bohr -> A
        E, G = self.fn(X)                          # Eh, Eh/A
        return {'energy': E, 'gradient': (G * BOHR).ravel()}   # Eh/bohr


def run(sym, X, fn, transition=False, hessian=None, maxiter=300, constraints=None):
    M = geometric.molecule.Molecule()
    M.elem = list(sym)
    M.xyzs = [np.array(X, float)]
    eng = FnEngine(M, fn)
    with tempfile.TemporaryDirectory() as td:
        kw = dict(customengine=eng, input=os.path.join(td, 'job'), maxiter=maxiter,
                  transition=transition, convergence_set='GAU')
        if hessian is not None:
            hf = os.path.join(td, 'hess.txt')
            np.savetxt(hf, hessian * BOHR ** 2)     # Eh/A^2 -> Eh/bohr^2
            kw['hessian'] = 'file:' + hf
        if constraints:
            cf = os.path.join(td, 'cons.txt')
            open(cf, 'w').write(constraints)
            kw['constraints'] = cf
        res = geometric.optimize.run_optimizer(**kw)
    Xf = res.xyzs[-1]
    return np.array(Xf), fn(np.array(Xf))[0]


def _free_coords(n_atoms, free):
    free = range(n_atoms) if free is None else free
    return np.array([3 * a + c for a in free for c in range(3)])


def fd_hessian(X, grad_fn, h=0.005, free=None):
    """central differences of the gradient (Eh/A) -> Hessian (Eh/A^2), symmetrised.
    With `free` (atom indices), only the block over those atoms' coordinates."""
    idx = _free_coords(len(X), free)
    Hs = np.zeros((idx.size, idx.size))
    x0 = X.ravel()
    for r, k in enumerate(idx):
        xp, xm = x0.copy(), x0.copy()
        xp[k] += h; xm[k] -= h
        Hs[r] = (grad_fn(xp.reshape(-1, 3)).ravel()[idx] - grad_fn(xm.reshape(-1, 3)).ravel()[idx]) / (2 * h)
    return 0.5 * (Hs + Hs.T)


def _project_tr(X, m):
    """projector removing translations and rotations (mass-weighted)"""
    n = len(X)
    sm = np.sqrt(np.repeat(m, 3))
    com = (X * m[:, None]).sum(0) / m.sum()
    Y = X - com
    D = np.zeros((6, 3 * n))
    for i in range(3):
        D[i, i::3] = 1.0
    for a in range(n):
        x, y, z = Y[a]
        D[3, 3 * a:3 * a + 3] = [0, -z, y]
        D[4, 3 * a:3 * a + 3] = [z, 0, -x]
        D[5, 3 * a:3 * a + 3] = [-y, x, 0]
    D = D * sm[None, :]
    Q, _ = np.linalg.qr(D.T)
    Q = Q[:, :np.linalg.matrix_rank(D.T, tol=1e-8)]
    return np.eye(3 * n) - Q @ Q.T


def frequencies(X, H, masses, free=None):
    """harmonic wavenumbers (cm-1, imaginary as negative), sorted.
    With `free` (atom indices), H is the free-atom block from fd_hessian; the
    frozen atoms break translational/rotational invariance, so nothing is
    projected out and every mode is kept."""
    m = np.asarray(masses)
    if free is not None:
        mw = 1 / np.sqrt(np.repeat(m[list(free)], 3))
        w = np.linalg.eigvalsh(H * mw[:, None] * mw[None, :])
    else:
        mw = 1 / np.sqrt(np.repeat(m, 3))
        Hm = H * mw[:, None] * mw[None, :]                # Eh/(A^2 amu)
        P = _project_tr(X, m)
        w = np.linalg.eigvalsh(P @ Hm @ P)
    conv = HARTREE_J / (1e-20 * AMU_KG)                   # -> s^-2
    nu = np.sign(w) * np.sqrt(np.abs(w) * conv) / (2 * np.pi * C_CM)
    if free is None:
        nu = nu[np.argsort(np.abs(nu))][6:]               # drop the 6 ~zero modes
    return np.sort(nu)


def moments(X, masses):
    m = np.asarray(masses)
    Y = X - (X * m[:, None]).sum(0) / m.sum()
    I = np.einsum('i,ij,ik->jk', m, Y, Y)
    I = np.trace(I) * np.eye(3) - I
    return np.linalg.eigvalsh(I)                          # amu A^2


def ln_q_vib_rot(X, nu_real, masses, T, rot=True):
    """ln(Q_rot * Q_vib) with Q_vib measured from the bottom of the well (ZPE included).
    rot=False for a cluster with frozen atoms, which has no free rotation."""
    x = H_PL * C_CM * np.asarray(nu_real) / (KB * T)
    lnqv = np.sum(-x / 2 - np.log1p(-np.exp(-x)))
    lnqr = 0.5 * np.sum(np.log(moments(X, masses))) if rot else 0.0   # constants cancel in ratios
    return lnqv + lnqr


def zpe(nu_real):
    return 0.5 * H_PL * C_CM * np.sum(nu_real) / HARTREE_J   # Eh


def _cart_projector(X):
    """remove rigid translations/rotations in plain Cartesian space"""
    n = len(X)
    Y = X - X.mean(0)
    D = np.zeros((6, 3 * n))
    for i in range(3):
        D[i, i::3] = 1.0
    for a in range(n):
        x, y, z = Y[a]
        D[3, 3 * a:3 * a + 3] = [0, -z, y]
        D[4, 3 * a:3 * a + 3] = [z, 0, -x]
        D[5, 3 * a:3 * a + 3] = [-y, x, 0]
    Q, _ = np.linalg.qr(D.T)
    return np.eye(3 * n) - Q @ Q.T


def prfo_follow(X, fn, guide, grad_fn=None, trust=0.08, gtol=3e-4, maxiter=60, max_pos=8,
                neg=1e-3, frozen=(), log=print):
    """Partitioned-RFO saddle search following the Hessian mode that overlaps
    best with `guide` (a 3N displacement, e.g. hydride moving donor->acceptor),
    with full finite-difference Hessians each step. Units: A, Eh.

    Gives up after `max_pos` consecutive steps in which the followed mode's
    curvature is above -`neg` (Eh/A^2): the start is then not in a saddle region
    and the search is only sliding downhill. A C-H-C hydride transfer mode is
    about -0.2 Eh/A^2; soft collective cluster modes are ~1e-4.

    Atoms in `frozen` are held fixed: the search runs in the free atoms'
    coordinates, with no translation/rotation projection (the frozen atoms
    already remove those motions)."""
    grad_fn = grad_fn or (lambda Y: fn(Y)[1])
    X = np.array(X, float)
    free = [a for a in range(len(X)) if a not in set(frozen)] if len(frozen) else None
    idx = _free_coords(len(X), free)
    prev = guide.ravel()[idx] / np.linalg.norm(guide.ravel()[idx])
    n_pos = 0
    for it in range(maxiter):
        E, G = fn(X)
        P = _cart_projector(X) if free is None else np.eye(idx.size)
        g = P @ G.ravel()[idx]
        Hs = P @ fd_hessian(X, grad_fn, free=free) @ P
        b, V = np.linalg.eigh(Hs)
        keep = np.abs(b) > 1e-6
        b, V = b[keep], V[:, keep]
        ov = np.abs(V.T @ prev)
        k = int(np.argmax(ov))
        gq = V.T @ g
        # maximise along k
        lp = 0.5 * b[k] + 0.5 * np.sqrt(b[k] ** 2 + 4 * gq[k] ** 2)
        step = np.zeros_like(gq)
        step[k] = -gq[k] / (b[k] - lp)
        # minimise along the rest: lowest root of l = sum g_i^2/(l - b_i)
        oth = np.array([i for i in range(len(b)) if i != k])
        lo = min(b[oth].min(), 0.0) - 1e-3
        ln = lo - 1.0
        for _ in range(200):          # bisection on f(l) = l - sum g^2/(l-b) for l < min b
            f = lambda l: l - np.sum(gq[oth] ** 2 / (l - b[oth]))
            a_, c_ = lo - 10.0, lo
            for _ in range(100):
                m_ = 0.5 * (a_ + c_)
                if f(m_) > 0: c_ = m_
                else: a_ = m_
            ln = 0.5 * (a_ + c_)
            break
        step[oth] = -gq[oth] / (b[oth] - ln)
        dx = np.zeros(X.size)
        dx[idx] = V @ step
        nrm = np.linalg.norm(dx)
        if nrm > trust:
            dx *= trust / nrm
        gr = np.sqrt(np.mean(g ** 2))
        log(f'  PRFO {it:2d} E={E:.6f} rmsG={gr:.2e} mode={k} eig={b[k]:+.4f} overlap={ov[k]:.2f} step={min(nrm, trust):.3f}')
        prev = V[:, k] * np.sign(V[:, k] @ prev)
        if gr < gtol and b[k] < -neg:
            return X, E, dict(iterations=it, followed_eig=float(b[k]), overlap=float(ov[k]))
        n_pos = n_pos + 1 if b[k] > -neg else 0
        if n_pos >= max_pos:
            raise RuntimeError(f'PRFO: followed mode has had curvature above {-neg:g} for {n_pos} '
                               f'steps (eig={b[k]:+.2e}); the guess is not near a saddle point')
        X = X + dx.reshape(-1, 3)
    raise RuntimeError('PRFO did not converge')
