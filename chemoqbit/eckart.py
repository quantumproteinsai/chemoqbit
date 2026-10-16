"""Tunnelling through a fitted asymmetric Eckart barrier.

The barrier V(x) = A y/(1+y) + B y/(1+y)^2, y = exp(x/L), is fitted to the
forward barrier V1, the reaction energy dE (= A) and the curvature at the top
(the TS imaginary frequency). Transmission P(E) is computed numerically with a
transfer-matrix solver -- no closed form needed -- and validated against the
exact analytic result for the symmetric sech^2 barrier (tests).

    kappa = beta * exp(beta*V1) * integral_{E0}^{inf} P(E) exp(-beta*E) dE

with E measured from the reactant asymptote and E0 = max(0, dE).
"""
from __future__ import annotations

import numpy as np

HBAR = 1.054571817e-34
KB = 1.380649e-23
C_CM = 2.99792458e10
AMU = 1.66053906660e-27
KCAL = 4184 / 6.02214076e23


def transmission(E, x, V, mass):
    """|t|^2 * k_out/k_in for a potential sampled on a uniform grid
    (piecewise constant), via 2x2 transfer matrices. E, V in J; x in m."""
    k = np.sqrt(2 * mass * (E - V + 0j)) / HBAR
    k = np.where(np.abs(k) < 1e-30, 1e-30, k)
    # boundaries between cells j and j+1 at xb = midpoints
    xb = 0.5 * (x[1:] + x[:-1])
    M = np.eye(2, dtype=complex)
    logscale = 0.0
    for j in range(len(xb)):
        k1, k2, b = k[j], k[j + 1], xb[j]
        r = k1 / k2
        # amplitudes (A,B) of e^{ikx}, e^{-ikx}: continuity of psi and psi'
        m = 0.5 * np.array([
            [(1 + r) * np.exp(1j * (k1 - k2) * b), (1 - r) * np.exp(-1j * (k1 + k2) * b)],
            [(1 - r) * np.exp(1j * (k1 + k2) * b), (1 + r) * np.exp(-1j * (k1 - k2) * b)]])
        M = m @ M
        sc = np.abs(M).max()
        if sc > 1e100:             # rescale, tracked exactly in logscale
            M /= sc
            logscale += np.log(sc)
    kin, kout = k[0], k[-1]
    if kout.imag > 1e-12 * abs(kout):   # closed channel on the right
        return 0.0
    # (t, 0) = M (1, r)  =>  t = det(M) / M11 ; det(M) = prod(k_j/k_j+1) = kin/kout
    # T = |t|^2 kout/kin = (kin/kout) / |M11|^2
    logT = np.log(abs(kin / kout)) - 2 * (np.log(abs(M[1, 1])) + logscale)
    return float(np.exp(min(logT, 0.0)))


def eckart_potential(V1, dE, nu_cm, mass=AMU, n=1500, span=22.0):
    """Return grid x (m), V(x) (J), mass. V1, dE in J; curvature fitted to nu."""
    A = dE
    B = (np.sqrt(V1) + np.sqrt(V1 - dE)) ** 2

    def V_of(u):   # u = x/L
        y = np.exp(u)
        return A * y / (1 + y) + B * y / (1 + y) ** 2

    # locate the top in u, then the curvature d2V/du2 there
    u = np.linspace(-span, span, 200001)
    vu = V_of(u)
    i = int(np.argmax(vu))
    h = 1e-4
    d2 = (V_of(u[i] + h) - 2 * V_of(u[i]) + V_of(u[i] - h)) / h ** 2
    omega = 2 * np.pi * C_CM * nu_cm
    L = np.sqrt(-d2 / (mass * omega ** 2))     # V''(x) = d2/L^2 = -m w^2
    x = np.linspace(-span, span, n) * L
    return x, V_of(x / L), mass, float(vu[i])


def eckart_kappa(nu_cm, V1_kcal, dE_kcal, T=310.15, n_energy=500, n_grid=1500):
    V1, dE = V1_kcal * KCAL, dE_kcal * KCAL
    x, V, m, vmax = eckart_potential(V1, dE, nu_cm, n=n_grid)
    assert abs(vmax - V1) < 1e-6 * V1, 'Eckart fit did not reproduce the barrier'
    kT = KB * T
    E0 = max(0.0, dE)
    Emax = V1 + 40 * kT
    # dense near and below the top, where the integrand peaks
    E = np.linspace(E0 + 1e-3 * kT, Emax, n_energy)
    P = np.array([transmission(e, x, V, m) for e in E])
    integrand = P * np.exp(-(E - V1) / kT)
    kappa = np.trapezoid(integrand, E) / kT
    # Boltzmann-weighted mean tunnelling energy below the top (diagnostic)
    return float(kappa), dict(E=E, P=P, integrand=integrand)
