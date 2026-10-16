"""Transition-state theory with tunnelling corrections.

Wigner (1932) is only valid for u = h nu / kT << 2 pi; proton transfers with
nu ~ 1500-2000i cm-1 violate this at 310 K, so Skodje-Truhlar (1981) is
reported as the production value and Wigner kept for comparison.
"""
import numpy as np

KB = 1.380649e-23        # J/K
H = 6.62607015e-34       # J s
C = 2.99792458e10        # cm/s
R_KCAL = 1.987204e-3     # kcal/mol/K
KCAL_J = 4184 / 6.02214076e23   # J per molecule per kcal/mol


def eyring(dG_kcal, T=310.15):
    return KB * T / H * np.exp(-dG_kcal / (R_KCAL * T))


def wigner(nu_cm, T=310.15):
    u = H * C * nu_cm / (KB * T)
    return 1 + u ** 2 / 24


def skodje_truhlar(nu_cm, V_kcal, dE_kcal, T=310.15):
    """nu_cm: |imaginary freq|; V_kcal: ZPE-corrected forward barrier;
    dE_kcal: ZPE-corrected reaction energy (product - reactant)."""
    alpha = 2 * np.pi / (H * C * nu_cm)          # 1/J
    beta = 1 / (KB * T)
    V = V_kcal * KCAL_J
    V0 = 0.0 if dE_kcal <= 0 else dE_kcal * KCAL_J
    if beta < alpha:
        x = beta * np.pi / alpha
        return x / np.sin(x) - beta / (alpha - beta) * np.exp((beta - alpha) * (V - V0))
    return beta / (beta - alpha) * (np.exp((beta - alpha) * (V - V0)) - 1)
