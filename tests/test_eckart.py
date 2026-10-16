import numpy as np
from chemoqbit.eckart import transmission, eckart_kappa, HBAR, AMU, KCAL
from chemoqbit.kinetics import wigner


def sech2_exact(E, V0, a, m):
    """Landau-Lifshitz: T for V0 sech^2(x/a)."""
    k = np.sqrt(2 * m * E) / HBAR
    g = 8 * m * V0 * a ** 2 / HBAR ** 2
    num = np.sinh(np.pi * k * a) ** 2
    den = num + (np.cosh(np.pi / 2 * np.sqrt(g - 1)) ** 2 if g > 1
                 else np.cos(np.pi / 2 * np.sqrt(1 - g)) ** 2)
    return num / den


def test_transfer_matrix_matches_exact_sech2_barrier():
    m, V0, a = AMU, 20 * KCAL, 0.25e-10
    x = np.linspace(-25, 25, 3000) * a
    V = V0 / np.cosh(x / a) ** 2
    for frac in (0.5, 0.8, 0.95, 1.05, 1.3):
        E = frac * V0
        T_num, T_ex = transmission(E, x, V, m), sech2_exact(E, V0, a, m)
        assert abs(np.log(T_num / T_ex)) < 0.03, (frac, T_num, T_ex)


def test_eckart_reduces_to_wigner_for_flat_barriers():
    # small curvature, high T: kappa -> 1 + u^2/24
    k, _ = eckart_kappa(150.0, 15.0, 0.0, T=600)
    assert abs(k - wigner(150.0, 600)) < 2e-3


def test_eckart_kappa_reasonable_for_proton_shift():
    k, _ = eckart_kappa(1913.0, 46.0, 12.0, T=300)
    assert 10 < k < 1e5
