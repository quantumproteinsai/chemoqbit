import numpy as np
import scipy.sparse.linalg as spla

from chemoqbit.chem import build_problem, run_vqe
from chemoqbit.circuit import Ansatz, FloatSim, FixedSim
from chemoqbit.pauli import to_sparse


def test_h2_matches_fci():
    p = build_problem('H 0 0 0; H 0 0 0.735', 'sto-3g', 2, 2, natural_orbitals=False)
    r = run_vqe(p)
    # Known H2/STO-3G FCI at 0.735 A
    assert abs(p.e_casci - (-1.137306)) < 2e-6
    assert abs(r.energy - p.e_casci) < 1e-8
    assert r.n_pauli_terms == 15


def test_qubit_hamiltonian_spectrum_matches_casci():
    p = build_problem('Li 0 0 0; H 0 0 1.595', 'sto-3g', 3, 2)
    H = to_sparse(p.qubit_op, p.n_qubits).real
    e0 = spla.eigsh(H, k=1, which='SA')[0][0]
    assert abs(e0 - p.e_casci) < 1e-8


def test_gate_compilation_reproduces_exact_state():
    ans = Ansatz.uccsd(6, 2)
    rng = np.random.default_rng(1)
    th = rng.normal(scale=0.3, size=ans.n_params)
    psi_exact = ans.state(th)
    psi_gates = FloatSim(6).run(ans.hf_state(), ans.gates(th))
    assert abs(abs(np.vdot(psi_exact, psi_gates)) - 1) < 1e-12
    # fixed point is close but not exact
    psi_fx = FixedSim(6, frac_bits=16, word_bits=18, coef_bits=16).run(ans.hf_state(), ans.gates(th))
    f = abs(np.vdot(psi_exact, psi_fx)) ** 2
    assert 1 - f < 1e-6


def test_lih_vqe():
    p = build_problem('Li 0 0 0; H 0 0 1.595', 'sto-3g', 3, 2)
    r = run_vqe(p)
    assert abs(r.energy - p.e_casci) < 1e-6


def test_adjoint_gradient_matches_finite_difference():
    from chemoqbit.chem import to_sparse
    p = build_problem('Li 0 0 0; H 0 0 1.595', 'sto-3g', 3, 2)
    H = to_sparse(p.qubit_op, p.n_qubits).real.tocsr()
    ans = Ansatz.uccsd(p.n_qubits, p.nelec)
    th = np.random.default_rng(3).normal(scale=0.2, size=ans.n_params)
    E, g = ans.energy_and_grad(th, H)
    h = 1e-6
    fd = [(ans.energy_and_grad(th + h*np.eye(len(th))[k], H)[0]
           - ans.energy_and_grad(th - h*np.eye(len(th))[k], H)[0]) / (2*h) for k in range(len(th))]
    assert np.allclose(g, fd, atol=1e-7)


def test_adapt_reaches_fci_with_fewer_params():
    from chemoqbit.chem import run_adapt
    p = build_problem('Li 0 0 0; H 0 0 1.595', 'sto-3g', 3, 2)
    r = run_adapt(p, gtol=1e-6)
    u = run_vqe(p, fixed_point=False)
    assert abs(r.energy - p.e_casci) < 1e-6
    assert r.n_params <= u.n_params


def test_isa_roundtrip_and_peephole():
    from chemoqbit.isa import encode, decode, peephole
    ans = Ansatz.uccsd(8, 4)
    th = np.random.default_rng(5).normal(scale=0.2, size=ans.n_params)
    g = ans.gates(th)
    ref = FloatSim(8).run(ans.hf_state(), g)
    gp = peephole(g)
    assert len(gp) < 0.85 * len(g)
    psi_p = FloatSim(8).run(ans.hf_state(), gp)
    assert abs(abs(np.vdot(ref, psi_p)) - 1) < 1e-12
    psi_q = FloatSim(8).run(ans.hf_state(), decode(encode(gp)))
    assert 1 - abs(np.vdot(ref, psi_q)) ** 2 < 1e-6
