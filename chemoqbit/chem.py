"""Active-space electronic structure (PySCF) -> qubit Hamiltonian -> VQE."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize
import scipy.sparse.linalg as spla

from pyscf import gto, scf, mcscf, ao2mo, cc

from .pauli import molecular_qubit_hamiltonian, to_sparse, qubitwise_commuting_groups
from .isa import encode, decode, peephole
from .circuit import Ansatz, FixedSim, gate_counts, uccsd_excitations, apply_pauli

HARTREE_KCAL = 627.509474
_CACHE = {}   # (atom, basis, charge, spin) -> (mf, natural orbitals, E_CCSD(T))


@dataclass
class ActiveSpaceProblem:
    label: str
    ecore: float
    h1: np.ndarray
    h2: np.ndarray
    nelec: int
    e_hf: float
    e_casci: float           # PySCF exact (FCI in active space) -- independent check
    e_ccsd_t: float | None = None
    qubit_op: dict = field(default=None, repr=False)

    @property
    def n_qubits(self):
        return 2 * self.h1.shape[0]


def build_problem(atom, basis, ncas, nelecas, label='', charge=0, spin=0,
                  with_ccsd_t=False, mo_coeff=None, natural_orbitals=True):
    key = (atom, basis, charge, spin)
    if key in _CACHE:
        mf, nat_mo, e_ccsd_t_cached = _CACHE[key]
    else:
        mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-11)
        nat_mo, e_ccsd_t_cached = None, None
    e_ccsd_t = None
    mo = mf.mo_coeff if mo_coeff is None else mo_coeff
    need_cc = (with_ccsd_t and e_ccsd_t_cached is None) or (natural_orbitals and nat_mo is None)
    if need_cc:
        mycc = cc.CCSD(mf).run(conv_tol=1e-9)
        if with_ccsd_t:
            e_ccsd_t_cached = mycc.e_tot + mycc.ccsd_t()
        if nat_mo is None:
            # active orbitals from the CCSD 1-RDM natural orbitals (stable along a path)
            dm = mycc.make_rdm1()
            occ, c = np.linalg.eigh(dm)
            nat_mo = mf.mo_coeff @ c[:, np.argsort(-occ)]
    _CACHE[key] = (mf, nat_mo, e_ccsd_t_cached)
    if with_ccsd_t:
        e_ccsd_t = e_ccsd_t_cached
    if natural_orbitals and mo_coeff is None:
        mo = nat_mo
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = 0
    e_casci = mc.kernel(mo)[0]
    h1, ecore = mc.get_h1eff(mo)
    h2 = ao2mo.restore(1, mc.get_h2eff(mo), ncas)
    prob = ActiveSpaceProblem(label, ecore, h1, h2, nelecas, mf.e_tot, e_casci, e_ccsd_t)
    prob.qubit_op = molecular_qubit_hamiltonian(ecore, h1, h2)
    return prob


@dataclass
class VQEResult:
    energy: float
    theta: np.ndarray
    n_params: int
    n_evals: int
    exact: float
    err_mha: float
    fixed_energy: float | None = None
    gate_counts: dict | None = None
    n_pauli_terms: int = 0
    n_meas_groups: int = 0
    fixed_norm_drift: float | None = None
    history: list | None = None
    operators: list | None = None
    n_words: int = 0


def _finish(out, prob, ans, H, fixed_point, frac_bits):
    # hardware path: compile -> peephole -> 32-bit ISA words -> decode
    gates = decode(encode(peephole(ans.gates(out.theta))))
    out.gate_counts = gate_counts(gates)
    out.n_words = len(gates) + 1
    out.n_pauli_terms = len(prob.qubit_op)
    out.n_meas_groups = len(qubitwise_commuting_groups(prob.qubit_op))
    if fixed_point:
        sim = FixedSim(prob.n_qubits, frac_bits=frac_bits, word_bits=frac_bits + 2,
                       coef_bits=frac_bits)
        psi = sim.run(ans.hf_state(), gates)
        nrm = np.vdot(psi, psi).real
        out.fixed_energy = float(np.real(np.vdot(psi, H @ psi)) / nrm)
        out.fixed_norm_drift = float(abs(nrm - 1))
    return out


def run_adapt(prob: ActiveSpaceProblem, gtol=1e-4, max_ops=200, fixed_point=True,
              frac_bits=16, log=None):
    """Fermionic ADAPT-VQE (Grimsley et al., Nat. Commun. 10, 3007, 2019).

    Pool: spin-conserving singles and doubles (generalised to all orbitals
    would also work; occ->vir keeps the pool small and particle-hole clean).
    Each cycle appends the operator with the largest energy gradient, then
    re-optimises all parameters with analytic adjoint gradients.
    """
    n = prob.n_qubits
    H = to_sparse(prob.qubit_op, n).real.tocsr()
    s, d = uccsd_excitations(n, prob.nelec)
    pool = s + d
    pool_ans = Ansatz.from_excitations(n, prob.nelec, pool)
    chosen, theta = [], np.zeros(0)
    ans = Ansatz.from_excitations(n, prob.nelec, [])
    idx = np.arange(2 ** n)
    history = []
    nfev = 0
    for it in range(max_ops):
        psi = ans.state(theta)
        hpsi = H @ psi
        grads = []
        for rot in pool_ans.rotations:
            g = sum(c * apply_pauli(psi, masks, idx) for masks, c, _ in rot)
            grads.append(abs(2 * np.real(np.vdot(hpsi, 1j * g))))
        grads = np.array(grads)
        gnorm = float(np.linalg.norm(grads))
        if gnorm < gtol:
            break
        k = int(np.argmax(grads))
        chosen.append(pool[k])
        ans = Ansatz.from_excitations(n, prob.nelec, chosen)
        res = minimize(lambda th: ans.energy_and_grad(th, H), np.append(theta, 0.0),
                       jac=True, method='BFGS', options=dict(gtol=1e-8, maxiter=5000))
        theta, nfev = res.x, nfev + res.nfev
        history.append(dict(op=pool[k], grad_norm=gnorm,
                            err_mha=(res.fun - prob.e_casci) * 1e3))
        if log:
            log(f'    ADAPT {it+1:3d} op={pool[k]} |g|={gnorm:.2e} '
                f'err={(res.fun-prob.e_casci)*1e3:.2e} mHa')
    E = ans.energy_and_grad(theta, H)[0] if len(theta) else prob.e_hf
    out = VQEResult(E, theta, len(theta), nfev, prob.e_casci, (E - prob.e_casci) * 1e3)
    out.history = history
    out.operators = chosen
    return _finish(out, prob, ans, H, fixed_point, frac_bits)


def run_vqe(prob: ActiveSpaceProblem, theta0=None, fixed_point=True, frac_bits=16):
    """Fixed UCCSD ansatz (all singles + doubles), one Trotter step."""
    n = prob.n_qubits
    H = to_sparse(prob.qubit_op, n).real.tocsr()   # H is real in this basis
    ans = Ansatz.uccsd(n, prob.nelec)
    th0 = np.zeros(ans.n_params) if theta0 is None else np.asarray(theta0)
    res = minimize(lambda th: ans.energy_and_grad(th, H), th0, jac=True,
                   method='BFGS', options=dict(maxiter=5000, gtol=1e-9))
    out = VQEResult(res.fun, res.x, ans.n_params, res.nfev, prob.e_casci,
                    (res.fun - prob.e_casci) * 1e3)
    return _finish(out, prob, ans, H, fixed_point, frac_bits)
