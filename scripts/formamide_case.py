"""Case study 1: formamide -> formamidic acid (1,3 proton shift).

Minimal model of the lactam -> lactim tautomerism of 5-fluorouracil.
1. B3LYP/6-31G(d) geometries + harmonic frequencies (reactant, TS, product)
2. CCSD(T)/cc-pVDZ single points (classical reference)
3. CASCI active spaces (4e,4o)->8 qubits and (6e,6o)->12 qubits on CCSD
   natural orbitals; UCCSD-VQE on the float and fixed-point simulators
4. Eyring rate at 310 K with Wigner and Skodje-Truhlar tunnelling
Results -> results/formamide.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo

from chemoqbit.chem import build_problem, run_vqe, HARTREE_KCAL
from chemoqbit import kinetics as K

OUT = Path(__file__).resolve().parents[1] / 'results'
OUT.mkdir(exist_ok=True)
T = 310.15

# planar 4-centre guess: C at origin, H_b bridging N and O
C_ = np.array([0.0, 0.0, 0.0])
O_ = np.array([1.040, 0.728, 0.0])
N_ = np.array([-1.081, 0.757, 0.0])
Hb_ts = np.array([0.0, 1.55, 0.0])
Hc = np.array([0.0, -1.09, 0.0])
Hn = np.array([-1.96, 0.25, 0.0])


def geom(hb):
    return [('C', C_), ('O', O_), ('N', N_), ('H', hb), ('H', Hc), ('H', Hn)]


def toward(a, b, d):
    v = b - a
    return a + d * v / np.linalg.norm(v)


guesses = {
    'reactant': geom(toward(N_, Hb_ts, 1.01) + np.array([0, 0, 0.05])),
    'ts': geom(Hb_ts),
    'product': geom(toward(O_, Hb_ts, 0.97) + np.array([0, 0, 0.05])),
}


def dft_mf(atoms):
    mol = gto.M(atom=[(s, tuple(x)) for s, x in atoms], basis='6-31g(d)', verbose=0)
    mf = dft.RKS(mol); mf.xc = 'b3lyp'; mf.conv_tol = 1e-10
    return mf


def opt_and_freq(name, atoms):
    mf = dft_mf(atoms)
    kw = dict(transition=True, hessian=True) if name == 'ts' else {}
    mol = optimize(mf, maxsteps=200, **kw)
    mf = dft.RKS(mol); mf.xc = 'b3lyp'; mf.conv_tol = 1e-10; mf.kernel()
    hess = mf.Hessian().kernel()
    freq = thermo.harmonic_analysis(mol, hess)
    th = thermo.thermo(mf, freq['freq_au'], T, 101325)
    nu = freq['freq_wavenumber']
    imag = [abs(x.imag) for x in nu if abs(x.imag) > 1e-3]
    return dict(
        atoms=[(mol.atom_symbol(i), (mol.atom_coord(i, unit='Angstrom')).tolist())
               for i in range(mol.natm)],
        e_dft=float(mf.e_tot),
        zpe=float(th['ZPE'][0]),
        g_corr=float(th['G_tot'][0] - mf.e_tot),
        n_imag=len(imag), imag_cm=imag[0] if imag else None,
    )


def _old_main():
    data = {}
    for name, atoms in guesses.items():
        t0 = time.time()
        data[name] = opt_and_freq(name, atoms)
        print(f'{name}: E={data[name]["e_dft"]:.6f} n_imag={data[name]["n_imag"]} '
              f'imag={data[name]["imag_cm"]} ({time.time()-t0:.0f}s)', flush=True)
    for name, d in data.items():
        atom = '; '.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in d['atoms'])
        d['spaces'] = {}
        for (ne, no) in [(4, 4), (6, 6)]:
            t0 = time.time()
            p = build_problem(atom, 'cc-pvdz', no, ne, label=name, with_ccsd_t=(ne == 4))
            if ne == 4:
                d['e_hf'], d['e_ccsd_t'] = p.e_hf, p.e_ccsd_t
            r = run_vqe(p)
            d['spaces'][f'{ne}e{no}o'] = dict(
                n_qubits=p.n_qubits, e_casci=p.e_casci, e_vqe=r.energy,
                e_vqe_fixed=r.fixed_energy, err_mha=r.err_mha,
                fixed_err_mha=(r.fixed_energy - p.e_casci) * 1e3,
                n_params=r.n_params, n_evals=r.n_evals, gates=r.gate_counts,
                n_pauli=r.n_pauli_terms, n_groups=r.n_meas_groups)
            print(f'  {name} {ne}e{no}o: CASCI={p.e_casci:.8f} VQE={r.energy:.8f} '
                  f'err={r.err_mha:.2e} mHa fixed_err={(r.fixed_energy-p.e_casci)*1e3:.2e} '
                  f'params={r.n_params} ({time.time()-t0:.0f}s)', flush=True)

    # barriers (kcal/mol), electronic + ZPE / free-energy corrections from DFT
    R, TS, P = data['reactant'], data['ts'], data['product']
    def bar(key):
        return (TS[key] - R[key]) * HARTREE_KCAL, (P[key] - R[key]) * HARTREE_KCAL
    summary = {}
    for lab, getter in [('HF', lambda d: d['e_hf']), ('B3LYP', lambda d: d['e_dft']),
                        ('CCSD(T)', lambda d: d['e_ccsd_t']),
                        ('VQE 8q', lambda d: d['spaces']['4e4o']['e_vqe']),
                        ('VQE 12q', lambda d: d['spaces']['6e6o']['e_vqe']),
                        ('VQE 12q fixed-pt', lambda d: d['spaces']['6e6o']['e_vqe_fixed'])]:
        dEa = (getter(TS) - getter(R)) * HARTREE_KCAL
        dEr = (getter(P) - getter(R)) * HARTREE_KCAL
        zpe_a = (TS['zpe'] - R['zpe']) * HARTREE_KCAL
        zpe_r = (P['zpe'] - R['zpe']) * HARTREE_KCAL
        g_a = (TS['g_corr'] - R['g_corr']) * HARTREE_KCAL
        dGa = dEa + g_a
        nu = TS['imag_cm']
        k_cl = K.eyring(dGa, T)
        kw = K.wigner(nu, T)
        kst = K.skodje_truhlar(nu, dEa + zpe_a, dEr + zpe_r, T)
        summary[lab] = dict(dE_barrier=dEa, dE_rxn=dEr, dG_barrier=dGa,
                            k_classical=k_cl, kappa_wigner=kw, kappa_st=kst,
                            k_st=k_cl * kst)
        print(f'{lab:18s} Ea={dEa:7.2f} dE={dEr:6.2f} dG‡={dGa:7.2f} kcal/mol '
              f'k_cl={k_cl:.2e} κW={kw:.2f} κST={kst:.2f} s^-1')
    data['summary'] = summary
    data['imag_cm'] = TS['imag_cm']
    (OUT / 'formamide.json').write_text(json.dumps(data, indent=1, default=float))


def main():
    from chemoqbit.workflow import run_case
    run_case('formamide', guesses, OUT, log=lambda s: print(s, flush=True))


if __name__ == '__main__':
    main()
