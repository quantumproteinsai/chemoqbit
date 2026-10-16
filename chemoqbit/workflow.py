"""Reusable reactant / TS / product workflow (used by every case study)."""
from __future__ import annotations

import json, time
from pathlib import Path
import numpy as np

from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo

from .chem import build_problem, run_vqe, run_adapt, HARTREE_KCAL
from . import kinetics as K
from .summarise import summarise


def dft_mf(atoms, basis='6-31g(d)', xc='b3lyp'):
    mol = gto.M(atom=[(s, tuple(x)) for s, x in atoms], basis=basis, verbose=0)
    mf = dft.RKS(mol); mf.xc = xc; mf.conv_tol = 1e-10
    return mf


def opt_and_freq(name, atoms, T=310.15):
    mf = dft_mf(atoms)
    kw = dict(transition=True, hessian=True) if name == 'ts' else {}
    mol = optimize(mf, maxsteps=300, **kw)
    mf = dft.RKS(mol); mf.xc = 'b3lyp'; mf.conv_tol = 1e-10; mf.kernel()
    hess = mf.Hessian().kernel()
    freq = thermo.harmonic_analysis(mol, hess)
    th = thermo.thermo(mf, freq['freq_au'], T, 101325)
    nu = freq['freq_wavenumber']
    imag = [abs(x.imag) for x in nu if abs(x.imag) > 1e-3]
    return dict(
        atoms=[(mol.atom_symbol(i), mol.atom_coord(i, unit='Angstrom').tolist())
               for i in range(mol.natm)],
        e_dft=float(mf.e_tot), zpe=float(th['ZPE'][0]),
        g_corr=float(th['G_tot'][0] - mf.e_tot),
        n_imag=len(imag), imag_cm=imag[0] if imag else None)


def run_case(tag, guesses, out_dir, spaces=((4, 4), (6, 6)), sp_basis='cc-pvdz',
             T=310.15, log=print):
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)
    geom_file = out_dir / f'{tag}_geom.json'
    data = json.loads(geom_file.read_text()) if geom_file.exists() else {}
    for name, atoms in guesses.items():
        if name in data:
            log(f'{name}: cached geometry, imag={data[name]["imag_cm"]}')
            continue
        t0 = time.time()
        data[name] = opt_and_freq(name, atoms, T)
        log(f'{name}: E={data[name]["e_dft"]:.6f} n_imag={data[name]["n_imag"]} '
            f'imag={data[name]["imag_cm"]} ({time.time()-t0:.0f}s)')
        geom_file.write_text(json.dumps(data, indent=1, default=float))
    for name, d in data.items():
        atom = '; '.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in d['atoms'])
        d.setdefault('spaces', {})
        for ne, no in spaces:
            if all(f'{ne}e{no}o_{m}' in d['spaces'] for m in ('uccsd', 'adapt')):
                log(f'  {name} {ne}e{no}o: checkpointed')
                continue
            t0 = time.time()
            p = build_problem(atom, sp_basis, no, ne, label=name, with_ccsd_t=True)
            d['e_hf'], d['e_ccsd_t'] = p.e_hf, p.e_ccsd_t
            for method, runner in (('uccsd', run_vqe), ('adapt', lambda q: run_adapt(q, gtol=5e-4, max_ops=100, log=log))):
                r = runner(p)
                d['spaces'][f'{ne}e{no}o_{method}'] = dict(
                    n_qubits=p.n_qubits, e_casci=p.e_casci, e_vqe=r.energy,
                    e_vqe_fixed=r.fixed_energy, err_mha=r.err_mha,
                    fixed_err_mha=(r.fixed_energy - p.e_casci) * 1e3,
                    fixed_norm_drift=r.fixed_norm_drift,
                    n_params=r.n_params, n_evals=r.n_evals, gates=r.gate_counts,
                    n_pauli=r.n_pauli_terms, n_groups=r.n_meas_groups,
                    operators=[list(o) for o in (r.operators or [])],
                    theta=[float(x) for x in r.theta], nelec=p.nelec,
                    history=r.history)
                log(f'  {name} {ne}e{no}o {method:5s}: CASCI={p.e_casci:.8f} VQE={r.energy:.8f} '
                    f'err={r.err_mha:.2e} mHa fixed_err={(r.fixed_energy-p.e_casci)*1e3:.2e} '
                    f'params={r.n_params} CNOT={r.gate_counts.get("CNOT", 0)} '
                    f'({time.time()-t0:.0f}s)')
                t0 = time.time()
                geom_file.write_text(json.dumps(data, indent=1, default=float))  # checkpoint
    summarise(data, T)
    (out_dir / f'{tag}.json').write_text(json.dumps(data, indent=1, default=float))
    return data
