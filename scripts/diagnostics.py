"""Side diagnostics quoted in the paper, saved so every number is reproducible.

1. AVAS active spaces on the reacting atoms (CASCI barriers)
2. ADAPT convergence plateau on 8 qubits (formamide reactant)
3. Peephole savings on full UCCSD circuits
4. Skodje-Truhlar vs Eckart tunnelling factors, incl. Truong (1997) inputs
-> results/diagnostics.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from chemoqbit.chem import build_problem, run_adapt, HARTREE_KCAL as H
from chemoqbit.circuit import Ansatz, gate_counts
from chemoqbit.isa import peephole
from chemoqbit import kinetics as K
from chemoqbit.eckart import eckart_kappa

OUT = ROOT / 'results' / 'diagnostics.json'
diag = json.loads(OUT.read_text()) if OUT.exists() else {}
save = lambda: OUT.write_text(json.dumps(diag, indent=1, default=float))
atoms = lambda d: '; '.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in d['atoms'])

# 1 -- AVAS
if 'avas' not in diag:
    diag['avas'] = []
    for tag, ix in [('formamide', dict(C=0, O=1, N=2, H=3)), ('fivefu', dict(C=6, O=7, N=8, H=11))]:
        d = json.loads((ROOT / f'results/{tag}.json').read_text())
        for name, labels in [
                ('N 2p + O 2p + H 1s', [f"{ix['N']} N 2p", f"{ix['O']} O 2p", f"{ix['H']} H 1s"]),
                ('N 2p + O 2p + H 1s + C 2p', [f"{ix['N']} N 2p", f"{ix['O']} O 2p",
                                               f"{ix['H']} H 1s", f"{ix['C']} C 2p"])]:
            E, act = {}, {}
            for g in ('reactant', 'ts'):
                mol = gto.M(atom=atoms(d[g]), basis='cc-pvdz', verbose=0)
                mf = scf.RHF(mol).run(conv_tol=1e-10)
                ncas, nelec, mo = avas.avas(mf, labels, canonicalize=True, verbose=0)
                mc = mcscf.CASCI(mf, ncas, nelec); mc.verbose = 0
                E[g] = mc.kernel(mo)[0]; act[g] = (int(ncas), int(nelec))
            row = dict(case=tag, labels=name, active_orbitals=act['ts'][0],
                       active_electrons=act['ts'][1], qubits=2 * act['ts'][0],
                       casci_barrier=(E['ts'] - E['reactant']) * H,
                       ccsd_t_barrier=(d['ts']['e_ccsd_t'] - d['reactant']['e_ccsd_t']) * H)
            diag['avas'].append(row); save()
            print('AVAS', row, flush=True)

# 2 -- ADAPT plateau
if 'adapt_plateau' not in diag:
    d = json.loads((ROOT / 'results/formamide.json').read_text())['reactant']
    p = build_problem(atoms(d), 'cc-pvdz', 4, 4)
    diag['adapt_plateau'] = {}
    for gtol in (1e-4, 1e-6):
        r = run_adapt(p, gtol=gtol, max_ops=80, fixed_point=False)
        diag['adapt_plateau'][str(gtol)] = dict(
            n_ops=r.n_params, err_mha=r.err_mha,
            trace_err_mha=[h['err_mha'] for h in r.history])
        save(); print('ADAPT', gtol, r.n_params, r.err_mha, flush=True)

# 3 -- peephole
if 'peephole' not in diag:
    diag['peephole'] = []
    for n, e in [(8, 4), (12, 6)]:
        a = Ansatz.uccsd(n, e)
        th = np.random.default_rng(0).normal(scale=0.1, size=a.n_params)
        g = a.gates(th); q = peephole(g)
        c0, c1 = gate_counts(g), gate_counts(q)
        diag['peephole'].append(dict(qubits=n, gates_before=len(g), gates_after=len(q),
                                     cnot_before=c0['CNOT'], cnot_after=c1['CNOT']))
    save(); print('peephole', diag['peephole'], flush=True)

# 4 -- tunnelling models
if 'tunnelling' not in diag:
    diag['tunnelling'] = []
    for tag in ('formamide', 'fivefu'):
        d = json.loads((ROOT / f'results/{tag}.json').read_text())
        R, TS, P = d['reactant'], d['ts'], d['product']
        za, zr = (TS['zpe'] - R['zpe']) * H, (P['zpe'] - R['zpe']) * H
        s = d['summary']['CCSD(T)']
        V1, dE = s['dE_barrier'] + za, s['dE_rxn'] + zr
        diag['tunnelling'].append(dict(
            case=tag, nu_cm=TS['imag_cm'], V1_zpe=V1, dE_zpe=dE, T=310.15,
            wigner=K.wigner(TS['imag_cm']), eckart=eckart_kappa(TS['imag_cm'], V1, dE)[0],
            skodje_truhlar=K.skodje_truhlar(TS['imag_cm'], V1, dE)))
    # Truong (1997) BH&H-LYP inputs: 51.8 classical barrier, 2117i, dE 12.6, 300 K.
    # ZPE corrections there are not quoted in the text we read; ours (-3.4, +0.6) are used.
    kE = eckart_kappa(2117, 51.8 - 3.4, 12.6 + 0.6, 300)[0]
    diag['tunnelling'].append(dict(case='Truong 1997 inputs (approx. ZPE)', nu_cm=2117, T=300,
                                   eckart=kE, wigner=K.wigner(2117, 300),
                                   zct0_reported_ratio_to_wigner=374,
                                   zct0_implied=374 * K.wigner(2117, 300)))
    save(); print('tunnelling', diag['tunnelling'], flush=True)
