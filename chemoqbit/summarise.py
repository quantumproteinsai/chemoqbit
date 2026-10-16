"""Barriers, rates and tunnelling factors from a saved case-study JSON.

Kept separate from the expensive workflow so kinetics choices can be
revisited without recomputing electronic structure.
"""
from __future__ import annotations

import json
from pathlib import Path

from .chem import HARTREE_KCAL as H
from . import kinetics as K
from .eckart import eckart_kappa


def levels(data):
    out = [('HF', lambda d: d['e_hf']), ('B3LYP', lambda d: d['e_dft']),
           ('CCSD(T)', lambda d: d['e_ccsd_t'])]
    keys = sorted({k.rsplit('_', 1)[0] for k in data['ts']['spaces']},
                  key=lambda k: int(k.split('e')[0]))
    for sp in keys:
        nq = data['ts']['spaces'][f'{sp}_adapt']['n_qubits']
        out.append((f'CASCI {nq}q (exact)', lambda d, k=f'{sp}_adapt': d['spaces'][k]['e_casci']))
        for m in ('uccsd', 'adapt'):
            k = f'{sp}_{m}'
            out.append((f'{m.upper()} {nq}q', lambda d, k=k: d['spaces'][k]['e_vqe']))
            out.append((f'{m.upper()} {nq}q fixed-pt', lambda d, k=k: d['spaces'][k]['e_vqe_fixed']))
    comp = data.get('composite') or {}
    for sp in keys:
        if all(f'{g}/{sp}' in comp for g in ('reactant', 'ts', 'product')):
            nq = data['ts']['spaces'][f'{sp}_adapt']['n_qubits']
            # E_CCSD(T)[full] + (E_x[active] - E_CCSD(T)[active])
            def corr(x, sp=sp):
                return lambda d, sp=sp, x=x: (d['e_ccsd_t'] + x(d, sp)
                                              - d['_comp'][sp]['e_ccsd_t_active'])
            out.append((f'CCSD(T)+static {nq}q (exact)',
                        corr(lambda d, sp: d['spaces'][f'{sp}_adapt']['e_casci'])))
            out.append((f'CCSD(T)+static {nq}q (ADAPT)',
                        corr(lambda d, sp: d['spaces'][f'{sp}_adapt']['e_vqe'])))
            out.append((f'CCSD(T)+static {nq}q (ADAPT fixed-pt)',
                        corr(lambda d, sp: d['spaces'][f'{sp}_adapt']['e_vqe_fixed'])))
    return out


def summarise(data, T=310.15):
    R, TS, P = data['reactant'], data['ts'], data['product']
    comp = data.get('composite') or {}
    for g in ('reactant', 'ts', 'product'):
        data[g]['_comp'] = {k.split('/')[1]: v for k, v in comp.items() if k.startswith(g + '/')}
    nu = TS['imag_cm']
    zpe_a = (TS['zpe'] - R['zpe']) * H
    zpe_r = (P['zpe'] - R['zpe']) * H
    g_a = (TS['g_corr'] - R['g_corr']) * H
    summary = {}
    for lab, g in levels(data):
        dEa = (g(TS) - g(R)) * H
        dEr = (g(P) - g(R)) * H
        dGa = dEa + g_a
        k_cl = K.eyring(dGa, T)
        kw = K.wigner(nu, T)
        ke, _ = eckart_kappa(nu, dEa + zpe_a, dEr + zpe_r, T)
        summary[lab] = dict(dE_barrier=dEa, dE_rxn=dEr, dG_barrier=dGa,
                            k_classical=k_cl, kappa_wigner=kw, kappa_eckart=ke,
                            k_wigner=k_cl * kw, k_eckart=k_cl * ke)
    for g in ('reactant', 'ts', 'product'):
        data[g].pop('_comp', None)
    data['summary'] = summary
    data['T'] = T
    data['imag_cm'] = nu
    return data


def main(paths):
    for p in map(Path, paths):
        d = json.loads(p.read_text())
        cf = p.with_name(p.stem + '_composite.json')
        if cf.exists():
            d['composite'] = json.loads(cf.read_text())
        d = summarise(d)
        p.write_text(json.dumps(d, indent=1, default=float))
        print(p.name)
        for lab, s in d['summary'].items():
            print(f'  {lab:20s} Ea={s["dE_barrier"]:6.2f} dE={s["dE_rxn"]:6.2f} '
                  f'dG‡={s["dG_barrier"]:6.2f}  kW={s["kappa_wigner"]:.2f} '
                  f'kE={s["kappa_eckart"]:.3g}  k=[{s["k_wigner"]:.2e}, {s["k_eckart"]:.2e}] s^-1')


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
