"""Free-energy barriers from the ONIOM(wB97X-D3:xTB) stationary points, for
comparison with the experimental barrier implied by kcat.

G = E_el + ZPE + H_vib(T) - T S_vib(T), rigid-rotor/harmonic-oscillator for the
free atoms of the partial-Hessian region (the cluster's fixed atoms remove
translation and rotation). The imaginary mode is dropped at the transition
state. Two treatments of the low-frequency modes are reported: plain harmonic,
and quasi-harmonic, with frequencies below 100 cm-1 raised to 100 cm-1 in the
entropy and thermal enthalpy (soft cluster modes otherwise inflate S).
E_el is the def2-TZVP ONIOM energy at the def2-SVP ONIOM geometry.

    python scripts/ts_thermo.py        # writes results/ts_thermo.json
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from chemoqbit.enzyme import tsopt as T

HC_K = T.H_PL * T.C_CM / T.KB            # cm -> K (second radiation constant)
EH_KCAL, KB_EH = 627.509474, T.KB / T.HARTREE_J
NAMES = ('reactant', 'ts', 'product')
MODELS = {'His147 neutral': ('ts_cluster_run.json', 'ts_oniom_opt.json'),
          'His147 protonated': ('ts_cluster_run.his147p.json', 'ts_oniom_opt.his147p.json'),
          'O4 protonated': ('ts_cluster_run.o4h.json', 'ts_oniom_opt.o4h.json')}


def thermo(nu, Tk, floor=None):
    """ZPE, H_vib(T) and S_vib(T) in Eh, Eh/K from real frequencies (cm-1)"""
    nu = np.asarray(nu, float)
    zpe = 0.5 * T.H_PL * T.C_CM * nu.sum() / T.HARTREE_J
    if floor:
        nu = np.maximum(nu, floor)
    x = HC_K * nu / Tk
    h = KB_EH * Tk * np.sum(x / np.expm1(x))
    s = KB_EH * np.sum(x / np.expm1(x) - np.log1p(-np.exp(-x)))
    return zpe, h, s


def main():
    out = {}
    for model, (cfile, ofile) in MODELS.items():
        cp, op_ = ROOT / 'results' / cfile, ROOT / 'results' / ofile
        if not (cp.exists() and op_.exists()):
            continue
        r, o = json.loads(cp.read_text()), json.loads(op_.read_text())
        if 'summary' not in o:
            continue
        sym = r['prep']['symbols']; region = o['config']['hessian_region']
        m = [T.AMU[s] for s in sym]
        res = {}
        for Tk in (293.15, 298.15):
            for label, floor in (('harmonic', None), ('quasi-harmonic', 100.0)):
                G = {}
                for n in NAMES:
                    nu = T.frequencies(np.array(o[n]['X']), np.array(o['hess'][n]['H']), m, free=region)
                    real = nu[nu > 0]
                    zpe, h, s = thermo(real, Tk, floor)
                    G[n] = dict(E=o['sp'][n]['E'], zpe=zpe, h=h, s=s, G=o['sp'][n]['E'] + zpe + h - Tk * s)
                d = lambda a, b, kk: (G[a][kk] - G[b][kk]) * EH_KCAL
                res[f'{Tk:.2f} K, {label}'] = dict(
                    dE=d('ts', 'reactant', 'E'), dZPE=d('ts', 'reactant', 'zpe'),
                    TdS=Tk * (G['ts']['s'] - G['reactant']['s']) * EH_KCAL,
                    dG_barrier=d('ts', 'reactant', 'G'), dG_reaction=d('product', 'reactant', 'G'))
        out[model] = res
        for key, v in res.items():
            print(f"{model:18s} {key:28s} dE {v['dE']:6.2f}  dZPE {v['dZPE']:+5.2f}  T*dS {v['TdS']:+5.2f}  "
                  f"dG barrier {v['dG_barrier']:6.2f}  dG reaction {v['dG_reaction']:+6.2f} kcal/mol")
    (ROOT / 'results' / 'ts_thermo.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
