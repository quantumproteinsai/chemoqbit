"""Temperature dependence of the H/T kinetic isotope effect, against the
measured values for E. coli thymidylate synthase.

Experiment: H/T KIE 7.62 +- 0.4 (278 K), 6.91 +- 0.05 (293 K), 7.40 +- 0.55
(303 K), 7.12 +- 0.17 (313 K) (Agrawal et al. 2004, as tabulated by Swiderek
et al. 2017, Table 8B); Arrhenius fit A_H/A_T = 6.8 +- 2.8, dEa(H-T) =
0.02 +- 0.25 kcal/mol, i.e. a temperature-independent KIE.

For each model and level, the KIE (TST, Wigner, Eckart) at the four
temperatures and the Arrhenius parameters from ln KIE = ln(A_H/A_T) + dEa/RT,
with dEa = Ea(T) - Ea(H). Uses kie_analysis, as the pipeline does.

    python scripts/ts_kie_temperature.py      # writes results/ts_kie_temperature.json
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
R_KCAL = 1.987204e-3
TEMPS = (278.15, 293.15, 303.15, 313.15)
EXP = {278.15: (7.62, 0.4), 293.15: (6.91, 0.05), 303.15: (7.40, 0.55), 313.15: (7.12, 0.17)}
NAMES = ('reactant', 'ts', 'product')
LEVELS = [('His147 neutral', 'GFN2-xTB', 'ts_cluster_run.json', None),
          ('His147 neutral', 'ONIOM', 'ts_cluster_run.json', 'ts_oniom_opt.json'),
          ('His147 protonated', 'GFN2-xTB', 'ts_cluster_run.his147p.json', None),
          ('His147 protonated', 'ONIOM', 'ts_cluster_run.his147p.json', 'ts_oniom_opt.his147p.json'),
          ('O4 protonated', 'GFN2-xTB', 'ts_cluster_run.o4h.json', None),
          ('O4 protonated', 'ONIOM', 'ts_cluster_run.o4h.json', 'ts_oniom_opt.o4h.json')]


def arrhenius(Ts, kies):
    x, y = 1 / (R_KCAL * np.array(Ts)), np.log(kies)
    slope, icpt = np.polyfit(x, y, 1)
    return float(np.exp(icpt)), float(slope)          # A_H/A_T, dEa = Ea(T) - Ea(H)


def main():
    import ts_cluster_nuc as S
    out = {}
    for model, level, cfile, ofile in LEVELS:
        cp = ROOT / 'results' / cfile
        if not cp.exists():
            continue
        r = json.loads(cp.read_text()); p = r['prep']
        if ofile is None:
            if 'analysis' not in r:
                continue
            free = S._free(p['symbols'], p['frozen'])
            coords = {n: r[n]['coords'] for n in NAMES}; hess = {n: r[n]['hessian'] for n in NAMES}
            ener = {n: r[n]['E'] for n in NAMES}
        else:
            op = ROOT / 'results' / ofile
            o = json.loads(op.read_text()) if op.exists() else {}
            if 'summary' not in o:
                continue
            free = o['config']['hessian_region']
            coords = {n: o[n]['X'] for n in NAMES}; hess = {n: o['hess'][n]['H'] for n in NAMES}
            ener = {n: o[n]['E'] for n in NAMES}
        rows = {}
        for Tk in TEMPS:
            _, _, _, kie = S.kie_analysis(p['symbols'], p['key'], free, coords, hess, ener, T_K=Tk)
            rows[Tk] = {m: kie['H/T'][m] for m in ('semiclassical', 'wigner', 'eckart')}
        fit = {m: arrhenius(TEMPS, [rows[t][m] for t in TEMPS]) for m in ('semiclassical', 'wigner', 'eckart')}
        out[f'{model} / {level}'] = dict(kie={str(t): v for t, v in rows.items()},
                                         arrhenius={m: dict(A_ratio=a, dEa=d) for m, (a, d) in fit.items()})
        print(f'{model} / {level}')
        for t in TEMPS:
            v = rows[t]
            print(f"  {t:.0f} K  TST {v['semiclassical']:.2f}  Wigner {v['wigner']:.2f}  Eckart {v['eckart']:.2f}"
                  f"   [exp {EXP[t][0]:.2f} +- {EXP[t][1]:.2f}]")
        for m, (a, d) in fit.items():
            print(f'  Arrhenius {m:13s} A_H/A_T {a:.2f}  dEa(H-T) {d:.2f} kcal/mol   [exp 6.8 +- 2.8, 0.02 +- 0.25]')
    (ROOT / 'results' / 'ts_kie_temperature.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
