"""Fixed-point word length vs energy error on real chemistry circuits."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from chemoqbit.chem import build_problem, to_sparse
from chemoqbit.circuit import Ansatz, FixedSim, FloatSim
from chemoqbit.isa import encode, decode, peephole

d = json.loads((ROOT / 'results/formamide.json').read_text())
out = []
for geom in ('ts',):
    atom = '; '.join(f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in d[geom]['atoms'])
    for sp in ('4e4o', '6e6o'):
        ne, no = int(sp[0]), int(sp[2])
        p = build_problem(atom, 'cc-pvdz', no, ne)
        H = to_sparse(p.qubit_op, p.n_qubits).real.tocsr()
        for method in ('uccsd', 'adapt'):
            r = d[geom]['spaces'][f'{sp}_{method}']
            ans = (Ansatz.uccsd(p.n_qubits, ne) if method == 'uccsd' else
                   Ansatz.from_excitations(p.n_qubits, ne, [tuple(o) for o in r['operators']]))
            gates = decode(encode(peephole(ans.gates(np.array(r['theta'])))))
            e_ref = FloatSim(p.n_qubits).run(ans.hf_state(), gates)
            e_ref = float(np.real(np.vdot(e_ref, H @ e_ref)))
            for F in (10, 12, 14, 16, 18, 20, 24):
                sim = FixedSim(p.n_qubits, frac_bits=F, word_bits=F + 2, coef_bits=F)
                psi = sim.run(ans.hf_state(), gates)
                nrm = float(np.vdot(psi, psi).real)
                e = float(np.real(np.vdot(psi, H @ psi)) / nrm)
                row = dict(geometry=geom, space=sp, qubits=p.n_qubits, method=method,
                           gates=len(gates), cnots=sum(g[0] == 'CNOT' for g in gates),
                           word_bits=F + 2, err_mha=abs(e - e_ref) * 1e3, norm_drift=abs(1 - nrm))
                out.append(row)
                print(f"{sp} {method:5s} gates={row['gates']:6d} word={F+2:2d} bits  "
                      f"|dE|={row['err_mha']:.2e} mHa  norm drift={row['norm_drift']:.1e}", flush=True)
(ROOT / 'results/precision_sweep.json').write_text(json.dumps(out, indent=1))
