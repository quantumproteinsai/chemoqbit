# chemoqbit

Quantum chemistry of chemotherapy-relevant reactions on qubits — end to end,
from PySCF integrals to the 32-bit instruction stream of the **qsim-pynq**
FPGA state-vector simulator (PYNQ-Z2).

Educational and methodological. Nothing here offers a quantum advantage;
the value is a complete, checkable workflow on hardware you can build.

## Pipeline

```
B3LYP/6-31G(d) geometries + frequencies   (reactant, TS, product; geomeTRIC)
        │
CCSD(T)/cc-pVDZ single points              (classical accuracy reference)
        │
CCSD natural orbitals → active space (4e,4o) / (6e,6o)
        │
Jordan–Wigner → Pauli Hamiltonian          (chemoqbit.pauli)
        │
UCCSD or ADAPT-VQE, adjoint gradients      (chemoqbit.circuit, chemoqbit.chem)
        │
gate compilation → peephole → 32-bit ISA   (chemoqbit.isa)
        │
fixed-point emulation of the FPGA datapath (FixedSim, 18-bit words)
        │
composite: CCSD(T) + (E_VQE − E_CCSD(T)) in the same active space   (scripts/composite.py)
        │
TST rate, tunnelling bracketed by Wigner and a fitted Eckart barrier at 310 K
```

Every VQE energy is validated against PySCF's exact diagonalisation in the
same active space (CASCI).

## Layout

| Path | Contents |
|---|---|
| `chemoqbit/pauli.py` | Pauli algebra, Jordan–Wigner, sparse matrices, measurement grouping |
| `chemoqbit/circuit.py` | UCCSD excitations, Pauli-exponential compilation, float and fixed-point simulators, adjoint gradient |
| `chemoqbit/chem.py` | active-space problems, `run_vqe` (UCCSD), `run_adapt` (ADAPT-VQE) |
| `chemoqbit/isa.py` | 32-bit instruction encoding for qsim-pynq, peephole optimiser |
| `chemoqbit/kinetics.py` | Eyring, Wigner (Skodje–Truhlar kept for reference only) |
| `chemoqbit/eckart.py` | asymmetric Eckart tunnelling, transfer-matrix transmission |
| `chemoqbit/summarise.py` | barriers, rates and composite levels from saved results |
| `chemoqbit/workflow.py` | reactant/TS/product driver used by all case studies |
| `scripts/formamide_case.py` | case 1: formamide → formamidic acid |
| `scripts/fivefu_case.py` | case 2: 5-fluorouracil N3-H → O4 lactim |
| `scripts/composite.py` | CCSD(T) inside each active space, for the composite correction |
| `scripts/precision_sweep.py` | fixed-point word length vs energy error |
| `scripts/diagnostics.py` | AVAS spaces, ADAPT plateau, peephole savings, tunnelling-model comparison |
| `service/` | FastAPI module + embeddable panel for the oncology platform |
| `tests/` | pytest suite (H₂ and LiH against FCI, gradients, ISA round-trip) |

## Instruction word (qsim-pynq)

```
[31:28] opcode  0 NOP · 1 H · 2 RX · 3 RZ · 4 CNOT · 15 END
[27:23] qubit a [22:18] qubit b  [17:16] reserved
[15:0]  angle   signed, units of π/2^15
```

## Running

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install pyscf geometric rdkit fastapi uvicorn pytest
.venv/bin/python -m pytest -q tests
OMP_NUM_THREADS=2 .venv/bin/python scripts/formamide_case.py   # ~15 min on 2 cores
OMP_NUM_THREADS=2 .venv/bin/python scripts/fivefu_case.py      # ~2.5 h on 2 cores; checkpointed, re-run resumes
OMP_NUM_THREADS=2 .venv/bin/python scripts/composite.py formamide fivefu
.venv/bin/python -m chemoqbit.summarise results/formamide.json results/fivefu.json
.venv/bin/uvicorn service.app:app --host 0.0.0.0 --port 8077
```

Geometries are cached in `results/<case>_geom.json`; delete the file to
re-optimise.

## Service endpoints

`/api/cases`, `/api/cases/{tag}`, `/api/cases/{tag}/circuit?geometry=ts&space=6e6o&method=adapt`
(hex words, one per line, ready for AXI-DMA), `/embed/{tag}` (iframe panel).

## Limits worth stating

* Circuits reproduce their exact active space (CASCI) to ≤ 0.3 kcal/mol, but
  active-space barriers alone are poor: +8 kcal/mol (formamide) and +24 kcal/mol
  (5-FU, 12 qubits) versus CCSD(T). The barrier is lowered by dynamical
  correlation spread over many orbitals; AVAS spaces on the reacting atoms do
  not fix it within 16 qubits. Use the composite levels.
* Composite barriers (CCSD(T) + circuit correction): 46.2 kcal/mol (formamide)
  and 45.7 kcal/mol (5-FU), within 0.13 kcal/mol of CCSD(T). The static
  correction is at most 0.31 mHa per structure, comparable to or smaller than
  the circuits' own residual error (0.2–1.0 mHa at 12 qubits).
* Tunnelling is bracketed (Wigner ≈ 4, Eckart ≈ 2–4 × 10³); Skodje–Truhlar
  gives an unphysical ~10⁷ for these barriers and is not used.
* 18-bit amplitude words keep the arithmetic error ≤ 0.17 mHa on every
  12-qubit production circuit (up to 13.4k gates); 16 bits is marginal.
* 16 qubits is the on-chip ceiling for 18-bit complex amplitudes in the
  PYNQ-Z2's 630 KiB of block RAM (288 KiB per copy at 16 qubits); beyond that the state vector spills to DDR.
* Gas-phase, unassisted proton shifts. Water-assisted paths (much lower
  barriers) are the natural next case.
* Not for clinical use.
