---
title: 'chemoqbit: reaction energetics on qubits, from molecular integrals to the instruction stream of an FPGA state-vector simulator'
tags:
  - Python
  - quantum chemistry
  - variational quantum eigensolver
  - ADAPT-VQE
  - QAOA
  - fixed-point arithmetic
  - FPGA
  - enzyme catalysis
  - tunnelling
authors:
  - name: Bertrand Mercier des Rochettes
    orcid: 0000-0002-1145-1881
    affiliation: 1
affiliations:
  - name: Quantum Proteins AI, Cergy-Pontoise, France
    index: 1
date: 20 October 2026
bibliography: paper.bib
---

# Summary

`chemoqbit` computes the energetics of chemical reactions with quantum circuits,
end to end. It starts from molecular integrals and ends with reaction rates, and
in between it produces the exact 32-bit instruction stream that a hardware
state-vector simulator executes. The target is qsim-pynq, a state-vector
simulator on a Xilinx PYNQ-Z2 board, whose datapath stores amplitudes as
fixed-point words. `chemoqbit` emulates that datapath bit for bit, so the effect
of finite word length on a chemical energy can be measured before anything runs
on the board.

The package covers four stages:

1. **Electronic structure.** Active spaces are built from PySCF [@pyscf2020]
   integrals and CCSD natural orbitals, optionally embedded in point charges,
   and mapped to qubits by the Jordan–Wigner transformation. Every circuit
   energy is checked against exact diagonalisation (CASCI) in the same space.
2. **Circuits.** UCCSD-VQE [@peruzzo2014] and fermionic ADAPT-VQE
   [@grimsley2019] use analytic adjoint gradients. The optimised circuits are
   compiled to Pauli exponentials, simplified by a peephole optimiser, encoded
   as 32-bit instruction words, and re-simulated in fixed point. For QAOA
   [@farhi2014], a diagonal cost layer is a single `DIAG` instruction rather than
   a compiled gate sequence.
3. **Reaction energetics.** Transition-state theory rates, Wigner and Eckart
   tunnelling corrections, and composite levels that combine CCSD(T) with the
   circuit result in the same active space.
4. **Applications.** Proton-transfer tautomerism of formamide and
   5-fluorouracil; a lattice protein-folding benchmark on the diamond lattice
   with Miyazawa–Jernigan contact energies [@miyazawa1985] and the dense turn
   encoding of @robert2021; and a 96-atom active-site cluster of thymidylate
   synthase built from a crystal structure. The cluster pipeline uses GFN2-xTB
   [@bannwarth2019], a partitioned rational-function saddle search that respects
   frozen atoms, kinetic isotope effects, and ONIOM layering of the embedded
   active space.

A small FastAPI service exposes the cases and serves their instruction streams,
ready for direct memory access to the FPGA.

# Statement of need

General-purpose quantum software such as Qiskit [@qiskit2024] and OpenFermion
[@openfermion2020] targets floating-point simulators or gate-model quantum
processors. Hardware state-vector simulators are a different target: an FPGA
holds the full state vector on chip, applies gates as streamed arithmetic on
fixed-point words, and can apply a diagonal operator in one pass over memory. To
use such a device for chemistry, a researcher needs answers that existing tools
do not give:

- Which word length keeps the energy error of a given circuit below chemical
  accuracy (1.6 mHa)? In `chemoqbit`, 18-bit amplitude words keep the error of
  every 12-qubit production circuit (up to 13,400 gates) at or below 0.17 mHa,
  while 16 bits is marginal.
- How many instructions does a circuit cost after compilation and peephole
  simplification on this instruction set?
- How does a circuit energy become a barrier or a rate, given that a small
  active space misses most of the dynamic correlation?

`chemoqbit` answers all three in one checked workflow, and it is honest about
the limits: nothing in it offers a quantum advantage, and active-space barriers
alone are poor without the composite or ONIOM corrections it provides. It is
intended for researchers designing or using FPGA quantum-circuit simulators,
for teaching the full path from integrals to hardware, and for methodological
studies of embedded active spaces in reactive systems.

# Design and quality control

Each stage writes JSON checkpoints and resumes after an interruption, which
matters for multi-hour cluster calculations. The enzyme pipeline guards against
failure modes that produce plausible but wrong numbers: energy jumps and bond
changes along a reaction scan, saddle searches that follow soft collective
modes, and optimisers that move frozen atoms. The test suite (pytest) checks
H~2~ and LiH against full configuration interaction, analytic against numerical
gradients, the instruction-set round trip, Eckart transmission, the folding
encoding and QAOA, and the web service.

# AI usage disclosure

**[To be completed and confirmed by the author before submission.]** Parts of
the code, documentation and this paper were written with the assistance of
Claude (Anthropic) through Claude Code; the sessions of October 2026 used
Claude Opus 5.5. The assistance included implementing and debugging the
thymidylate synthase cluster pipeline (`chemoqbit/enzyme`, `scripts/ts_*`), the
embedding and ONIOM stages, the report, dataset and notification scripts, and
drafting this paper. *[Add: which other parts of the package were AI-assisted,
and with which tools and versions.]* The author reviewed and validated all
AI-assisted outputs, ran and checked the tests and results, and made the core
design and scientific decisions.

# Acknowledgements

*[Funding, if any; otherwise state that the work was carried out without
external funding.]*

# References
