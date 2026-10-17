# Running the thymidylate synthase cluster on the NUC

The 96-atom active-site cluster is too heavy for a 2-core cloud container.
Everything is checkpointed in `results/ts_cluster_run.json`, so a run that is
interrupted resumes where it stopped.

## Setup (LXC container, Debian/Ubuntu)

One script does everything: system packages, venv, dependencies, tests, and a
thread-timing check.

```bash
# from a clone
bash setup_nuc.sh
# or from an unzipped copy
bash setup_nuc.sh /path/to/chemoqbit
```

It works around two traps found the hard way:

* Debian patches setuptools so geomeTRIC's build fails with
  `AttributeError: install_layout`; a venv with fresh setuptools fixes it.
* **tblite is slower multi-threaded** on systems this size: 0.13 s per
  gradient at `OMP_NUM_THREADS=1` against 1.75 s at 2. The run script pins
  the xTB stages to one thread and gives DFT all cores.

`data/1TLS.pdb` is in the repository; nothing else needs downloading.

## The model

From PDB 1TLS chain A (ecTS·FdUMP·CH2H4folate, 2.60 Å), 96 atoms, charge +1:

| Fragment | Modelled as |
|---|---|
| FdUMP | 5-methylene-dihydrouracil, N1 capped with H (F5 deleted, sugar-phosphate dropped) |
| CH2H4folate | tetrahydropterin, C9 capped with H; C6-H is the hydride donor |
| Cys146 | methylthio, covalently bonded to C6 of the nucleotide |
| Arg166 | methylguanidinium (+1) |
| Tyr94 | phenol |
| Asn177 | acetamide (H-bond to O4, 2.77 Å in the crystal) |
| His147 | 4-methylimidazole, NE2-H tautomer |
| Waters | HOH 387, 327, 453 |

1TLS is the **trapped** complex: C11 bridges the nucleotide C5 and the folate
N5. The builder cuts that bond and deletes F5, which is exactly the step 5-FU
blocks, leaving the exocyclic methylene intermediate whose hydride transfer we
then compute.

Cap hydrogens and the four outer heavy atoms (Tyr94 CG, His147 CB, Asn177 CB,
Arg166 CD) are frozen; everything else relaxes.

**Known starting-geometry issue.** In the relaxed 1TLS conformation the donor
and acceptor carbons are 2.50 Å apart, but the hydride points roughly
perpendicular to the acceptor (H···C11 2.69 Å). The crystal captures the state
*before* the folate reorients. The ξ-scan is what drives that reorganisation,
so do not expect a low barrier from the first scan points.

## Running

```bash
tmux new -s chemoqbit     # so the run survives a disconnect
bash run_nuc.sh           # every stage, right thread count for each
```

`run_nuc.sh` logs to `results/nuc_<date>.log` and prints the barrier and the
isotope effects at the end. To drive the stages yourself:

```bash
# stages 0-4: build, scan, saddle, Hessians, KIE  (GFN2-xTB; a few hours on 8 cores)
OMP_NUM_THREADS=8 .venv/bin/python scripts/ts_cluster_nuc.py --stages 0,1,2,3,4

# stage 5: DFT single points, B3LYP-D3(BJ)/def2-SVP, C-PCM eps=4
OMP_NUM_THREADS=8 .venv/bin/python scripts/ts_cluster_nuc.py --stages dft --basis def2-svp
OMP_NUM_THREADS=8 .venv/bin/python scripts/ts_cluster_nuc.py --stages dft --basis def2-tzvp

# stage 6: CCSD(T) + UCCSD/ADAPT circuits on a small space around the hydride
OMP_NUM_THREADS=8 .venv/bin/python scripts/ts_cluster_nuc.py --stages active
```

Stages can be run separately and in any order after `prep`; each skips work
already in the checkpoint. Delete the corresponding key in
`results/ts_cluster_run.json` to redo a stage.

## What to compare against

| Quantity | Literature |
|---|---|
| Hydride-transfer barrier | 12.9 kcal/mol from 1TLS, 26.3 from 2TSC (QM/MM, Świderek 2015); 26.5 (AM1:M06-2X/MM, JCTC 2017) |
| TS donor–acceptor distance | 2.70–2.89 Å |
| Intrinsic KIE H/T at 293 K | 6.91 ± 0.05, temperature-independent |
| Secondary KIE H/T on kcat/KM | 1.104 ± 0.004 |
| Tunnelling factor κ(H) | 9–32 |

A cluster model is not QM/MM: it will not reproduce the protein's
electrostatics, and the barrier may land high. The KIE is the more robust
target, because it depends on the transition-state geometry and frequencies
rather than on the absolute energy.

## If the barrier comes out far too high

1. Check the scan: did the folate reorient, or did the hydride get pushed
   through a wall? Watch the `DAD` column fall toward 2.7–2.9 Å near the top.
2. Try starting from 1TSN (2.20 Å, same complex, better resolution).
3. Relax more of the environment: remove entries from `CAP_FREEZE` in
   `scripts/ts_cluster_nuc.py`.
4. Add the second shell: Trp80, Trp83 and HOH453 sit within 4.1 Å and are not
   in the present model.
