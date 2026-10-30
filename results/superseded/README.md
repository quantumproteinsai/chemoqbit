# Superseded result files

Checkpoints kept because a diagnosis quotes their numbers, not because the
numbers are usable. Nothing reads this directory; the audit skips it.

## zipper_sod1ct.pre-sheet-symmetry.json

The 4NIP alanine scan as it stood on 2026-10-27, before the two fixes of
2026-10-30. Its four ddE values measure which local minimum of the hydrogen
network each structure happened to fall into, not what the side chains
contribute, and its `wt` interaction energy of -3.43 kcal/mol carries the same
error. `results/diagnoses.json` under `zipper_sod1ct.json` -> `-> Ala` explains
it and quotes these numbers: wt sheet asymmetry -36.0 kcal/mol, VAL2 -69.5,
GLN7 -94.6, against an exact zero required by symmetry operation 4.

Kept so the diagnosis can be checked against the data that produced it. The
live scan starts again from scratch.
