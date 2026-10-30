#!/usr/bin/env python
"""Are the residues Rowlands finds modified the residues that copper loss exposes?

Rowlands et al. 2025 name four PTM alterations that CuATSM shifts 2-to-8-fold back
toward baseline -- oxidation H63, glycation K30, phosphorylation S25, deamidation N139
-- and report significant oxidation of all four copper-binding histidines, in the
substantia nigra of a copper-deficient mouse.

If solvent exposure is what permits those modifications, the modified residues should
be accessible, and more accessible when copper is scarce. This measures that directly,
per residue, across the metallation series rather than as a site total.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

from chemoqbit.sasa import sasa

MODIFIED_AA = {"CSO", "CSD", "CME", "OCS", "MSE", "SEP", "TPO", "PTR", "KCX"}
STATES = {"2C9V": "Cu-Zn", "2C9U": "as-isolated", "2C9S": "Zn-Zn",
          "1HL5": "holo (1HL5)", "1HL4": "1HL4 (mixed)"}
# residue -> what Rowlands reports there
REPORTED = {46: "oxidation (Cu ligand)", 48: "oxidation (Cu ligand)",
            63: "oxidation (Cu ligand), CuATSM-responsive", 120: "oxidation (Cu ligand)",
            30: "glycation, CuATSM-responsive", 25: "phosphorylation, CuATSM-responsive",
            139: "deamidation, CuATSM-responsive"}
REF = {25: "SER", 30: "LYS", 46: "HIS", 48: "HIS", 63: "HIS", 120: "HIS", 139: "ASN"}


def chain_atoms(path):
    at = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        resn, ch = line[17:20].strip(), line[21]
        el = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        if el in ("H", "D"):
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if line.startswith("HETATM") and resn in ("CU", "ZN"):
            at[ch].append((resn, None, el, xyz))
        elif line.startswith("ATOM") or (line.startswith("HETATM") and resn in MODIFIED_AA):
            at[ch].append((resn, int(line[22:26]), el, xyz))
    return at


data = Path(sys.argv[1])
out = {}
print(f"{'residue':<22}{'reported':<36}" + ''.join(f"{s:>14}" for s in STATES.values()))
per_state = defaultdict(dict)
for pdb, label in STATES.items():
    d = data / f"rcsb_{pdb.lower()}"
    f = next(iter(d.glob("*.pdb")), None)
    if f is None:
        continue
    vals = defaultdict(list)
    for ch, atoms in chain_atoms(f).items():
        resi = {a[1] for a in atoms if a[1] is not None}
        if not {6, 57, 146} <= resi or 46 not in resi:
            continue
        sym = [a[2] if a[2] in "CNOSP" else "C" for a in atoms]
        per = sasa(sym, [a[3] for a in atoms], n_points=128)
        for r in REPORTED:
            tot = sum(per[i] for i, a in enumerate(atoms)
                      if a[1] == r and a[0] not in ("CU", "ZN"))
            if any(a[1] == r for a in atoms):
                vals[r].append(float(tot))
    for r, v in vals.items():
        per_state[r][label] = float(np.mean(v))

for r in sorted(REPORTED):
    name = f"{REF[r]}{r}"
    line = f"  {name:<20}{REPORTED[r]:<36}"
    for label in STATES.values():
        v = per_state[r].get(label)
        line += f"{v:>14.1f}" if v is not None else f"{'-':>14}"
    print(line)
    out[name] = dict(reported=REPORTED[r], sasa={k: round(v, 2) for k, v in per_state[r].items()})

Path(sys.argv[2]).write_text(json.dumps(dict(
    question="Are the residues Rowlands finds modified the ones copper loss exposes?",
    source="Rowlands et al., Acta Neuropathol Commun 13:133 (2025), main text and Table 1",
    note=("Supplementary Table 4 carries the full 33-residue list and was not available; "
          "these are the residues named in the main text."),
    residues=out), indent=1))
print(f"\nwritten: {sys.argv[2]}")
