#!/usr/bin/env python
"""Oxidised histidines versus distance to copper, versus solvent exposure.

Rowlands Supplementary Table 4 reports oxidation at SIX histidines: H43, H46, H48,
H63, H110, H120. Human SOD1 has eight. The two that are NOT oxidised are H71 and H80 --
both zinc ligands. That is a control group the experiment supplies for free.

If solvent exposure permits the oxidation, the exposed histidines should be the hit
ones. If instead the oxidant is generated at the copper site, proximity to copper
should predict it and exposure should not.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

from chemoqbit.sasa import sasa

MODIFIED_AA = {"CSO", "CSD", "CME", "OCS", "MSE", "SEP", "TPO", "PTR", "KCX"}
# Supplementary Table 4, oxidation (H), log2 ratio SOCK vs hSOD1WT. None = not reported.
OXID = {43: 0.97, 46: 0.97, 48: 2.83, 63: 3.57, 110: 1.00, 120: 1.12, 71: None, 80: None}
LIGAND = {46: "Cu", 48: "Cu", 63: "Cu+Zn", 120: "Cu", 71: "Zn", 80: "Zn",
          43: "-", 110: "-"}

def parse(path):
    at, metals = defaultdict(list), defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")) or line[16] not in (" ", "A"):
            continue
        resn, ch = line[17:20].strip(), line[21]
        el = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        if el in ("H", "D"):
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if line.startswith("HETATM") and resn in ("CU", "ZN"):
            metals[ch].append((resn, np.array(xyz)))
            at[ch].append((resn, None, el, xyz))
        elif line.startswith("ATOM") or (line.startswith("HETATM") and resn in MODIFIED_AA):
            at[ch].append((resn, int(line[22:26]), el, xyz))
    return at, metals

data = Path(sys.argv[1])
rows = defaultdict(lambda: {"d_cu": [], "d_zn": [], "sasa": []})
# 2C9V is the atomic-resolution Cu-Zn structure: the one with copper where copper belongs
for pdb in ("2C9V", "1HL5"):
    f = next(iter((data / f"rcsb_{pdb.lower()}").glob("*.pdb")), None)
    if f is None:
        continue
    at, metals = parse(f)
    for ch, atoms in at.items():
        resi = {a[1] for a in atoms if a[1] is not None}
        if not {6, 46, 57, 146} <= resi:
            continue
        cu = [p for n, p in metals[ch] if n == "CU"]
        zn = [p for n, p in metals[ch] if n == "ZN"]
        if not cu or not zn:
            continue
        sym = [a[2] if a[2] in "CNOSP" else "C" for a in atoms]
        per = sasa(sym, [a[3] for a in atoms], n_points=128)
        for r in OXID:
            pts = [np.array(a[3]) for a in atoms if a[1] == r]
            if not pts:
                continue
            rows[r]["d_cu"].append(min(np.linalg.norm(p - c) for p in pts for c in cu))
            rows[r]["d_zn"].append(min(np.linalg.norm(p - z) for p in pts for z in zn))
            rows[r]["sasa"].append(sum(per[i] for i, a in enumerate(atoms) if a[1] == r))

print("Human SOD1 has eight histidines. Six are reported oxidised; two are not.\n")
print(f"{'His':<6}{'ligand':<8}{'oxidised?':<22}{'min dist to Cu':>16}{'min dist to Zn':>16}{'SASA':>9}")
out = []
for r in sorted(OXID, key=lambda r: (rows[r]['d_cu'] and np.mean(rows[r]['d_cu']) or 99)):
    if not rows[r]["d_cu"]:
        continue
    dcu, dzn = np.mean(rows[r]["d_cu"]), np.mean(rows[r]["d_zn"])
    sa = np.mean(rows[r]["sasa"])
    ox = f"yes  log2 {OXID[r]:+.2f}" if OXID[r] is not None else "NO"
    print(f"  H{r:<4}{LIGAND[r]:<8}{ox:<22}{dcu:>13.2f} A{dzn:>13.2f} A{sa:>9.1f}")
    out.append(dict(residue=f"H{r}", ligand=LIGAND[r], oxidised=OXID[r],
                    d_cu=round(float(dcu), 2), d_zn=round(float(dzn), 2), sasa=round(float(sa), 1)))

ox = [o for o in out if o["oxidised"] is not None]
no = [o for o in out if o["oxidised"] is None]
print(f"\n  oxidised  (n={len(ox)}): Cu distance {np.mean([o['d_cu'] for o in ox]):5.2f} A, "
      f"SASA {np.mean([o['sasa'] for o in ox]):5.1f} A^2")
print(f"  NOT oxid. (n={len(no)}): Cu distance {np.mean([o['d_cu'] for o in no]):5.2f} A, "
      f"SASA {np.mean([o['sasa'] for o in no]):5.1f} A^2")
r = np.corrcoef([o["d_cu"] for o in ox], [o["oxidised"] for o in ox])[0, 1]
print(f"\n  among the six oxidised, correlation of log2 oxidation with Cu distance: r = {r:+.2f}")
Path(sys.argv[2]).write_text(json.dumps(dict(
    question="Does proximity to copper, or solvent exposure, predict which histidines oxidise?",
    source="Rowlands et al. 2025, Supplementary Table 4; geometry from 2C9V and 1HL5",
    residues=out), indent=1))
print(f"\nwritten: {sys.argv[2]}")
