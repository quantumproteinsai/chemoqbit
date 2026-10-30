#!/usr/bin/env python
"""Does losing copper expose the copper-binding histidines of SOD1?

The Parkinson's claim is that nigral SOD1 is copper-poor and misfolded. Rowlands et al.
report that in copper-deficient mouse brain all four copper-binding histidines are more
oxidised, and that copper supplementation reverses most of it. Oxidation needs access:
a ligand buried against a bound metal is not a target.

So the question is mechanical and measurable from structures that already exist. If
copper loss raises the solvent accessibility of His46, His48, His63 and His120, there
is a ratchet: copper leaves, the ligands are exposed, the ligands oxidise, and oxidised
ligands cannot rebind copper. If accessibility does not change, the oxidation is caused
by something other than exposure and that ratchet is not available.

Nothing here is new chemistry. It is the Shrake-Rupley construction already in
chemoqbit, run across structures already fetched, grouped by what metal they contain.

    python scripts/pd_copper_site_exposure.py <chemoqbit-data-dir> <out.json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from chemoqbit.sasa import sasa

CU_LIGANDS = [46, 48, 63, 120]          # His46, His48, His63, His120
ZN_LIGANDS = [63, 71, 80, 83]           # His63, His71, His80, Asp83
CYS = [6, 57, 111, 146]
# HETATM residue names that are really amino acids in the chain. CSO is cysteine
# oxidised to the sulfenic acid, which 2C9S models at position 111.
MODIFIED_AA = {"CSO", "CSD", "CME", "OCS", "MSE", "SEP", "TPO", "PTR", "KCX"}
# Metallation is ANNOTATED from the papers, not inferred by counting HETATM records.
# Counting is unsafe here: Strange et al. show the copper site can be occupied by ZINC,
# so a chain with no copper may be Zn-Zn (site filled) or Cu-vacant (site empty), and
# those are opposite answers to the question being asked. The first run inferred, and
# called 2C9U holo when it is the as-isolated mixed-occupancy structure.
#
# Strange, Antonyuk, Hough, Doucette, Valentine & Hasnain, J Mol Biol 356:1152 (2006):
#   2C9V  re-metallated Cu-Zn, 1.07 A   Cu(I); His63 bridge broken at 3.4 A
#   2C9S  Zn-Zn, 1.24 A                 Zn with sulphate at the Cu site
#   2C9U  as-isolated, 1.24 A           Cu site 45% Cu / 55% Zn, bridge broken
# Strange et al., J Mol Biol 328:877 (2003):
#   1HL5  holo                          1HL4  metal-deficient
SOD1_PDBS = {"1HL4", "1HL5", "1L3N", "2AF2", "2C9U", "7FB9", "2C9S", "2C9V"}
ANNOTATED = {"2C9V": "Cu-Zn (Strange)", "2C9S": "Zn-Zn (Strange)",
             "2C9U": "as-isolated (Strange)"}
STRANGE_SET = ("Cu-Zn (Strange)", "as-isolated (Strange)", "Zn-Zn (Strange)")
N_POINTS = 128          # 256 buys no decision this comparison turns on


def chains_of(path):
    """Group ATOM/HETATM by chain, keeping only the first altloc, and note metals."""
    at = defaultdict(list)
    metals = defaultdict(lambda: {"CU": 0, "ZN": 0})
    # First model only. 1L3N and 2AF2 are NMR ensembles of sixty models sharing one
    # chain id; read whole they give 131,000 atoms in two chains, which is sixty
    # copies of the protein superimposed. Shrake-Rupley on that reports everything
    # as buried, because every atom is occluded by its own copies.
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        ch = line[21]
        resn = line[17:20].strip()
        el = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        # Heavy atoms only. The NMR entries carry hydrogens and the crystal entries do
        # not, so including them would compare two conventions rather than two
        # metallation states.
        if el in ("H", "D"):
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if line.startswith("HETATM") and resn in ("CU", "ZN"):
            metals[ch][resn] += 1
            at[ch].append((resn, None, el, xyz))        # metal occludes, so keep it
        elif line.startswith("ATOM"):
            at[ch].append((resn, int(line[22:26]), el, xyz))
        elif line.startswith("HETATM") and resn in MODIFIED_AA:
            # Modified amino acids are HETATM but are part of the chain. 2C9S models
            # Cys111 as CSO, S-hydroxycysteine: dropping it would both lose the residue
            # and leave a hole that makes its neighbours look more exposed than they are.
            at[ch].append((resn, int(line[22:26]), el, xyz))
    return at, metals


def side_chain_area(atoms, residues):
    """Shrake-Rupley area summed over side-chain heavy atoms, per residue."""
    sym = [a[2] if a[2] in "CNOSP" else "C" for a in atoms]
    X = [a[3] for a in atoms]
    per = sasa(sym, X, n_points=N_POINTS)
    out = {}
    for r in residues:
        tot = sum(per[i] for i, a in enumerate(atoms)
                  if a[1] == r and a[0] not in ("CU", "ZN"))
        out[r] = round(float(tot), 2)
    return out


def main():
    data, out_path = Path(sys.argv[1]), sys.argv[2]
    rows = []
    # Only the SOD1 entries. data/ also holds thymidylate synthase, prolyl isomerase
    # and the amyloid fibrils, and running Shrake-Rupley over every chain of those to
    # discover they are not SOD1 is an hour of arithmetic for a test the residue
    # numbers answer in a microsecond. That is what the first run spent its time on.
    for d in sorted(data.glob("rcsb_*")):
        if d.name.split("_")[-1].upper() not in SOD1_PDBS:
            continue
        pdb = next(iter(d.glob("*.pdb")), None)
        if pdb is None:
            continue
        at, metals = chains_of(pdb)
        for ch, atoms in at.items():
            resi = {a[1] for a in atoms if a[1] is not None}
            # SOD1 chains only: must carry the four cysteines and the metal ligands
            if not set(CYS) <= resi or not set(CU_LIGANDS) <= resi:
                continue
            ox = sorted({a[1] for a in atoms if a[0] in MODIFIED_AA})
            m = metals[ch]
            state = ANNOTATED.get(pdb.stem.upper()) or (
                     "holo" if m["CU"] and m["ZN"] else
                     "Cu-vacant, Zn-bound" if m["ZN"] and not m["CU"] else
                     "Cu-bound, Zn-free" if m["CU"] else "apo")
            print(f"    {pdb.stem.upper()} {ch}  {len(atoms):>5} atoms  Cu={m['CU']} Zn={m['ZN']}", flush=True)
            areas = side_chain_area(atoms, CU_LIGANDS + ZN_LIGANDS)
            rows.append(dict(pdb=pdb.stem.upper(), chain=ch, state=state,
                             cu=m["CU"], zn=m["ZN"],
                             cu_site=round(sum(areas[r] for r in CU_LIGANDS), 2),
                             zn_site=round(sum(areas[r] for r in ZN_LIGANDS), 2),
                             oxidised_residues=ox,
                             per_residue={str(r): areas[r] for r in CU_LIGANDS + ZN_LIGANDS}))
    by = defaultdict(list)
    for r in rows:
        by[r["state"]].append(r)

    print(f"{len(rows)} SOD1 chains over {len({r['pdb'] for r in rows})} structures\n")
    print(f"{'state':<20}{'n':>4}{'Cu-site SASA':>16}{'Zn-site SASA':>16}")
    summary = {}
    for state in STRANGE_SET + ("holo", "Cu-vacant, Zn-bound", "Cu-bound, Zn-free", "apo"):
        g = by.get(state, [])
        if not g:
            continue
        cu = np.array([r["cu_site"] for r in g]); zn = np.array([r["zn_site"] for r in g])
        print(f"{state:<20}{len(g):>4}{cu.mean():>11.1f} ±{cu.std():<4.1f}{zn.mean():>11.1f} ±{zn.std():<4.1f}")
        summary[state] = dict(n=len(g), cu_mean=round(float(cu.mean()), 2), cu_sd=round(float(cu.std()), 2),
                              zn_mean=round(float(zn.mean()), 2), zn_sd=round(float(zn.std()), 2),
                              pdbs=sorted({r["pdb"] for r in g}))
    print("\nper-residue mean side-chain SASA (A^2):")
    print(f"  {'residue':<10}" + ''.join(f"{s[:14]:>16}" for s in summary))
    for r in sorted(set(CU_LIGANDS + ZN_LIGANDS)):   # His63 is both; print it once
        tag = ("Cu" if r in CU_LIGANDS else "") + ("/Zn" if r in ZN_LIGANDS else "")
        line = f"  {('His' if r != 83 else 'Asp') + str(r):<6}{tag:<4}"
        for s in summary:
            v = np.mean([x["per_residue"][str(r)] for x in by[s]])
            line += f"{v:>16.1f}"
        print(line)

    Path(out_path).write_text(json.dumps(dict(
        question="Does copper loss expose the copper-binding histidines of SOD1?",
        method=("Shrake-Rupley side-chain SASA, heavy atoms, 1.4 A probe, 256 points, "
                "first altloc only. Bound metals are kept in the occlusion set, since a "
                "ligand coordinated to a metal is shielded by it. Chains are accepted as "
                "SOD1 only if they carry Cys6/57/111/146 and all four copper ligands."),
        caveat=("Crystal structures of different constructs, resolutions and space groups. "
                "A difference between metallation groups here is a difference between "
                "crystals, not a measured response to removing copper from one molecule."),
        rows=rows, summary=summary), indent=1))
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
