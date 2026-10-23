"""An amyloid steric zipper as a cluster model.

Two beta-sheets mate through a dry, interdigitated side-chain interface. The
model here is built from a microcrystal structure of a fibril-forming segment:
tau's PHF6 motif, 306-VQIVYK-311 (PDB 2ON9, 1.51 A; Sawaya et al., Nature 2007),
whose two chains are the two mating sheets, touching only through Val1, Ile3 and
Tyr5 -- one face of the strand.

Resolution is the reason this segment is used rather than a fibril of full-length
Abeta or tau: those cryo-EM structures cover the familial mutation sites but sit
at 3.4-4.0 A, where side-chain rotamers cannot carry a quantum-chemical argument.
A 1.51 A microcrystal can, at the price of being a fragment, so the question it
answers is what makes this sequence zip, not what a familial mutation does.

A sheet is one strand plus translated copies along the stacking axis (CRYST1 a),
since within a sheet the strands repeat by that translation.
"""
from __future__ import annotations

import numpy as np

from .pocket import read as _read

CH = 1.09
NTERM, CTERM, LYS = 1, -1, 1          # free amine, free carboxylate, lysine


def cell_a(pdb):
    for l in open(pdb):
        if l.startswith('CRYST1'):
            return float(l[6:15])
    raise ValueError('no CRYST1')


def build(pdb, chains=('A', 'B'), neighbours=0, to_alanine=()):
    """Heavy atoms of the two mating sheets.

    neighbours: strands added either side along the stacking axis (0 = one
    strand per sheet). to_alanine: residue numbers truncated to CB, applied in
    the central strand of each sheet only.
    Returns dict(symbols, X, labels, charge, sheet, n_per_strand).
    """
    a = cell_a(pdb)
    sym, X, label, sheet = [], [], [], []
    axis = np.array([a, 0.0, 0.0])
    for s, ch in enumerate(chains):
        at = _read(pdb, ch)
        res = sorted({(rn, num) for rn, num, _ in at})
        for k in range(-neighbours, neighbours + 1):
            for rn, num in res:
                keep = None if (num not in to_alanine or k != 0) else ['N', 'CA', 'C', 'O', 'CB', 'OXT']
                for (r2, n2, nm), (x, el) in at.items():
                    if (r2, n2) != (rn, num):
                        continue
                    if keep is not None and nm not in keep:
                        continue
                    sym.append(el); X.append(x + k * axis)
                    label.append(f'{ch}{k:+d}:{rn}{num}:{nm}'); sheet.append(s)
    n_strand = len(res)
    charge = len(chains) * (2 * neighbours + 1) * (NTERM + CTERM + LYS)
    charge -= sum(LYS for num in to_alanine
                  for ch in chains if any(rn == 'LYS' and n == num for rn, n, _ in _read(pdb, ch)))
    return dict(symbols=sym, X=np.array(X), labels=label, charge=charge,
                sheet=np.array(sheet), n_residues=n_strand)


def protonate(c):
    """Hydrogens and the termini/lysine formal charges."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    sym, X, lab = c['symbols'], c['X'], c['labels']
    m = Chem.RWMol()
    conf = Chem.Conformer(len(sym))
    for i, (s, x) in enumerate(zip(sym, X)):
        m.AddAtom(Chem.Atom(s))
        conf.SetAtomPosition(i, x.tolist())
    m.AddConformer(conf)
    mol = m.GetMol()
    rdDetermineBonds.DetermineConnectivity(mol, useHueckel=False, charge=c['charge'])
    em = Chem.RWMol(mol)
    idx = {l: i for i, l in enumerate(lab)}
    # the N-terminus is the lowest residue number of each strand; OXT exists only
    # at a free C-terminus, so it needs no such test
    first = {}
    for l in lab:
        strand, res, _nm = l.split(':')
        num = int(''.join(ch for ch in res if ch.isdigit()))
        first[strand] = min(first.get(strand, num), num)
    for l, i in idx.items():
        strand, res, nm = l.split(':')
        num = int(''.join(ch for ch in res if ch.isdigit()))
        if nm == 'N' and num == first[strand]:
            em.GetAtomWithIdx(i).SetFormalCharge(1)          # NH3+
        elif nm == 'OXT':
            em.GetAtomWithIdx(i).SetFormalCharge(-1)         # COO-
        elif nm == 'NZ' and res.startswith('LYS'):
            em.GetAtomWithIdx(i).SetFormalCharge(1)
    for at in em.GetAtoms():
        at.SetNoImplicit(False)
        at.SetNumExplicitHs(0)
    mol = em.GetMol()
    Chem.SanitizeMol(mol)
    molH = Chem.AddHs(mol, addCoords=True)
    Chem.SanitizeMol(molH)
    confH = molH.GetConformer()
    XH = np.array([list(confH.GetAtomPosition(i)) for i in range(molH.GetNumAtoms())])
    symH = [at.GetSymbol() for at in molH.GetAtoms()]
    # extend the sheet mask over the added hydrogens
    sh = np.full(len(symH), -1)
    sh[:len(sym)] = c['sheet']
    for at in molH.GetAtoms():
        if at.GetSymbol() == 'H':
            nb = at.GetNeighbors()
            if nb:
                sh[at.GetIdx()] = sh[nb[0].GetIdx()]
    return symH, XH, lab, Chem.GetFormalCharge(molH), sh
