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


def cell(pdb):
    """(cell vectors in the PDB Cartesian frame, space group) from CRYST1."""
    for l in open(pdb):
        if l.startswith('CRYST1'):
            a, b, c = float(l[6:15]), float(l[15:24]), float(l[24:33])
            al, be, ga = (np.radians(float(l[33 + 7 * i:40 + 7 * i])) for i in range(3))
            sg = l[55:66].strip()
            ca, cb, cg, sg_ = np.cos(al), np.cos(be), np.cos(ga), np.sin(ga)
            v = np.sqrt(1 - ca ** 2 - cb ** 2 - cg ** 2 + 2 * ca * cb * cg)
            return np.array([[a, 0, 0],
                             [b * cg, b * sg_, 0],
                             [c * cb, c * (ca - cb * cg) / sg_, c * v / sg_]]), sg
    raise ValueError('no CRYST1')


def symmetry_ops(pdb):
    """[(3x3 rotation, translation)] from REMARK 290 SMTRY, in Cartesian coordinates."""
    rows = [l for l in open(pdb) if l.startswith('REMARK 290   SMTRY')]
    ops, cur = [], []
    for l in rows:
        cur.append([float(v) for v in l.split()[-4:]])
        if len(cur) == 3:
            M = np.array(cur)
            ops.append((M[:, :3], M[:, 3]))
            cur = []
    return ops


def stacking_axis(V, lo=4.4, hi=5.1):
    """The cell vector that is the beta-strand repeat: a cell edge of 4.4-5.1 A."""
    lens = np.linalg.norm(V, axis=1)
    i = [k for k in range(3) if lo <= lens[k] <= hi]
    assert len(i) == 1, f'expected one cell edge in {lo}-{hi} A, got lengths {lens.round(2)}'
    return V[i[0]], i[0]


def mating_sheet(pdb, chain='A', cutoff=4.5):
    """Find the symmetry image that forms the dry zipper interface with `chain`.

    A steric zipper is dry and side-chain mediated, so the partner is chosen by
    the number of SIDE-CHAIN to SIDE-CHAIN contacts, and an image whose
    backbones come within 4 A is rejected: that is a hydrogen-bonded extension
    of the same sheet, not a mating sheet. Images related to the original by a
    translation along the stacking axis alone are the in-sheet neighbours.
    Returns (rotation, translation, side-chain contacts, interface residues).
    """
    at = _read(pdb, chain)
    keys = list(at)
    X = np.array([at[k][0] for k in keys])
    bb = np.array([k[2] in ('N', 'CA', 'C', 'O', 'OXT') for k in keys])
    side = ~bb
    resid = [f'{k[0]}{k[1]}' for k in keys]
    V, _sg = cell(pdb)
    axis, ai = stacking_axis(V)
    best = None
    for R, t in symmetry_ops(pdb):
        for na in range(-2, 3):
            for nb in range(-2, 3):
                for nc in range(-2, 3):
                    n_cell = (na, nb, nc)
                    shift = na * V[0] + nb * V[1] + nc * V[2]
                    Y = X @ R.T + t + shift
                    if (np.allclose(R, np.eye(3)) and np.linalg.norm(t) < 1e-6
                            and all(v == 0 for k, v in enumerate(n_cell) if k != ai)):
                        # a translation along the stacking axis alone (or none): that is
                        # the hydrogen-bonded neighbour inside the same sheet, not a mate
                        continue
                    d = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
                    if d.min() < 2.0:                 # clash: not a real packing partner
                        continue
                    if d[np.ix_(bb, bb)].min() < 4.0:  # same sheet, hydrogen bonded
                        continue
                    n = int((d[np.ix_(side, side)] < cutoff).sum())
                    if n and (best is None or n > best[2]):
                        face = sorted({resid[i] for i in np.flatnonzero(side)
                                       if d[i][side].min() < cutoff},
                                      key=lambda r: int(''.join(c for c in r if c.isdigit())))
                        best = (R, t + shift, n, face)
    assert best is not None, 'no mating image found'
    return best


def build(pdb, chain='A', mate=None, neighbours=0, to_alanine=(), substitutions=None):
    """Heavy atoms of the two mating sheets.

    mate:        (R, t) of the symmetry image forming the interface; found
                 automatically when None, or given as a deposited chain id.
    neighbours:  strands added either side along the stacking axis.
    to_alanine:  residue numbers truncated to CB, in the central strand of both
                 sheets. substitutions: {resnum: 'ALA'} style relabelling is NOT
                 done here -- only truncation, which needs no new atoms.
    """
    V, _sg = cell(pdb)
    axis, _ai = stacking_axis(V)
    at = _read(pdb, chain)
    res = sorted({(rn, num) for rn, num, _ in at})
    if mate is None:
        R, t, _n, _face = mating_sheet(pdb, chain)
        sheets = [(np.eye(3), np.zeros(3), chain), (R, t, chain)]
    elif isinstance(mate, str):
        sheets = [(np.eye(3), np.zeros(3), chain), (np.eye(3), np.zeros(3), mate)]
    else:
        R, t = mate
        sheets = [(np.eye(3), np.zeros(3), chain), (R, t, chain)]

    sym, X, label, sheet = [], [], [], []
    n_lys = 0
    for s_, (R, t, ch) in enumerate(sheets):
        a_ch = _read(pdb, ch)
        rs = sorted({(rn, num) for rn, num, _ in a_ch})
        for k in range(-neighbours, neighbours + 1):
            for rn, num in rs:
                keep = None if num not in to_alanine else ['N', 'CA', 'C', 'O', 'CB', 'OXT']
                if rn == 'LYS' and (num not in to_alanine):
                    n_lys += 1
                for (r2, n2, nm), (x, el) in a_ch.items():
                    if (r2, n2) != (rn, num) or (keep is not None and nm not in keep):
                        continue
                    sym.append(el)
                    X.append((x + k * axis) @ R.T + t)
                    label.append(f'{ch}{s_}{k:+d}:{rn}{num}:{nm}')
                    sheet.append(s_)
    n_strands = len(sheets) * (2 * neighbours + 1)
    charge = n_strands * (NTERM + CTERM) + n_lys
    return dict(symbols=sym, X=np.array(X), labels=label, charge=charge,
                sheet=np.array(sheet), n_residues=len(res), n_strands=n_strands)


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
