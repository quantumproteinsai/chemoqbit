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
# Crystallisation and purification additives: present in the crystal, not in the
# biology. 5WMJ carries a trifluoroacetate, whose own OXT was being charged as a
# peptide C-terminus and left the model one unit short.
ADDITIVES = {'TFA', 'GOL', 'EDO', 'PEG', 'PG4', 'MPD', 'SO4', 'PO4', 'ACT', 'ACY', 'CIT',
             'DMS', 'IMD', 'TRS', 'EPE', 'MES', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'NO3',
             'FMT', 'IPA', 'BME', 'DTT'}


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
    """The cell vector that is the beta-strand repeat, about 4.8 A.

    An antiparallel sheet repeats by a 2-fold screw, so its cell edge holds TWO
    strands and comes out near 9.5 A: SOD1 30-35 (5WMJ) is such a case. Half of
    that edge is then the strand-to-strand spacing, even though the operation
    relating neighbours is a screw rather than a pure translation.
    """
    lens = np.linalg.norm(V, axis=1)
    i = [k for k in range(3) if lo <= lens[k] <= hi]
    if len(i) == 1:
        return V[i[0]], i[0]
    j = [k for k in range(3) if 2 * lo <= lens[k] <= 2 * hi]
    if len(j) == 1:
        return V[j[0]] / 2.0, j[0]
    raise AssertionError(f'no cell edge near {lo}-{hi} A or twice it; lengths {lens.round(2)}')


BACKBONE = {'N', 'CA', 'C', 'O', 'OXT'}


def mating_sheet(pdb, chain='A', probe_waters=4.0):
    """Find the sheet that forms the steric zipper with `chain`.

    Sawaya et al. (Nature 447, 453 (2007)) define a steric zipper as a pair of
    sheets whose facing side chains interdigitate in a dry interface, so the
    partner is chosen by the area it BURIES, and by how much of that area is
    side chain -- not by counting close contacts, which scores an edge-on
    packing contact just as highly. Images related to the original by a
    translation along the stacking axis alone are strands of the same sheet.

    Returns (rotation, translation, descriptors) with the buried area in A^2,
    the side-chain fraction of it, and the number of waters within
    `probe_waters` of both sheets.
    """
    from .sasa import buried, buried_per_atom
    at = _read(pdb, chain)
    keys = list(at)
    X = np.array([at[k][0] for k in keys])
    el = [at[k][1] for k in keys]
    side = np.array([k[2] not in BACKBONE for k in keys])
    W = np.array([v[0] for k, v in _read(pdb, chain).items() if k[0] == 'HOH'] or [])
    if W.size == 0:
        W = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(pdb)
                      if l.startswith(('ATOM', 'HETATM')) and l[17:20].strip() == 'HOH'] or []).reshape(-1, 3)
    V, _sg = cell(pdb)
    _axis, ai = stacking_axis(V)
    best = None
    for ci, (R, t) in enumerate(symmetry_ops(pdb)):
        for na in range(-1, 2):
            for nb in range(-1, 2):
                for nc in range(-1, 2):
                    nn = (na, nb, nc)
                    if ci == 0 and all(v == 0 for q, v in enumerate(nn) if q != ai):
                        continue                      # same sheet, or itself
                    shift = na * V[0] + nb * V[1] + nc * V[2]
                    Y = X @ R.T + t + shift
                    d = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
                    if d.min() < 2.2 or d.min() > 5.0:
                        continue
                    tot = buried(el, X, el, Y)
                    pa = buried_per_atom(el, X, el, Y)
                    frac = float(pa[side].sum() / max(pa.sum(), 1e-9))
                    wb = 0
                    if W.size:
                        d1 = np.linalg.norm(W[:, None, :] - X[None, :, :], axis=2).min(axis=1)
                        d2 = np.linalg.norm(W[:, None, :] - Y[None, :, :], axis=2).min(axis=1)
                        wb = int(((d1 < probe_waters) & (d2 < probe_waters)).sum())
                    desc = dict(buried=float(tot), side_fraction=frac, interfacial_waters=wb,
                                op=ci + 1, cell=nn)
                    if best is None or tot > best[2]['buried']:
                        best = (R, t + shift, desc)
    assert best is not None, 'no packing partner within 5 A'
    return best


def interface_of(pdb, chain, Y):
    """Buried area, side-chain fraction and interfacial waters for an explicit partner."""
    from .sasa import buried, buried_per_atom
    at = _read(pdb, chain)
    keys = list(at)
    X = np.array([at[k][0] for k in keys]); el = [at[k][1] for k in keys]
    side = np.array([k[2] not in BACKBONE for k in keys])
    tot = buried(el, X, el, Y)
    pa = buried_per_atom(el, X, el, Y)
    return dict(buried=float(tot), side_fraction=float(pa[side].sum() / max(pa.sum(), 1e-9)))


def build(pdb, chain='A', mate=None, neighbours=0, to_alanine=(), substitutions=None,
          keep_residues=None):
    """Heavy atoms of the two mating sheets.

    mate:        (R, t) of the symmetry image forming the interface; found
                 automatically when None, or given as a deposited chain id.
    neighbours:  strands added either side along the stacking axis.
    to_alanine:  residue numbers truncated to CB, in the central strand of both
                 sheets. substitutions: {resnum: 'ALA'} style relabelling is NOT
                 done here -- only truncation, which needs no new atoms.
    keep_residues: residue numbers to keep, dropping the rest whole. Waters have
                 their own numbers and must be listed to survive. The charge
                 below is n_strands.(NTERM + CTERM) + n_lys, which assumes a free
                 amine at the lowest kept residue and a free carboxylate at the
                 highest; protonate() puts NH3+ on the lowest number it is given,
                 so the amine end takes care of itself, but OXT exists only where
                 the deposited peptide ended. A window that stops short of the
                 deposited C-terminus therefore has no carboxylate, the CTERM
                 term in the charge is then wrong, and the cut carbonyl would be
                 capped by rdkit as an aldehyde. Callers must end the window at
                 the deposited C-terminus; cut_c_terminus below says whether they
                 did, so the caller can refuse rather than compute a model whose
                 charge is a guess.
    """
    V, _sg = cell(pdb)
    axis, _ai = stacking_axis(V)
    at = _read(pdb, chain)
    res = sorted({(rn, num) for rn, num, _ in at if rn not in ADDITIVES})
    if mate is None:
        R, t, _desc = mating_sheet(pdb, chain)
        sheets = [(np.eye(3), np.zeros(3), chain), (R, t, chain)]
    elif isinstance(mate, str):
        sheets = [(np.eye(3), np.zeros(3), chain), (np.eye(3), np.zeros(3), mate)]
    else:
        R, t = mate
        sheets = [(np.eye(3), np.zeros(3), chain), (R, t, chain)]
    # A symmetry image carries the same atoms of the same chain in a rotated frame,
    # so the two sheets are the same fragment and every per-sheet quantity must
    # match. A second deposited chain carries no such promise: 5WMJ chain B has two
    # more ordered waters than chain A, so there the sheets legitimately differ.
    equivalent = not isinstance(mate, str)
    mate_transform = (sheets[1][0], sheets[1][1])

    sym, X, label, sheet = [], [], [], []
    n_lys = 0
    for s_, (R, t, ch) in enumerate(sheets):
        a_ch = _read(pdb, ch)
        rs = sorted({(rn, num) for rn, num, _ in a_ch if rn not in ADDITIVES})
        if keep_residues is not None:
            rs = [(rn, num) for rn, num in rs if num in set(keep_residues)]
            if not rs:
                raise ValueError(f'keep_residues {sorted(keep_residues)} kept no residue '
                                 f'of chain {ch}')
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
    cut_c = (None if keep_residues is None
             else not any(l.rsplit(':', 1)[1] == 'OXT' for l in label))
    return dict(symbols=sym, X=np.array(X), labels=label, charge=charge,
                sheet=np.array(sheet), n_residues=len(res), n_strands=n_strands,
                cut_c_terminus=cut_c, sheets_equivalent=equivalent,
                mate_transform=mate_transform)


def protonate(c, seed=None):
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
    parent = {}
    for at in molH.GetAtoms():
        if at.GetSymbol() == 'H':
            nb = at.GetNeighbors()
            if nb:
                sh[at.GetIdx()] = sh[nb[0].GetIdx()]
                parent[at.GetIdx()] = nb[0].GetIdx()
    hmap = {}
    for h, q_ in sorted(parent.items()):
        hmap.setdefault(lab[q_], []).append(h)
    if seed:
        XH = _seed_hydrogens(XH, hmap, seed)
    if c.get('sheets_equivalent'):
        XH = _image_hydrogens(XH, sh, parent, c)
    return symH, XH, lab, Chem.GetFormalCharge(molH), sh, hmap


def _seed_hydrogens(XH, hmap, seed):
    """Start each hydrogen from a relaxed position carried in from another model.

    An alanine ddE is a difference of two interaction energies, so it only means
    "what that side chain contributes" if everything else about the two models is
    the same. It was not: every mutant got its own AddHs guess and its own local
    optimise_frozen, so the wild type and each mutant settled into a DIFFERENT
    local minimum of the same hydrogen-bond network -- three waters, two of them
    interfacial, plus the backbone and the Gln7 amide. The interaction energy is a
    difference of numbers two hundred times its own size, so a basin change at the
    interface passes into ddE whole, and two of 4NIP's four came out at -31 and
    -28 kcal/mol for the loss of a valine and a glutamine side chain.

    Seeding every surviving hydrogen from the wild type's relaxed geometry puts all
    the models in one basin, leaving the mutation as the only difference. A parent
    whose hydrogen count the mutation changed -- CB goes from one hydrogen to three
    when a side chain is truncated to alanine -- keeps its own guess, since there is
    no wild-type position for the two new ones.
    """
    for l, idx in hmap.items():
        x = seed.get(l)
        if x is None or len(x) != len(idx):
            continue
        for i, xi in zip(idx, x):
            XH[i] = xi
    return XH


def _image_hydrogens(XH, sh, parent, c):
    """Replace sheet 1's hydrogens with the rigid image of sheet 0's.

    rdkit's AddHs places a hydrogen from its parent's own local frame. That fixes
    it at a tetrahedral or planar centre, but a free rotor -- a water, a hydroxyl,
    NH3+, an amide -- is left on whichever rotamer that frame happens to pick, and
    the mating sheet's frame is rotated. The two sheets of 4NIP came out 0.499 A
    apart in their hydrogens while their heavy atoms agreed to 0.000000 A, and
    E_sheet0 - E_sheet1, which symmetry fixes at zero, was 36 kcal/mol for the wild
    type and up to 95 for a mutant. Waters are the worst case because an isolated
    oxygen has no heavy-atom frame to orient against at all.

    That asymmetry is the detector rather than the error itself: interaction()
    slices one coordinate set for the complex and for both fragments, so a static
    misplacement cancels in E(both) - E(sheet0) - E(sheet1). What does not cancel
    is WHICH local minimum of the hydrogen-bond network each structure fell into,
    because optimise_frozen relaxes locally and the wild type and every mutant get
    an independent guess. Placing the hydrogens symmetrically removes one arbitrary
    degree of freedom and makes the model match its own symmetry; it does not on
    its own put the wild type and its mutants in a COMMON basin, which is what an
    alanine ddE needs. sheet_equivalence records the residual either way.
    """
    R, t = c['mate_transform']
    n = len(c['symbols']) // 2
    image = lambda x: x @ R.T + t
    off = np.abs(image(XH[:n]) - XH[n:2 * n]).max()
    if off > 1e-4:
        raise ValueError(f'sheet 1 is not the image of sheet 0 under the mate '
                         f'transform ({off:.2e} A); the index correspondence this '
                         f'relies on no longer holds')
    by_parent = ({}, {})
    for h, p in sorted(parent.items()):
        by_parent[p >= n][p % n] = by_parent[p >= n].get(p % n, []) + [h]
    for p, hs0 in by_parent[0].items():
        hs1 = by_parent[1].get(p, [])
        if len(hs0) != len(hs1):
            raise ValueError(f'heavy atom {p} carries {len(hs0)} hydrogens in sheet 0 '
                             f'but {len(hs1)} in its symmetry image')
        for a, b in zip(hs0, hs1):
            XH[b] = image(XH[a])
    return XH
