"""Cut a ligand-binding pocket out of a PDB entry as a cluster model.

Built for the cystic-fibrosis question: lumacaftor and tezacaftor bind a
hydrophobic pocket in CFTR's first transmembrane domain and stabilise it,
allosterically rescuing dF508 (Fiedorczuk & Chen, Cell 2022; PDB 7SVD, 7SVR).
Mutating pocket residues made dF508-CFTR insensitive to both drugs, so the
observable to compute is a DIFFERENCE: how much each pocket residue
contributes to holding the drug, by truncating it to alanine.

Side chains are kept from CB outward and capped with hydrogen along the
CA-CB bond; glycine contributes nothing and is dropped. Backbone is left out,
so this models side-chain contacts only -- stated plainly because a pocket
lined by backbone carbonyls would need a different cut.

An interaction energy is E(complex) - E(pocket) - E(ligand) at a common
geometry, which cancels the internal energies and leaves the contact.
"""
from __future__ import annotations

import numpy as np

CH = 1.09
# side-chain atoms kept per residue, from CB outward
SIDECHAIN = {
    'ALA': ['CB'], 'VAL': ['CB', 'CG1', 'CG2'], 'LEU': ['CB', 'CG', 'CD1', 'CD2'],
    'ILE': ['CB', 'CG1', 'CG2', 'CD1'], 'MET': ['CB', 'CG', 'SD', 'CE'],
    'PHE': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'TRP': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'SER': ['CB', 'OG'], 'THR': ['CB', 'OG1', 'CG2'], 'CYS': ['CB', 'SG'],
    'ASN': ['CB', 'CG', 'OD1', 'ND2'], 'GLN': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
    'ASP': ['CB', 'CG', 'OD1', 'OD2'], 'GLU': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
    'LYS': ['CB', 'CG', 'CD', 'CE', 'NZ'],
    'ARG': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'HIS': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'PRO': ['CB', 'CG', 'CD'], 'GLY': [],
}
FORMAL = {'LYS': 1, 'ARG': 1, 'ASP': -1, 'GLU': -1}


def read(pdb, chain='A'):
    """{(resname, resnum, atom): (xyz, element)} for one chain"""
    out = {}
    for l in open(pdb):
        if l.startswith(('ATOM', 'HETATM')) and l[21] == chain:
            if l[16] not in (' ', 'A'):               # skip alternate conformations
                continue
            el = l[76:78].strip() or l[12:16].strip()[0]
            out[(l[17:20].strip(), int(l[22:26]), l[12:16].strip())] = (
                np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]), el.capitalize())
    return out


def cut(pdb, ligand, chain='A', cutoff=4.5, waters=True, drop=(), to_alanine=()):
    """Pocket cluster around `ligand` (residue name).

    drop:        residue numbers to leave out entirely
    to_alanine:  residue numbers truncated to CB (the in-silico alanine mutation)
    Returns dict with symbols, X, labels, frozen, charge, and the index ranges
    of the ligand and of the pocket.
    """
    at = read(pdb, chain)
    lig = {k: v for k, v in at.items() if k[0] == ligand}
    assert lig, f'{ligand} not found in chain {chain}'
    L = np.array([v[0] for v in lig.values()])

    res = {}
    for (rn, num, a), (x, el) in at.items():
        if rn == ligand or rn == 'HOH' or num in drop:
            continue
        if rn not in SIDECHAIN:
            continue
        if np.min(np.linalg.norm(L - x, axis=1)) <= cutoff:
            res.setdefault((rn, num), set()).add(a)

    sym, X, label, frozen = [], [], [], []

    def add(el, x, lab, fixed=False):
        sym.append(el); X.append(np.asarray(x, float)); label.append(lab); frozen.append(fixed)

    for a, (x, el) in lig.items():
        add(el, x, f'lig:{a[2]}')
    n_lig = len(sym)

    charge = 0
    for (rn, num) in sorted(res, key=lambda k: k[1]):
        keep = ['CB'] if num in to_alanine else SIDECHAIN[rn]
        got = [a for a in keep if (rn, num, a) in at]
        if not got:
            continue
        for a in got:
            x, el = at[(rn, num, a)]
            add(el, x, f'{rn}{num}:{a}')
        if num not in to_alanine:
            charge += FORMAL.get(rn, 0)
        # cap the CA-CB bond with hydrogen on CB
        if (rn, num, 'CA') in at and (rn, num, 'CB') in at:
            cb = at[(rn, num, 'CB')][0]; ca = at[(rn, num, 'CA')][0]
            add('H', cb + CH * (ca - cb) / np.linalg.norm(ca - cb), f'{rn}{num}:capCA', True)

    if waters:
        for (rn, num, a), (x, el) in at.items():
            if rn == 'HOH' and np.min(np.linalg.norm(L - x, axis=1)) <= cutoff:
                add('O', x, f'wat{num}:O')

    return dict(symbols=sym, X=np.array(X), labels=label, frozen=frozen, charge=charge,
                n_ligand=n_lig, residues=sorted(f'{rn}{num}' for rn, num in res))


def split(c):
    """(ligand indices, pocket indices) of a cut pocket"""
    n = c['n_ligand']
    return list(range(n)), list(range(n, len(c['symbols'])))


def protonate(c, ligand_charge=0):
    """Add hydrogens to a cut pocket with RDKit, setting the formal charges the
    truncation implies: Lys NZ and Arg NH2 cationic, Asp/Glu carboxylates
    anionic. `ligand_charge` is a modelling choice the structure cannot settle --
    lumacaftor's benzoic acid is drawn neutral by default, since the pocket is
    buried and hydrophobic.
    Returns (symbols, X, labels_of_heavy_atoms, total_charge, molH)."""
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
    total = c['charge'] + ligand_charge
    rdDetermineBonds.DetermineConnectivity(mol, useHueckel=False, charge=total)
    em = Chem.RWMol(mol)
    idx = {l: i for i, l in enumerate(lab)}
    for l, i in idx.items():
        rn = l.split(':')[0]
        a = l.split(':')[1]
        if rn.startswith('LYS') and a == 'NZ':
            em.GetAtomWithIdx(i).SetFormalCharge(1)
        elif rn.startswith('ARG') and a == 'NH2':
            em.GetAtomWithIdx(i).SetFormalCharge(1)
            b = em.GetBondBetweenAtoms(i, idx[f'{rn}:CZ'])
            if b:
                b.SetBondType(Chem.BondType.DOUBLE)
        elif rn.startswith(('ASP', 'GLU')) and a in ('OD2', 'OE2'):
            em.GetAtomWithIdx(i).SetFormalCharge(-1)
    for a in em.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    mol = em.GetMol()
    Chem.SanitizeMol(mol)
    molH = Chem.AddHs(mol, addCoords=True)
    Chem.SanitizeMol(molH)
    confH = molH.GetConformer()
    XH = np.array([list(confH.GetAtomPosition(i)) for i in range(molH.GetNumAtoms())])
    symH = [a.GetSymbol() for a in molH.GetAtoms()]
    return symH, XH, lab, Chem.GetFormalCharge(molH), molH
