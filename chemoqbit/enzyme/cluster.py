"""Active-site cluster model of thymidylate synthase from PDB 1TLS (chain A).

1TLS is the trapped covalent ternary complex: Cys146-S bonded to C6 of FdUMP,
and C5 of FdUMP bonded to the folate methylene C11. That is the dead end
FdUMP creates. To reach the step that FOLLOWS in normal catalysis -- the
hydride transfer from H4folate C6 to the exocyclic methylene -- the crystal
adduct is edited:

    * F5 is deleted (5-FU's fluorine; the natural substrate has H there,
      and after C5-H abstraction C5 carries the exocyclic methylene)
    * the N5-C11 bond is cut (the folate has already eliminated), leaving
      C5=C11 as the exocyclic methylene on the nucleotide
    * the folate keeps its C6-H, the hydride donor

Fragments are truncated at the bonds listed in CUTS and capped with hydrogen
along the original bond vector. Cap hydrogens (and, optionally, all heavy
atoms) are frozen in later optimisations so the cluster keeps the protein's
geometry.

Charge +1 overall: guanidinium of Arg166, everything else neutral.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

CH = 1.09          # cap C-H
NH = 1.01
OH = 0.97

# fragment -> (residue name, number, atoms to keep)
FRAGMENTS = [
    ('nucleotide', 'UFP', 265, ['N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'O2', 'O4']),
    ('methylene', 'C2F', 266, ['C11']),
    ('folate', 'C2F', 266, ['N5', 'C6', 'C7', 'N8', 'C4A', 'C8A', 'C4', 'O4',
                            'N3', 'C2', 'NA2', 'N1']),
    ('cys146', 'CYS', 146, ['SG', 'CB']),
    ('arg166', 'ARG', 166, ['NE', 'CZ', 'NH1', 'NH2', 'CD']),
    ('tyr94', 'TYR', 94, ['OH', 'CZ', 'CE1', 'CE2', 'CD1', 'CD2', 'CG']),
    ('asn177', 'ASN', 177, ['CG', 'OD1', 'ND2', 'CB']),
    ('his147', 'HIS', 147, ['CG', 'ND1', 'CD2', 'CE1', 'NE2', 'CB']),
]
WATERS = [387, 327, 453]
# (residue, number, kept atom, removed neighbour) -> cap H on the kept atom
CUTS = [('UFP', 265, 'N1', "C1'"), ('C2F', 266, 'C6', 'C9'), ('CYS', 146, 'CB', 'CA'),
        ('ARG', 166, 'CD', 'CG'), ('TYR', 94, 'CG', 'CB'), ('ASN', 177, 'CB', 'CA'),
        ('HIS', 147, 'CB', 'CA')]


def read_chain(pdb, chain='A'):
    out = {}
    for l in Path(pdb).read_text().splitlines():
        if l.startswith(('ATOM', 'HETATM')) and l[21] == chain:
            key = (l[17:20].strip(), int(l[22:26]), l[12:16].strip())
            out[key] = (np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]),
                        (l[76:78].strip() or l[12:16].strip()[0]).capitalize())
    return out


def build(pdb='data/1TLS.pdb', chain='A', waters=True, with_his=True, with_tyr=True):
    at = read_chain(pdb, chain)
    sym, X, label, frozen = [], [], [], []

    def add(el, x, lab, fixed=False):
        sym.append(el); X.append(np.asarray(x, float)); label.append(lab); frozen.append(fixed)

    for name, res, num, atoms in FRAGMENTS:
        if name == 'his147' and not with_his:
            continue
        if name == 'tyr94' and not with_tyr:
            continue
        for a in atoms:
            x, el = at[(res, num, a)]
            add(el, x, f'{name}:{a}')
    if waters:
        for w in WATERS:
            x, el = at[('HOH', w, 'O')]
            add('O', x, f'wat{w}:O')
    # cap hydrogens along the cut bonds
    for res, num, keep, gone in CUTS:
        lab = next((l.split(':')[0] for l in label
                    if l.endswith(':' + keep) and _res_of(l) == (res, num)), None)
        if lab is None:
            continue
        a = at[(res, num, keep)][0]
        b = at[(res, num, gone)][0]
        add('H', a + CH * (b - a) / np.linalg.norm(b - a), f'{lab}:cap{gone}', True)
    return sym, np.array(X), label, frozen


_RESOF = {name: (res, num) for name, res, num, _ in FRAGMENTS}


def _res_of(lab):
    return _RESOF.get(lab.split(':')[0], (None, None))


def add_hydrogens(sym, X, label, charge=1, his_protonated=False, o4_protonated=False):
    """Infer bonds, set formal charges, add missing hydrogens with coordinates.
    his_protonated: His147 as imidazolium (ND1-H and NE2-H, +1) instead of the
    neutral NE2-H tautomer. o4_protonated: O4 of the nucleotide protonated
    (C4=O4H+, +1), the strongest stabilisation of the enolate formed on hydride
    transfer."""
    m = Chem.RWMol()
    conf = Chem.Conformer(len(sym))
    for i, (s, x) in enumerate(zip(sym, X)):
        m.AddAtom(Chem.Atom(s))
        conf.SetAtomPosition(i, x.tolist())
    m.AddConformer(conf)
    mol = m.GetMol()
    rdDetermineBonds.DetermineConnectivity(mol, useHueckel=False, charge=charge)
    em = Chem.RWMol(mol)
    # break the bonds the mechanism requires to be absent
    idx = {l: i for i, l in enumerate(label)}
    for a, b in (('folate:N5', 'methylene:C11'),):
        if a in idx and b in idx and em.GetBondBetweenAtoms(idx[a], idx[b]):
            em.RemoveBond(idx[a], idx[b])
    # exocyclic methylene: C5=C11
    bd = em.GetBondBetweenAtoms(idx['nucleotide:C5'], idx['methylene:C11'])
    if bd:
        bd.SetBondType(Chem.BondType.DOUBLE)
    # guanidinium: CZ=NH2(+)
    if 'arg166:CZ' in idx:
        em.GetBondBetweenAtoms(idx['arg166:CZ'], idx['arg166:NH2']).SetBondType(Chem.BondType.DOUBLE)
        em.GetAtomWithIdx(idx['arg166:NH2']).SetFormalCharge(1)
    # carbonyls and aromatic/conjugated bonds of the two rings
    for c, o in (('nucleotide:C2', 'nucleotide:O2'), ('nucleotide:C4', 'nucleotide:O4'),
                 ('folate:C4', 'folate:O4'), ('asn177:CG', 'asn177:OD1')):
        if c in idx and o in idx:
            em.GetBondBetweenAtoms(idx[c], idx[o]).SetBondType(Chem.BondType.DOUBLE)
    if o4_protonated:
        # O+ with a double bond to C4: valence 3, so the fill below adds one H
        em.GetAtomWithIdx(idx['nucleotide:O4']).SetFormalCharge(1)
    # pterin: C4A=C8A (2-amino-4-oxo-tetrahydropteridine)
    if 'folate:C4A' in idx:
        em.GetBondBetweenAtoms(idx['folate:C4A'], idx['folate:C8A']).SetBondType(Chem.BondType.DOUBLE)
        em.GetBondBetweenAtoms(idx['folate:C2'], idx['folate:N3']).SetBondType(Chem.BondType.SINGLE)
        em.GetBondBetweenAtoms(idx['folate:C2'], idx['folate:N1']).SetBondType(Chem.BondType.DOUBLE)
    # Tyr94 phenol and His147 imidazole: restore aromatic rings
    for ring, atoms in (('tyr94', ['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2']),
                        ('his147', ['CG', 'ND1', 'CE1', 'NE2', 'CD2'])):
        keys = [f'{ring}:{a}' for a in atoms]
        if not all(k in idx for k in keys):
            continue
        for p_, q_ in zip(keys, keys[1:] + keys[:1]):
            b = em.GetBondBetweenAtoms(idx[p_], idx[q_])
            b.SetBondType(Chem.BondType.AROMATIC); b.SetIsAromatic(True)
        for k in keys:
            em.GetAtomWithIdx(idx[k]).SetIsAromatic(True)
        if ring == 'his147':
            # NE2-H tautomer: NE2 donates to O4 of the nucleotide (3.5 A)
            em.GetAtomWithIdx(idx['his147:ND1']).SetNumExplicitHs(1 if his_protonated else 0)
            if his_protonated:
                em.GetAtomWithIdx(idx['his147:ND1']).SetFormalCharge(1)
            em.GetAtomWithIdx(idx['his147:NE2']).SetNumExplicitHs(1)
            em.GetAtomWithIdx(idx['his147:NE2']).SetNoImplicit(True)
            em.GetAtomWithIdx(idx['his147:ND1']).SetNoImplicit(True)

    # DetermineConnectivity marks every atom NoImplicit; let valence fill with H
    for a in em.GetAtoms():
        if a.GetNoImplicit() and a.GetIdx() in (idx.get('his147:ND1'), idx.get('his147:NE2')):
            continue
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    mol = em.GetMol()
    Chem.SanitizeMol(mol)
    molH = Chem.AddHs(mol, addCoords=True)
    Chem.SanitizeMol(molH)
    return molH


def to_xyz(molH, comment=''):
    conf = molH.GetConformer()
    L = [f'{a.GetSymbol()} ' + ' '.join(f'{v:.6f}' for v in conf.GetAtomPosition(a.GetIdx()))
         for a in molH.GetAtoms()]
    return f'{molH.GetNumAtoms()}\n{comment}\n' + '\n'.join(L) + '\n'
