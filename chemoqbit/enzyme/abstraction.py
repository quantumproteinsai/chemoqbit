"""Cluster model of the C5 proton abstraction in thymidylate synthase.

The step before the hydride transfer of the companion work: in the covalent
ternary complex (Cys146-S bonded to C6, folate N5-CH2 bonded to C5) the C5
proton is removed and the Cys146 thiolate eliminated, in a concerted manner
(Kholodar et al., Biochemistry 60, 1243 (2021)), restoring C5=C6 and giving
the noncovalent bisubstrate intermediate.

PDB 1TLS is the crystal structure of this step's reactant: unlike the hydride
transfer, nothing has to be edited into a reactive state. Only the C5
substituent changes between the two models:

    c5='H'   dUMP:  the proton that catalysis removes
    c5='F'   FdUMP: 5-fluorouracil's metabolite, which has no C5 proton

No single residue is the general base: the acceptor is a water in a hydrogen-
bond network with Tyr94, Glu58, His147 and Asn177 (Ghosh et al., Phys. Chem.
Chem. Phys. 17, 30867 (2015); Kholodar et al. 2021). The cluster of the
hydride-transfer work holds Tyr94, His147, Asn177 and waters 387/327/453 but
not Glu58, the only carboxylate in that network, so Glu58 is added here. In
1TLS the relay runs

    C5 -> HOH387 (3.47 A) -> Tyr94 OH (2.65) -> His147 NE2 (3.59)
       -> HOH327 (2.66) -> Glu58 OE2 (2.94)

with Tyr94 OH itself 3.10 A from C5. ACCEPTORS lists the candidates tested.

Charge 0 overall: Arg166 guanidinium +1, Glu58 carboxylate -1. The concerted
product is neutral too (aromatic pyrimidine, Cys146 thiolate -1, protonated
base +1), so no charge is created along the path.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from .cluster import CH, CUTS, FRAGMENTS, WATERS, read_chain

# Glu58, the carboxylate terminating the proton relay, cut at CG-CB
GLU58 = ('glu58', 'GLU', 58, ['OE1', 'OE2', 'CD', 'CG'])
GLU58_CUT = ('GLU', 58, 'CG', 'CB')

# candidate proton acceptors, as (label, atom that takes the proton)
ACCEPTORS = {'wat387': 'wat387:O', 'tyr94': 'tyr94:OH', 'wat327': 'wat327:O',
             'glu58': 'glu58:OE2'}


def build(pdb='data/1TLS.pdb', chain='A', c5='H'):
    """Heavy atoms of the covalent ternary complex plus Glu58.
    c5: 'H' (dUMP, the proton added later with the other hydrogens) or 'F' (FdUMP).
    Returns (symbols, X, labels, frozen)."""
    at = read_chain(pdb, chain)
    sym, X, label, frozen = [], [], [], []

    def add(el, x, lab, fixed=False):
        sym.append(el); X.append(np.asarray(x, float)); label.append(lab); frozen.append(fixed)

    for name, res, num, atoms in FRAGMENTS + [GLU58]:
        for a in atoms:
            x, el = at[(res, num, a)]
            add(el, x, f'{name}:{a}')
    if c5 == 'F':
        x, el = at[('UFP', 265, 'F5')]
        add('F', x, 'nucleotide:F5')
    for w in WATERS:
        x, el = at[('HOH', w, 'O')]
        add('O', x, f'wat{w}:O')
    _RES = {n: (r, k) for n, r, k, _ in FRAGMENTS + [GLU58]}
    for res, num, keep, gone in CUTS + [GLU58_CUT]:
        lab = next((l.split(':')[0] for l in label
                    if l.endswith(':' + keep) and _RES.get(l.split(':')[0]) == (res, num)), None)
        if lab is None:
            continue
        a, b = at[(res, num, keep)][0], at[(res, num, gone)][0]
        add('H', a + CH * (b - a) / np.linalg.norm(b - a), f'{lab}:cap{gone}', True)
    return sym, np.array(X), label, frozen


def add_hydrogens(sym, X, label, charge=0):
    """Bonds, formal charges and hydrogens for the covalent ternary complex.

    Differs from the hydride-transfer model: N5-C11 and C5-C11 are kept, C5 and
    C6 are sp3 (so C5 receives its proton here unless it carries F), the
    pyrimidine is not aromatic, and Glu58 is a carboxylate."""
    m = Chem.RWMol()
    conf = Chem.Conformer(len(sym))
    for i, (s, x) in enumerate(zip(sym, X)):
        m.AddAtom(Chem.Atom(s))
        conf.SetAtomPosition(i, x.tolist())
    m.AddConformer(conf)
    mol = m.GetMol()
    rdDetermineBonds.DetermineConnectivity(mol, useHueckel=False, charge=charge)
    em = Chem.RWMol(mol)
    idx = {l: i for i, l in enumerate(label)}

    # the adduct's own bonds: Cys146 S-C6, C5-C11, C11-N5 are all single and
    # must exist; DetermineConnectivity finds them from the crystal distances
    for a, b in (('cys146:SG', 'nucleotide:C6'), ('nucleotide:C5', 'methylene:C11'),
                 ('methylene:C11', 'folate:N5')):
        bd = em.GetBondBetweenAtoms(idx[a], idx[b])
        assert bd is not None, f'{a}-{b} not perceived'
        bd.SetBondType(Chem.BondType.SINGLE)

    # guanidinium
    em.GetBondBetweenAtoms(idx['arg166:CZ'], idx['arg166:NH2']).SetBondType(Chem.BondType.DOUBLE)
    em.GetAtomWithIdx(idx['arg166:NH2']).SetFormalCharge(1)
    # Glu58 carboxylate: C=OE1, OE2(-)
    em.GetBondBetweenAtoms(idx['glu58:CD'], idx['glu58:OE1']).SetBondType(Chem.BondType.DOUBLE)
    em.GetBondBetweenAtoms(idx['glu58:CD'], idx['glu58:OE2']).SetBondType(Chem.BondType.SINGLE)
    em.GetAtomWithIdx(idx['glu58:OE2']).SetFormalCharge(-1)
    # carbonyls
    for c, o in (('nucleotide:C2', 'nucleotide:O2'), ('nucleotide:C4', 'nucleotide:O4'),
                 ('folate:C4', 'folate:O4'), ('asn177:CG', 'asn177:OD1')):
        em.GetBondBetweenAtoms(idx[c], idx[o]).SetBondType(Chem.BondType.DOUBLE)
    # pterin
    em.GetBondBetweenAtoms(idx['folate:C4A'], idx['folate:C8A']).SetBondType(Chem.BondType.DOUBLE)
    em.GetBondBetweenAtoms(idx['folate:C2'], idx['folate:N3']).SetBondType(Chem.BondType.SINGLE)
    em.GetBondBetweenAtoms(idx['folate:C2'], idx['folate:N1']).SetBondType(Chem.BondType.DOUBLE)
    # Tyr94 phenol, His147 imidazole (NE2-H tautomer)
    for ring, atoms in (('tyr94', ['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2']),
                        ('his147', ['CG', 'ND1', 'CE1', 'NE2', 'CD2'])):
        keys = [f'{ring}:{a}' for a in atoms]
        for p_, q_ in zip(keys, keys[1:] + keys[:1]):
            b = em.GetBondBetweenAtoms(idx[p_], idx[q_])
            b.SetBondType(Chem.BondType.AROMATIC); b.SetIsAromatic(True)
        for k in keys:
            em.GetAtomWithIdx(idx[k]).SetIsAromatic(True)
    for a, n in (('his147:ND1', 0), ('his147:NE2', 1)):
        em.GetAtomWithIdx(idx[a]).SetNumExplicitHs(n)
        em.GetAtomWithIdx(idx[a]).SetNoImplicit(True)

    keep_noimplicit = {idx['his147:ND1'], idx['his147:NE2']}
    for a in em.GetAtoms():
        if a.GetIdx() in keep_noimplicit:
            continue
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    mol = em.GetMol()
    Chem.SanitizeMol(mol)
    molH = Chem.AddHs(mol, addCoords=True)
    Chem.SanitizeMol(molH)
    return molH


def c5_hydrogen(molH, label):
    """index of the hydrogen on C5 (the proton catalysis removes); None if C5 carries F."""
    c5 = label.index('nucleotide:C5')
    for nb in molH.GetAtomWithIdx(c5).GetNeighbors():
        if nb.GetSymbol() == 'H':
            return nb.GetIdx()
    return None
