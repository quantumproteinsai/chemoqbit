"""Reduced active-site model for the thymidylate synthase hydride transfer.

Acceptor  A : 5-methylene-6-(methylthio)-5,6-dihydrouracil (exocyclic methylene
              intermediate; S-methyl stands in for Cys146, N1-H for the sugar)
Donor     D : N-methylisopropylamine (C6-H of tetrahydrofolate: a tertiary C-H
              alpha to the secondary amine N5)
Arg166    G : guanidinium, H-bonded to the sulfur

Reaction (concerted, asynchronous; Kanaan 2009, Islam 2014):
    D-H + CH2=C5(U)-C6H(SMe)  ->  D(+)=  +  CH3-C5(U)=C6H  +  MeS(-)
i.e. hydride to C7, C6-S cleavage, thymine formed, thiolate leaves to Arg.

Atom indices of the key atoms are returned so scans and analyses never rely
on hard-coded numbers.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

SMILES = dict(A='C=C1C(SC)NC(=O)NC1=O', D='CC(C)NC', G='NC(N)=[NH2+]')


def _embed(smi, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return m, m.GetConformer().GetPositions()


def _unit(v):
    return v / np.linalg.norm(v)


def _rot_to(a, b):
    """rotation matrix taking unit vector a onto unit vector b"""
    a, b = _unit(a), _unit(b)
    v, c = np.cross(a, b), a @ b
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def build(face='anti', d_hc=2.7, seed=1):
    mA, XA = _embed(SMILES['A'], seed)
    mD, XD = _embed(SMILES['D'], seed)
    mG, XG = _embed(SMILES['G'], seed)
    sA = [a.GetSymbol() for a in mA.GetAtoms()]
    # acceptor key atoms
    C7 = 0                                   # exocyclic CH2 carbon (first in SMILES)
    C5 = 1
    C6 = 2
    S = 3
    nb = lambda m, i: [n.GetIdx() for n in m.GetAtomWithIdx(i).GetNeighbors()]
    H7 = [j for j in nb(mA, C7) if sA[j] == 'H']
    # p-orbital axis at C7 = normal of the plane (C5, C7, H7a, H7b)
    nrm = _unit(np.cross(XA[H7[0]] - XA[C7], XA[H7[1]] - XA[C7]))
    s_side = np.sign((XA[S] - XA[C6]) @ nrm)
    face_dir = -s_side * nrm if face == 'anti' else s_side * nrm
    # donor: tertiary C-H (C bonded to N and two methyls)
    sD = [a.GetSymbol() for a in mD.GetAtoms()]
    Cd = 1
    Hd = [j for j in nb(mD, Cd) if sD[j] == 'H'][0]
    # orient donor so its C-H points at C7, H at d_hc from C7
    R = _rot_to(XD[Hd] - XD[Cd], -face_dir)
    XD = (XD - XD[Hd]) @ R.T + (XA[C7] + d_hc * face_dir)
    # guanidinium: central C 3.6 A beyond S along C6->S, an N-H pointing at S
    u = _unit(XA[S] - XA[C6])
    XG = XG - XG[1] + XA[S] + 3.6 * u
    # assemble
    sym = sA + sD + [a.GetSymbol() for a in mG.GetAtoms()]
    X = np.vstack([XA, XD, XG])
    nA, nD = len(sA), len(sD)
    key = dict(C7=C7, C5=C5, C6=C6, S=S, Cd=nA + Cd, Hd=nA + Hd,
               Nd=nA + 3, Cg=nA + nD + 1)
    charge = +1                               # guanidinium
    return sym, X, key, charge


def to_xyz(sym, X, comment=''):
    return f'{len(sym)}\n{comment}\n' + '\n'.join(
        f'{s} {x:.6f} {y:.6f} {z:.6f}' for s, (x, y, z) in zip(sym, X)) + '\n'
