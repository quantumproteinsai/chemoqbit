"""Capped Xaa-Pro dipeptides for the cis/trans isomerisation of a prolyl amide.

The conformational switch behind cis-pThr231-Pro tau: phosphorylation of the
residue before a proline changes how readily that amide flips, and Pin1
converts the cis form back to trans. The question here is what phosphorylation
alone does to the barrier and to the cis/trans equilibrium, with no enzyme.

Models are Ac-Xaa-Pro-NHMe, small enough for coupled-cluster single points, so
for once the absolute barrier is checkable against experiment (prolyl
isomerisation barriers are reported in the 14-24 kcal/mol range) rather than
only differences between models.

The coordinate is the amide torsion omega = CA(Xaa)-C(Xaa)-N(Pro)-CA(Pro):
180 deg trans, 0 deg cis, with the saddle near +-90 deg where the amide
pi bond is broken.

Phosphate protonation is a modelling choice, not a fact: the monoanion and the
dianion are both built, since a phosphomonoester's second pKa sits near
physiological pH.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Ac-Xaa-Pro-NHMe; L-Pro throughout, L-Thr is (2S,3R)
MODELS = {
    'ala':   ('CC(=O)N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)NC', 0),
    'thr':   ('CC(=O)N[C@@H]([C@@H](O)C)C(=O)N1CCC[C@H]1C(=O)NC', 0),
    'pthr1': ('CC(=O)N[C@@H]([C@@H](OP(=O)(O)[O-])C)C(=O)N1CCC[C@H]1C(=O)NC', -1),
    'pthr2': ('CC(=O)N[C@@H]([C@@H](OP(=O)([O-])[O-])C)C(=O)N1CCC[C@H]1C(=O)NC', -2),
    'ser':   ('CC(=O)N[C@@H](CO)C(=O)N1CCC[C@H]1C(=O)NC', 0),
    'pser1': ('CC(=O)N[C@@H](COP(=O)(O)[O-])C(=O)N1CCC[C@H]1C(=O)NC', -1),
    'pser2': ('CC(=O)N[C@@H](COP(=O)([O-])[O-])C(=O)N1CCC[C@H]1C(=O)NC', -2),
}
# the prolyl amide: a non-ring carbonyl carbon bonded to the ring nitrogen
_AMIDE = Chem.MolFromSmarts('[CX3;!R](=O)[NX3;R]')


def build(name, seed=0xC15):
    """(symbols, X, omega_atoms, charge) for one model, embedded in 3D.
    omega_atoms = (CA_Xaa, C_Xaa, N_Pro, CA_Pro)."""
    smiles, charge = MODELS[name]
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, ps) == 0, f'embedding failed for {name}'
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    m = mol.GetSubstructMatches(_AMIDE)
    assert len(m) == 1, f'{len(m)} prolyl amides matched in {name}'
    c_xaa, _o, n_pro = m[0]
    # CA(Xaa): the carbon neighbour of the carbonyl carbon
    ca_xaa = next(a.GetIdx() for a in mol.GetAtomWithIdx(c_xaa).GetNeighbors()
                  if a.GetSymbol() == 'C' and a.GetIdx() != _o)
    # CA(Pro): the ring carbon on N that carries the C-terminal carbonyl
    ca_pro = next(a.GetIdx() for a in mol.GetAtomWithIdx(n_pro).GetNeighbors()
                  if a.GetSymbol() == 'C' and a.IsInRing()
                  and any(nb.GetSymbol() == 'C' and not nb.IsInRing()
                          and any(x.GetSymbol() == 'O' for x in nb.GetNeighbors())
                          for nb in a.GetNeighbors()))
    conf = mol.GetConformer()
    X = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    return sym, X, (ca_xaa, c_xaa, n_pro, ca_pro), charge


def torsion(X, atoms):
    """signed dihedral in degrees"""
    i, j, k, l = atoms
    b1, b2, b3 = X[j] - X[i], X[k] - X[j], X[l] - X[k]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    L = np.linalg.norm(b2)
    return float(np.degrees(np.arctan2(np.dot(np.cross(n1, n2), b2 / L), np.dot(n1, n2))))


def pyramidalisation(X, atoms, mol_neighbours):
    """How far the prolyl nitrogen is from planar, in degrees: 0 planar, ~60 sp3.
    mol_neighbours: the three heavy atoms bonded to N."""
    n = atoms[2]
    a, b, c = (X[i] - X[n] for i in mol_neighbours)
    nrm = np.cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm)
    d = (a + b + c) / 3 - X[n]
    return float(np.degrees(np.arcsin(abs(np.dot(d, nrm)) / max(np.linalg.norm(d), 1e-9))))
