"""Chalcogen S_N2: why selenium catalyses disulfide formation where sulfur does not.

The accessibility and attack-geometry results closed the destructive question. The
Cys57-Cys146 disulfide cannot be reached in the folded protein: not from outside
(both sulfurs 0.00 A^2 exposed against 128.7 for a free sulfur) and not from inside
either, since Cys6 sits 8.37 A away at 133 deg and Cys111 18.36 A away at 167.8 deg.
A barrier for breaking it needs a destabilised ensemble that nobody has a structure
for, so any number computed on the folded state describes a reaction that cannot
start.

This module turns the question round. Ebselen is reported to work in the PROTECTIVE
direction, catalysing formation of that same disulfide and keeping SOD1 oxidised and
folded. That question is well posed for exactly the reason the destructive one is
not: its product is the structure we do know, at 1.24 A.

Mechanistically the catalysis is two successive S_N2 displacements at a chalcogen,
each an attack on a dichalcogenide:

    X(-)  +  Y-Z   ->   X-Y  +  Z(-)        xi = d(X...Y) - d(Y-Z)

which is the same antisymmetric stretch, driven by the same restraint and found by
the same P-RFO saddle search, as the proton transfers in the first two manuscripts.
Two things make it worth doing. The systems are small enough that CCSD(T) converges
rather than being estimated, and solution rates for thiol-disulfide and
selenol-diselenide exchange are measured, so the answer can be checked instead of
merely reported.

The set below is chosen so that each row isolates one substitution:

    ss       thiolate   + disulfide     attack S    the reference, and the measured one
    se_s     thiolate   + Se-S          attack Se   the ebselen turnover step
    s_se     thiolate   + Se-S          attack S    the other regiochemistry, same molecule
    sese     selenolate + diselenide    attack Se   both partners soft
    se_nuc   selenolate + disulfide     attack S    nucleophile changed, electrophile not
    neutral  thiol      + disulfide     attack S    control: must come out far higher

`se_s` against `ss` is the electrophile comparison and `se_nuc` against `ss` the
nucleophile one, so the two effects separate instead of being folded together.
`s_se` and `se_s` are the same molecule attacked at its two ends, which cancels
almost everything except the identity of the atom under attack. `neutral` exists
because only the thiolate reacts at a useful rate; if the neutral control does not
come out far above the anions, something is wrong with the model and not with the
chemistry.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# label -> (nucleophile SMILES, dichalcogenide SMILES, attacked element, leaving element, charge)
SYSTEMS = {
    'ss':      ('[CH3][S-]',  'CSSC',       'S',  'S',  -1),
    'se_s':    ('[CH3][S-]',  'C[Se]SC',    'Se', 'S',  -1),
    's_se':    ('[CH3][S-]',  'C[Se]SC',    'S',  'Se', -1),
    'sese':    ('[CH3][Se-]', 'C[Se][Se]C', 'Se', 'Se', -1),
    'se_nuc':  ('[CH3][Se-]', 'CSSC',       'S',  'S',  -1),
    'neutral': ('CS',         'CSSC',       'S',  'S',   0),
}

CHALCOGENS = ('S', 'Se')

# Where a reactive encounter complex sits, and what a formed bond looks like. Used to
# place the nucleophile and to check afterwards that the placement was sane.
APPROACH = 3.4          # A, initial X...Y separation: past bonding, inside vdW contact
BONDED = {('S', 'S'): 2.05, ('S', 'Se'): 2.19, ('Se', 'S'): 2.19, ('Se', 'Se'): 2.33}


def _embed(smiles):
    """3D coordinates for a small molecule. ETKDG then MMFF where MMFF knows the atom
    types; selenium and the anions it does not, and xTB relaxes those downstream."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE) != 0:
        raise ValueError(f'could not embed {smiles}')
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception:
        pass
    conf = m.GetConformer()
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    X = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
    return m, sym, X


def _rotation(a, b):
    """Rotation matrix taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def _dichalcogenide_pair(m, sym):
    """The two mutually bonded chalcogens, as (index, index)."""
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if sym[i] in CHALCOGENS and sym[j] in CHALCOGENS:
            return i, j
    raise ValueError('no chalcogen-chalcogen bond found')


def build(label):
    """A reactant encounter complex with the nucleophile placed anti to the leaving
    group, which is the only arrangement an S_N2 at a chalcogen can proceed from.

    Returns (sym, X, charge, idx) with idx naming nuc / attacked / leaving."""
    nuc_smi, di_smi, at_el, lv_el, charge = SYSTEMS[label]

    dm, dsym, dX = _embed(di_smi)
    i, j = _dichalcogenide_pair(dm, dsym)
    # pick which end is attacked; for a homo-dichalcogenide the two are equivalent
    if dsym[i] == at_el and dsym[j] == lv_el:
        attacked, leaving = i, j
    elif dsym[j] == at_el and dsym[i] == lv_el:
        attacked, leaving = j, i
    else:
        raise ValueError(f'{label}: {di_smi} has no {at_el}-{lv_el} pair')

    nm, nsym, nX = _embed(nuc_smi)
    # the nucleophilic atom, and the carbon it hangs off
    cand = [k for k, s in enumerate(nsym) if s in CHALCOGENS]
    if len(cand) != 1:
        raise ValueError(f'{label}: expected one chalcogen in {nuc_smi}')
    nuc_local = cand[0]
    c_local = [b.GetOtherAtomIdx(nuc_local) for b in nm.GetAtomWithIdx(nuc_local).GetBonds()
               if nsym[b.GetOtherAtomIdx(nuc_local)] == 'C'][0]

    # backside: approach along the leaving->attacked direction, continued past it
    axis = dX[attacked] - dX[leaving]
    axis /= np.linalg.norm(axis)
    target = dX[attacked] + APPROACH * axis

    # orient the nucleophile so its own X->C bond points away from the electrophile,
    # which keeps its methyl out of the way of the incoming trajectory
    R = _rotation(nX[c_local] - nX[nuc_local], axis)
    nX = (nX - nX[nuc_local]) @ R.T + target

    sym = list(dsym) + list(nsym)
    X = np.vstack([dX, nX])
    idx = dict(nuc=len(dsym) + nuc_local, attacked=attacked, leaving=leaving,
               nuc_carbon=len(dsym) + c_local,
               attacked_el=at_el, leaving_el=lv_el,
               n_di=len(dsym), n_nuc=len(nsym))
    return sym, X, charge, idx


def xi(X, idx):
    """The reaction coordinate: d(X...Y) - d(Y-Z). Negative on the reactant side once
    the new bond is shorter than the old one, zero near the symmetric point."""
    a = np.linalg.norm(X[idx['nuc']] - X[idx['attacked']])
    b = np.linalg.norm(X[idx['attacked']] - X[idx['leaving']])
    return float(a - b)


def attack_angle(X, idx):
    """The X...Y-Z angle in degrees. An S_N2 at a chalcogen wants this near 180; a
    value well below that means the restraint has pulled the nucleophile in sideways
    and the saddle it finds is not the displacement."""
    u = X[idx['nuc']] - X[idx['attacked']]
    v = X[idx['leaving']] - X[idx['attacked']]
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def bond_lengths(X, idx):
    """The forming and breaking distances, for the sanity checks."""
    return dict(forming=float(np.linalg.norm(X[idx['nuc']] - X[idx['attacked']])),
                breaking=float(np.linalg.norm(X[idx['attacked']] - X[idx['leaving']])))


def expected_bond(idx, which='forming'):
    """What the formed bond should measure, from the covalent radii of the pair."""
    if which == 'forming':
        key = ('Se' if idx['nuc_el'] == 'Se' else 'S', idx['attacked_el']) \
            if 'nuc_el' in idx else (idx['attacked_el'], idx['attacked_el'])
    else:
        key = (idx['attacked_el'], idx['leaving_el'])
    return BONDED.get(key, 2.05)


def product_geometry(sym, X, idx):
    """Mirror the reactant complex through the symmetric point: the leaving group
    pushed out to the approach distance and the nucleophile brought in to bonding.
    A starting guess for the product relaxation, not a product."""
    X = np.array(X, float)
    axis = X[idx['attacked']] - X[idx['leaving']]
    axis /= np.linalg.norm(axis)
    d_new = BONDED.get((idx['attacked_el'], idx['leaving_el']), 2.05)
    # move the whole nucleophile fragment in to bonding distance
    n0 = idx['n_di']
    shift = (X[idx['attacked']] + d_new * axis) - X[idx['nuc']]
    X[n0:] += shift
    # and push the leaving fragment out along the same axis
    leaving_side = _fragment(sym, X, idx['leaving'], exclude=idx['attacked'])
    X[list(leaving_side)] -= APPROACH * axis
    return X


def _fragment(sym, X, seed, exclude):
    """Atoms connected to `seed` without passing through `exclude`, by distance."""
    cut = 2.6
    seen, stack = {seed}, [seed]
    while stack:
        a = stack.pop()
        for b in range(len(X)):
            if b in seen or b == exclude:
                continue
            if np.linalg.norm(X[a] - X[b]) < cut:
                seen.add(b)
                stack.append(b)
    return seen
