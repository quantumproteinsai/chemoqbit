"""The conformational work of bringing Cys6 into attack range of the SOD1 disulfide.

The 2013 mutational work (C6A stays cytotoxic, C111A much less so) established that
Cys6 is the initiator of disulfide scrambling, and explained it structurally as
"the proximity of C146 to C6". Proximity is the word doing the work there, and it
is doing too much: our own measurement puts the two sulfurs 8.37 A apart with a
133 deg approach angle, which is not a reactive encounter but a quarter of the
protein away in S_N2 terms.

What nobody has is the energy. This module computes it.

Looking at what sits on the line between them settles the shape of the problem.
The straight path from Cys6 S(gamma) to Cys146 S(gamma) is blocked by MAIN CHAIN,
not by side chains that could rotate aside:

    GLY 147  N    1.35 A off the axis
    LEU 117  O    1.78 A off the axis
    LEU 8    CD2  2.91 A
    ALA 145  O    3.08 A

and the two cysteines sit on packed neighbouring strands of the beta-barrel:
Cys6 CA is 4.7 A from Ile149 CA, Cys146 CA is 4.6 A from Leu8 CA. So the approach
cannot be a side-chain rotation. It needs the backbone to move, which is the
structural content of the literature's "denaturation allows Cys6 and Cys111 to
attack" and the reason a barrier computed on the folded state means nothing.

Hence a cluster that keeps WHOLE RESIDUES with their backbone. `pocket.cut` keeps
side chains only and caps at CA-CB, which answers a different question and would
delete the atoms that are in the way.

The quantity computed is the work W(d) to bring the two sulfurs from 8.37 A to
contact, under two constraint regimes that bracket the real protein:

    rigid   every backbone N, CA, C, O held at its crystallographic position.
            The fold cannot yield at all. An UPPER bound.
    mobile  only the outer shell held. The local region relaxes freely.
            A LOWER bound.

The true cost lies between them, and either bound can be decisive on its own: if
even the mobile lower bound clears the 31 kcal/mol that a fifty-year disease
allows, then scrambling cannot be the initiating event in the folded protein, and
something else has to happen first.
"""
import numpy as np

# Chain F conformer B is the one with a genuine 2.00 A disulfide; chain A carries
# the SSBOND record but models it at 2.75 A, too long to be bonded.
PDB = 'data/rcsb_2c9u/2C9U.pdb'
CHAIN = 'F'
ALTLOC = ('', 'B')

# Everything within reach of the Cys6 -> Cys146 axis, plus the strands that hold
# the two cysteines in place. Chosen from the blocking analysis above rather than
# by a distance cutoff, so the atoms that must move are all present.
RESIDUES = (6, 7, 8, 9, 17, 18, 57, 116, 117, 118, 119, 145, 146, 147, 148, 149, 150)

BACKBONE = ('N', 'CA', 'C', 'O')
# The four residues whose backbone sits ON the Cys6...Cys146 axis and therefore
# has to move for the approach to happen at all: Gly147 N is 1.34 A off the line,
# Leu117 O 1.78 A, and Cys146's own CA and CB 1.19 and 1.23 A. In the `mobile`
# regime these four keep their backbone free and every other residue does not.
AXIS_BACKBONE = (6, 117, 146, 147)

FORMAL = {'LYS': 1, 'ARG': 1, 'ASP': -1, 'GLU': -1}

NH = 1.01
CH = 1.09


def read_chain(pdb=PDB, chain=CHAIN, altloc=ALTLOC):
    """{(resnum, atomname): (resname, xyz)} for the chosen chain and conformer."""
    out = {}
    for ln in open(pdb):
        if ln[:4] != 'ATOM' or ln[21] != chain:
            continue
        if ln[16] not in (' ',) + tuple(a for a in altloc if a):
            continue
        rn, ri, an = ln[17:20].strip(), int(ln[22:26]), ln[12:16].strip()
        if an.startswith('H') or an[0].isdigit():
            continue
        x = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        out[(ri, an)] = (rn, x)
    return out


def fragments(residues=RESIDUES):
    """Contiguous runs, so each is capped once at each end rather than per residue."""
    out, run = [], [residues[0]]
    for r in residues[1:]:
        if r == run[-1] + 1:
            run.append(r)
        else:
            out.append(run)
            run = [r]
    out.append(run)
    return out


def build(pdb=PDB, chain=CHAIN, residues=RESIDUES, cys6_thiolate=True):
    """Whole-residue cluster with hydrogen caps at the chain breaks.

    Returns (symbols, X, charge, index, frozen_rigid, frozen_mobile) where `index`
    names the two sulfurs the restraint acts on and `frozen_*` are the two
    constraint regimes."""
    at = read_chain(pdb, chain)
    present = {ri for (ri, _) in at}
    missing = [r for r in residues if r not in present]
    if missing:
        raise ValueError(f'residues absent from chain {chain}: {missing}')

    sym, X, label = [], [], []

    def add(el, x, lab):
        sym.append(el)
        X.append(np.asarray(x, float))
        label.append(lab)

    def el_of(an):
        return an[0] if an[0] in 'CNOS' else 'C'

    for run in fragments(residues):
        for ri in run:
            names = sorted(an for (r, an) in at if r == ri)
            rn = at[(ri, names[0])][0]
            for an in names:
                _, x = at[(ri, an)]
                add(el_of(an), x, f'{rn}{ri}:{an}')
        # cap the two broken peptide bonds of this fragment using the real
        # positions of the neighbours we are leaving out, so the cap points the
        # way the chain actually went
        first, last = run[0], run[-1]
        rn_f = at[(first, 'N')][0]
        n = at[(first, 'N')][1]
        prev_c = at.get((first - 1, 'C'))
        direc = (prev_c[1] - n) if prev_c else (n - at[(first, 'CA')][1])
        add('H', n + NH * direc / np.linalg.norm(direc), f'{rn_f}{first}:capN')

        rn_l = at[(last, 'C')][0]
        c = at[(last, 'C')][1]
        next_n = at.get((last + 1, 'N'))
        direc = (next_n[1] - c) if next_n else (c - at[(last, 'CA')][1])
        add('H', c + CH * direc / np.linalg.norm(direc), f'{rn_l}{last}:capC')

    charge = sum(FORMAL.get(at[(ri, 'CA')][0], 0) for ri in residues)
    if cys6_thiolate:
        charge -= 1          # the nucleophile is the thiolate, not the thiol

    def find(ri, an):
        for i, l in enumerate(label):
            if l.endswith(f'{ri}:{an}'):
                return i
        raise KeyError(f'{ri}:{an}')

    idx = dict(S6=find(6, 'SG'), S146=find(146, 'SG'), S57=find(57, 'SG'))

    rigid, mobile = frozen_sets(label)
    return sym, np.array(X), charge, idx, label, rigid, mobile


def frozen_sets(label):
    """The two constraint regimes, as atom indices to hold.

    rigid   every backbone N, CA, C, O at its crystallographic position
    mobile  the same, except on the four residues lying on the axis

    Hydrogens are free in both. Their positions are not crystallographic -- they
    were added here -- so holding them would freeze a guess."""
    rigid, mobile = [], []
    for i, l in enumerate(label):
        if ':' not in l:
            continue                       # an added hydrogen, never held
        res, an = l.split(':')
        if an not in BACKBONE:
            continue
        rigid.append(i)
        if int(''.join(c for c in res if c.isdigit())) not in AXIS_BACKBONE:
            mobile.append(i)
    return rigid, mobile


def separation(X, idx):
    return float(np.linalg.norm(X[idx['S6']] - X[idx['S146']]))


def blocking(X, label, idx, within=4.0):
    """Atoms lying near the Cys6...Cys146 axis, with how far along it they sit.
    Recomputed from whatever geometry is passed, so it can be followed along a
    scan to see what actually moved out of the way."""
    a, b = X[idx['S6']], X[idx['S146']]
    u = b - a
    L = float(np.linalg.norm(u))
    u = u / L
    out = []
    for i, (x, l) in enumerate(zip(X, label)):
        if i in (idx['S6'], idx['S146']):
            continue
        v = x - a
        t = float(np.dot(v, u))
        if not (0.5 < t < L - 0.5):
            continue
        perp = float(np.linalg.norm(v - t * u))
        if perp < within:
            out.append(dict(label=l, along=t, off_axis=perp))
    return sorted(out, key=lambda d: d['off_axis'])


def protonate(sym, X, charge, idx, label):
    """Add hydrogens with RDKit.

    Bond perception is by distance, then every backbone C=O is set double by hand.
    That is safe here and would not be in general: the seventeen residues of this
    cluster are Cys, Val, Leu, Lys, Ile, Thr, Ala and Gly, all aliphatic, so the
    backbone carbonyls are the ONLY multiple bonds in it. There is no aromatic
    ring, no carboxylate and no side-chain amide to get wrong.

    Returns (symbols, X, charge, index, label) with hydrogens appended; the heavy
    atoms keep their indices, so `idx` and the frozen lists stay valid."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D

    rw = Chem.RWMol()
    conf = Chem.Conformer(len(sym))
    for i, (e, x) in enumerate(zip(sym, X)):
        rw.AddAtom(Chem.Atom(e))
        conf.SetAtomPosition(i, Point3D(*map(float, x)))
    mol = rw.GetMol()
    mol.AddConformer(conf)
    rdDetermineBonds.DetermineConnectivity(mol)

    rw = Chem.RWMol(mol)
    for i, l in enumerate(label):
        if not l.endswith(':C'):
            continue
        res = l.split(':')[0]
        o = next((j for j, m in enumerate(label) if m == f'{res}:O'), None)
        if o is not None and rw.GetBondBetweenAtoms(i, o) is not None:
            rw.GetBondBetweenAtoms(i, o).SetBondType(Chem.BondType.DOUBLE)

    rw.GetAtomWithIdx(idx['S6']).SetFormalCharge(-1)
    for i, l in enumerate(label):
        if l.endswith(':NZ'):
            rw.GetAtomWithIdx(i).SetFormalCharge(1)
    for a in rw.GetAtoms():
        a.SetNoImplicit(False)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    molH = Chem.AddHs(mol, addCoords=True)

    conf = molH.GetConformer()
    symH = [a.GetSymbol() for a in molH.GetAtoms()]
    XH = np.array([list(conf.GetAtomPosition(i)) for i in range(molH.GetNumAtoms())])
    labH = list(label) + [f'H{i}' for i in range(len(symH) - len(label))]
    return symH, XH, Chem.GetFormalCharge(molH), idx, labH


# Covalent radii, enough for the elements these clusters contain. The 1.25 factor
# is the usual slack for calling a distance a bond; it is deliberately generous,
# because the list is built once from a RELAXED geometry and the point is to
# notice a bond leaving, not to argue about where one begins.
RCOV = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05}


def covalent_bonds(sym, X, slack=1.25):
    """[(i, j, length)] for every pair bonded in this geometry.

    Built once from the relaxed reference so that later geometries are judged
    against it rather than against a fresh guess: re-deriving the list at every
    scan point would silently accept a broken bond as simply absent.
    """
    X = np.asarray(X, float)
    out = []
    for i in range(len(sym)):
        d = np.linalg.norm(X[i + 1:] - X[i], axis=1)
        for k, dk in enumerate(d):
            j = i + 1 + k
            if dk < slack * (RCOV[sym[i]] + RCOV[sym[j]]):
                out.append((i, j, float(dk)))
    return out


def worst_stretch(bonds, X):
    """(ratio, i, j) for the bond of `bonds` most stretched in geometry X."""
    X = np.asarray(X, float)
    return max((float(np.linalg.norm(X[i] - X[j]) / r0), i, j) for i, j, r0 in bonds)


def new_bonds(sym, X, bonds, slack=1.25):
    """Pairs bonded in X that are not in `bonds`. A rupture usually comes with
    one of these, and it is the more legible half of the diagnosis: the Cys146
    sulfur that left its own beta carbon bonded the Gly147 carbonyl oxygen."""
    X = np.asarray(X, float)
    had = {(i, j) for i, j, _ in bonds}
    out = []
    for i in range(len(sym)):
        d = np.linalg.norm(X[i + 1:] - X[i], axis=1)
        for k, dk in enumerate(d):
            j = i + 1 + k
            if (i, j) not in had and dk < slack * (RCOV[sym[i]] + RCOV[sym[j]]):
                out.append((i, j, float(dk)))
    return out


def attack_angle(X, idx):
    """The Cys6...Cys146-Cys57 angle in degrees. An S_N2 at sulfur wants this near
    180: the nucleophile has to arrive behind the bond it is breaking. The crystal
    structure gives 133.2 and the relaxed cluster 121.1, so a scan that drives only
    the S...S distance reaches contact at the wrong angle and measures a compressed
    contact pair instead of an approach."""
    u = X[idx['S6']] - X[idx['S146']]
    v = X[idx['S57']] - X[idx['S146']]
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
