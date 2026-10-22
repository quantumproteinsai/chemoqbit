"""Scheme 1 of Manuscript 2: the hydride transfer, drawn with RDKit.
Writes results/figures/scheme1.png. R1 = deoxyribose 5'-phosphate,
R2 = CH2 of Cys146, R3 = the CH2-NH-pABA-Glu tail of the folate."""
import io
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUT = Path(__file__).resolve().parents[1] / 'results' / 'figures' / 'scheme1.png'
P = Chem.SmilesParserParams(); P.removeHs = False
SPECIES = [  # smiles, title, {atom index: note}
    ('[H]/C([H])=C1/C(=O)NC(=O)N([*:1])C1([H])S[*:2]', 'exocyclic methylene\nintermediate', {}),
    ('O=c1[nH]c(N)nc2c1NC([H])([*:3])CN2', 'tetrahydrofolate', {}),
    ('[H]C([H])([H])C1=C([O-])NC(=O)N([*:1])C1([H])S[*:2]', 'enolate', {}),
    ('O=c1[nH]c(N)nc2c1[NH+]=C([*:3])CN2', 'dihydrofolate\n(iminium)', {}),
]
LABEL = {1: 'R<sup>1</sup>', 2: 'R<sup>2</sup>', 3: 'R<sup>3</sup>'}


def note_atoms(m, which):
    """label C5, C6, C11, O4 (nucleotide) or C6, N5 (folate) from the ring topology"""
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetProp('atomLabel', LABEL[a.GetAtomMapNum()]); a.SetAtomMapNum(0)
    if which in (0, 2):
        s = next(a for a in m.GetAtoms() if a.GetSymbol() == 'S')
        c6 = next(n for n in s.GetNeighbors() if n.GetSymbol() == 'C' and n.IsInRing())
        c5 = next(n for n in c6.GetNeighbors() if n.GetSymbol() == 'C' and n.IsInRing())
        c11 = next(n for n in c5.GetNeighbors() if n.GetSymbol() == 'C' and not n.IsInRing())
        c4 = next(n for n in c5.GetNeighbors() if n.IsInRing() and n.GetIdx() != c6.GetIdx())
        o4 = next(n for n in c4.GetNeighbors() if n.GetSymbol() == 'O')
        for a, t in ((c5, 'C5'), (c6, 'C6'), (c11, 'C11'), (o4, 'O4')):
            a.SetProp('atomNote', t)
        return [h.GetIdx() for h in c11.GetNeighbors() if h.GetSymbol() == 'H'][:1] if which == 2 else []
    c6 = next(a for a in m.GetAtoms() if a.GetSymbol() == 'C' and any(n.GetAtomicNum() == 0 for n in a.GetNeighbors()))
    n5 = next(n for n in c6.GetNeighbors() if n.GetSymbol() == 'N' and n.IsInRing() and
              any(x.GetIsAromatic() for x in n.GetNeighbors()))
    c6.SetProp('atomNote', 'C6'); n5.SetProp('atomNote', 'N5')
    return [h.GetIdx() for h in c6.GetNeighbors() if h.GetSymbol() == 'H']


# ring cores shared by reactant and product, so each product keeps its reactant's orientation
CORES = {0: '[#6]~[#6]1~[#6](~[#8])~[#7]~[#6](~[#8])~[#7]~[#6]~1~[#16]',
         1: '[#6]1~[#7]~[#6]2~[#6](~[#8])~[#7]~[#6](~[#7])~[#7]~[#6]~2~[#7]~[#6]~1'}
REF = {}


def png(i, smi):
    m = Chem.MolFromSmiles(smi, P)
    hl = note_atoms(m, i)
    rdDepictor.SetPreferCoordGen(True)
    core = Chem.MolFromSmarts(CORES[i % 2])
    if i < 2:
        rdDepictor.Compute2DCoords(m); REF[i] = m
    else:
        rdDepictor.GenerateDepictionMatching2DStructure(m, REF[i - 2], refPatt=core)
    d = rdMolDraw2D.MolDraw2DCairo(420, 360)
    o = d.drawOptions(); o.addStereoAnnotation = False; o.annotationFontScale = 0.75
    o.setHighlightColour((0.16, 0.47, 0.84, 0.35)); o.bondLineWidth = 2
    d.DrawMolecule(m, highlightAtoms=hl, highlightBonds=[])
    d.FinishDrawing()
    return mpimg.imread(io.BytesIO(d.GetDrawingText()), format='png')


fig = plt.figure(figsize=(16, 4.6), dpi=150)
xs = [0.0, 0.25, 0.53, 0.78]
for i, (smi, title, _) in enumerate(SPECIES):
    ax = fig.add_axes([xs[i], 0.12, 0.22, 0.78]); ax.imshow(png(i, smi)); ax.axis('off')
    ax.set_title(title, fontsize=13, color='#1f1f1e')
fig.text(0.235, 0.5, '+', fontsize=28, ha='center', va='center', color='#1f1f1e')
fig.text(0.765, 0.5, '+', fontsize=28, ha='center', va='center', color='#1f1f1e')
fig.patches.append(matplotlib.patches.FancyArrowPatch((0.475, 0.5), (0.535, 0.5), transform=fig.transFigure,
                                                      arrowstyle='-|>', mutation_scale=22, lw=2, color='#1f1f1e'))
fig.text(0.505, 0.58, 'H⁻', fontsize=14, ha='center', color='#2a78d6')
fig.text(0.5, 0.03, 'R¹ = 2′-deoxyribose 5′-phosphate;  R² = CH₂ of Cys146;  '
         'R³ = CH₂–NH–pABA–Glu.  The transferring hydrogen is highlighted.',
         fontsize=11, ha='center', color='#55554f')
fig.savefig(OUT, facecolor='white'); print(OUT)
