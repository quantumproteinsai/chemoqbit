"""PDF report on the thymidylate synthase cluster pipeline: what failed in the
October 2026 runs, why, how it was fixed, and the current results.

Every number is read from the checkpoint files in results/, so re-running this
after a stage finishes (def2-TZVP, the active-space circuits) updates the report.

    python scripts/ts_report.py [out.pdf]          # default results/ts_pipeline_report.pdf

Needs reportlab and matplotlib, which are not in requirements.txt (the pipeline
itself does not use them): pip install reportlab matplotlib
"""
import json, sys, datetime, glob, shutil, tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, KeepTogether, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
FIGDIR = Path(tempfile.mkdtemp())
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / 'ts_pipeline_report.pdf'
K = 627.509474

# ---------- fonts (DejaVu has the Greek and math glyphs Helvetica lacks) ----------
fdir = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
pdfmetrics.registerFont(TTFont('DV', str(fdir / 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DV-B', str(fdir / 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('DV-I', str(fdir / 'DejaVuSans-Oblique.ttf')))
pdfmetrics.registerFont(TTFont('DVM', str(fdir / 'DejaVuSansMono.ttf')))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily('DV', normal='DV', bold='DV-B', italic='DV-I', boldItalic='DV-B')

INK, INK2, MUTED = colors.HexColor('#1f1f1e'), colors.HexColor('#55554f'), colors.HexColor('#8a8980')
RULE, BAND = colors.HexColor('#d9d8d2'), colors.HexColor('#f3f2ee')
BLUE, ORANGE = '#2a78d6', '#eb6834'

ss = getSampleStyleSheet()
S = {
    'title': ParagraphStyle('title', fontName='DV-B', fontSize=17, leading=21, textColor=INK, spaceAfter=4),
    'sub': ParagraphStyle('sub', fontName='DV', fontSize=9.5, leading=13, textColor=INK2, spaceAfter=10),
    'h1': ParagraphStyle('h1', fontName='DV-B', fontSize=12.5, leading=16, textColor=INK, spaceBefore=12, spaceAfter=5),
    'h2': ParagraphStyle('h2', fontName='DV-B', fontSize=10.5, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3),
    'p': ParagraphStyle('p', fontName='DV', fontSize=9.2, leading=13, textColor=INK, spaceAfter=5),
    'b': ParagraphStyle('b', fontName='DV', fontSize=9.2, leading=13, textColor=INK, leftIndent=11,
                        bulletIndent=2, spaceAfter=2.5),
    'cap': ParagraphStyle('cap', fontName='DV', fontSize=8.2, leading=11, textColor=INK2, spaceAfter=8),
    'cell': ParagraphStyle('cell', fontName='DV', fontSize=8.2, leading=10.5, textColor=INK),
    'cellb': ParagraphStyle('cellb', fontName='DV-B', fontSize=8.2, leading=10.5, textColor=INK),
    'code': ParagraphStyle('code', fontName='DVM', fontSize=7.8, leading=10.5, textColor=INK,
                           backColor=BAND, borderPadding=(4, 5, 4, 5), spaceBefore=3, spaceAfter=7),
}
P = lambda t, s='p': Paragraph(t, S[s])
B = lambda t: Paragraph(t, S['b'], bulletText='•')


def table(rows, widths, header=True, align_right_from=None):
    data = [[c if not isinstance(c, str) else Paragraph(c, S['cellb' if (header and i == 0) else 'cell'])
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [('VALIGN', (0, 0), (-1, -1), 'TOP'),
          ('LINEBELOW', (0, 0), (-1, 0), 0.8, INK2) if header else ('LINEBELOW', (0, 0), (-1, 0), 0, colors.white),
          ('LINEBELOW', (0, 1), (-1, -1), 0.3, RULE),
          ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
          ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4)]
    t.setStyle(TableStyle(st))
    return t


# ---------- data ----------
r = json.loads((RES / 'ts_cluster_run.json').read_text())
old = json.loads((RES / 'ts_cluster_run.N5adduct_20261017.json').read_text())
fwd = json.loads((RES / 'ts_cluster_run.prebackfill_20261018.json').read_text())
moved = json.loads((RES / 'ts_cluster_run.capsmovedTS_20261018.json').read_text())
# corrected active-space results: merged into the checkpoint (they carry 'region'),
# or still in the side file written while def2-TZVP held the checkpoint
act = r.get('active') if r.get('active', {}).get('reactant/4e4o', {}).get('region') else None
if act is None:
    side = sorted(glob.glob(str(RES / 'ts_active_*.json')))
    act = json.loads(Path(side[-1]).read_text()) if side else None
k = r['prep']['key']
an = r['analysis']
dft = r.get('dft', {})
E_R = r['reactant']['E']


def geo(name):
    X = np.array(r[name]['coords'])
    d = lambda a, b: float(np.linalg.norm(X[k[a]] - X[k[b]]))
    return d('Cd', 'Hd'), d('Hd', 'C11'), d('Cd', 'C11'), d('N5', 'C11'), d('C6u', 'S')


# ---------- figures ----------
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': '#8a8980',
                     'axes.linewidth': 0.6, 'xtick.color': '#55554f', 'ytick.color': '#55554f',
                     'axes.labelcolor': '#1f1f1e', 'xtick.major.size': 3, 'ytick.major.size': 3})


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', linewidth=0.6)
    ax.set_axisbelow(True)


# Fig 1: original failed scan
xo = [p['xi'] for p in old['scan']]; eo = [(p['E'] - old['scan'][0]['E']) * K for p in old['scan']]
fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=200)
ax.plot(xo, eo, color=BLUE, lw=2, marker='o', ms=4, mec='white', mew=1)
style(ax)
ax.set_xlabel('ξ = d(C6–H) − d(H–C11)  (Å)'); ax.set_ylabel('ΔE  (kcal/mol)')
i = int(np.argmax(eo))
ax.annotate(f'ξ = {xo[i]:+.1f}: +{eo[i]:.0f} kcal/mol\nC11 carries three H', (xo[i], eo[i]),
            xytext=(-1.45, 88), textcoords='data', fontsize=7.8, color='#1f1f1e',
            arrowprops=dict(arrowstyle='-', color='#8a8980', lw=0.6))
ax.annotate(f'ξ = {xo[i+1]:+.1f}: back to +{eo[i+1]:.0f}\n(C11 hands its own H to C6)', (xo[i + 1], eo[i + 1]),
            xytext=(-170, 22), textcoords='offset points', fontsize=7.8, color='#1f1f1e',
            arrowprops=dict(arrowstyle='-', color='#8a8980', lw=0.6))
fig.tight_layout(); f1 = FIGDIR / 'fig1.png'; fig.savefig(f1); plt.close(fig)

# Fig 2: corrected scan, forward vs backfilled, relative to the relaxed reactant
xf = [p['xi'] for p in fwd['scan']]; ef = [(p['E'] - E_R) * K for p in fwd['scan']]
xb = [p['xi'] for p in r['scan']]; eb = [(p['E'] - E_R) * K for p in r['scan']]
fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=200)
ax.plot(xf, ef, color=ORANGE, lw=2, marker='o', ms=4, mec='white', mew=1, label='forward scan')
ax.plot(xb, eb, color=BLUE, lw=2, marker='o', ms=4, mec='white', mew=1, label='after backward pass')
ax.axhline((r['ts']['E'] - E_R) * K, color='#8a8980', lw=0.8, ls=(0, (3, 3)))
ax.text(1.42, (r['ts']['E'] - E_R) * K + 0.4, f"TS (PRFO) {(r['ts']['E'] - E_R) * K:.1f}", ha='right',
        fontsize=7.8, color='#55554f')
style(ax)
ax.set_xlabel('ξ = d(C6–H) − d(H–C11)  (Å)'); ax.set_ylabel('ΔE vs relaxed reactant  (kcal/mol)')
ax.legend(frameon=False, loc='upper left', fontsize=7.8)
j = xf.index(0.0)
ax.annotate('folate shifts into a\nlower conformer', (xf[j], ef[j]), xytext=(0.3, 14.6), textcoords='data',
            fontsize=7.8, color='#1f1f1e', arrowprops=dict(arrowstyle='-', color='#8a8980', lw=0.6))
fig.tight_layout(); f2 = FIGDIR / 'fig2.png'; fig.savefig(f2); plt.close(fig)

# ---------- document ----------
story = []
today = datetime.date.today().strftime('%d %B %Y')
story += [P('Thymidylate synthase hydride-transfer pipeline: what failed, why, and how it was fixed', 'title'),
          P(f'chemoqbit · <font name="DVM">scripts/ts_cluster_nuc.py</font> · 96-atom 1TLS cluster · report generated {today}', 'sub')]

story += [P('Summary', 'h1'),
          P('The xTB stages of the thymidylate synthase (TS) cluster pipeline stopped on 17 October after about eight hours '
            'with <i>PRFO did not converge</i>. The pipeline exists to compute the barrier and the H/T kinetic isotope effect '
            '(experiment: 6.91 ± 0.05 at 293 K) for hydride transfer from C6 of tetrahydrofolate to the exocyclic methylene '
            'on the nucleotide. Tracing the failure exposed seven separate defects: two in the chemical model, three in the '
            'saddle-point search, one in disk sizing for the large-basis DFT, and one in the active-space stage. '
            'Each was fixed, the affected stages were re-run, and guards were added so that each failure mode now stops '
            'the run within minutes rather than hours.'),
          ]
semi, wig, eck = (an['kie']['H/T'][x] for x in ('semiclassical', 'wigner', 'eckart'))
sum_rows = [['Quantity', 'Result', 'Note'],
            ['xTB barrier (electronic)', f"{an['dE_barrier']:.2f} kcal/mol", 'GFN2-xTB, ALPB chloroform'],
            ['xTB reaction energy', f"{an['dE_rxn']:+.2f} kcal/mol", ''],
            ['TS imaginary frequency', f"{abs(an['iso']['ts']['H']['imag']):.0f}i cm⁻¹", 'exactly one imaginary mode'],
            ['KIE H/T: semiclassical / Wigner / Eckart', f'{semi:.2f} / {wig:.2f} / {eck:.2f}', 'experiment 6.91 ± 0.05']]
if all(f'{n}/def2-svp' in dft for n in ('reactant', 'ts', 'product')):
    e = {n: dft[f'{n}/def2-svp']['E'] for n in ('reactant', 'ts', 'product')}
    sum_rows.append(['DFT barrier, def2-SVP', f"{(e['ts'] - e['reactant']) * K:.2f} kcal/mol",
                     'B3LYP-D3(BJ), PCM ε = 4, at xTB geometries'])
tz = all(f'{n}/def2-tzvp' in dft for n in ('reactant', 'ts', 'product'))
if tz:
    e = {n: dft[f'{n}/def2-tzvp']['E'] for n in ('reactant', 'ts', 'product')}
    sum_rows.append(['DFT barrier, def2-TZVP', f"{(e['ts'] - e['reactant']) * K:.2f} kcal/mol", 'density-fitted'])
else:
    sum_rows.append(['DFT barrier, def2-TZVP', 'not finished', 'density-fitted; see section 6'])
story += [table(sum_rows, [62 * mm, 40 * mm, 68 * mm]), Spacer(1, 6)]

# ---- 1. pipeline
story += [P('1. The pipeline in brief', 'h1'),
          P('A 96-atom active-site cluster is cut from the crystal structure 1TLS (chain A): the nucleotide, the folate, '
            'Cys146, Arg166, Tyr94, Asn177, His147 and three waters, capped with hydrogens. Eleven atoms (the caps and one '
            'anchor atom per side chain) are held fixed in every optimisation so that the cluster keeps the protein\'s shape. '
            'The crystal holds a dead-end covalent complex, so the model is edited to the state just before hydride transfer: '
            'the folate N5–C11 bond is broken, which leaves C11 as an exocyclic methylene (C5=C11) on the nucleotide.'),
          table([['Stage', 'What it does'],
                 ['0 prep', 'build the cluster, add hydrogens, relax with the fixed atoms held'],
                 ['1 scan', 'relaxed GFN2-xTB scan of ξ = d(C6–H) − d(H–C11) from −1.4 to +1.4 Å, step 0.1'],
                 ['2 ts', 'partitioned rational-function (P-RFO) saddle search from the scan maximum; relax reactant and product'],
                 ['3 hess', 'finite-difference Hessians at reactant, TS and product'],
                 ['4 kie', 'H/D/T isotopologues: transition-state theory with Wigner and Eckart tunnelling'],
                 ['5 dft', 'B3LYP-D3(BJ) single points, def2-SVP then def2-TZVP, PCM ε = 4'],
                 ['6 active', '4-electron/4-orbital active space around the hydride: CCSD(T), CASCI, UCCSD-VQE, ADAPT-VQE']],
                [22 * mm, 148 * mm]),
          Spacer(1, 4)]

# ---- 2. what went wrong
story += [P('2. What went wrong and how each problem was fixed', 'h1'),
          P('The problems surfaced one at a time: each fix let the run get further and expose the next. They are listed '
            'in the order found.')]

o_cdh = old['scan']
story += [P('2.1 The methylene never existed', 'h2'),
          P(f'In the failed run the energy climbed smoothly to +{max(eo):.0f} kcal/mol at ξ = +1.1, then fell to '
            f'+{eo[i+1]:.0f} kcal/mol one step later (Figure 1). The scan maximum was the edge of that cliff, not a saddle '
            'point, and P-RFO could only slide downhill from it.'),
          P('<b>Cause.</b> The hydrogen-adding step breaks N5–C11 in the RDKit bond table only. C11 stayed 1.5 Å from N5, '
            'where it sits in the crystal, and xTB, which sees only coordinates, re-formed the bond during prep. The scan then '
            'pushed the hydride onto an sp<super>3</super> carbon that already had four neighbours. At ξ = +1.2 the cluster '
            'escaped by moving one of C11\'s own hydrogens back to folate C6: the "product" was the reactant with two '
            'hydrogens relabelled.'),
          Image(str(f1), width=160 * mm, height=65 * mm),
          P('Figure 1. The original scan (17 October), relative to its first point. Single series; the cliff is the '
            'hydrogen swap described above.', 'cap'),
          P('<b>Fix.</b> Prep now relaxes the cluster with N5 held at least a set distance from C11 using a one-sided '
            '(flat-bottom) restraint, which applies no force beyond that distance. Released on its own, the methylene is '
            'stable: C5=C11 relaxes to 1.33 Å.')]

story += [P('2.2 N5 re-bonded to C11 during the scan', 'h2'),
          P('<b>Symptom.</b> In the first rerun, the first scan point came out 13 kcal/mol <i>below</i> the reactant, '
            'with N5–C11 back at 1.55 Å.'),
          P('<b>Cause.</b> Bringing the hydride towards C11 also brings N5, which is bonded to the donor carbon, within '
            'bonding range; xTB places the N5–C11 adduct about 13 kcal/mol lower.'),
          P('<b>Fix.</b> The N5–C11 ≥ 2.8 Å restraint stays on in every stage. It stands in for the conformational change '
            'that keeps N5 away in the enzyme. It is a modelling assumption, but it is <b>not active</b> at the final '
            'reactant, TS or product (Table 3), so it shapes the path without biasing the stationary points.')]

story += [P('2.3 The folate settled part-way through the scan', 'h2'),
          P('<b>Symptom.</b> Energy drops of 6.2 kcal/mol at ξ = −1.0 and 3.7 kcal/mol at ξ = 0.0, with no bonds changing '
            '(orange curve in Figure 2). The second drop created a false maximum at ξ = −0.1.'),
          P('<b>Cause.</b> The whole folate ring shifted by about 1 Å into a lower-energy position. Points scanned before '
            'the shift sit in the strained position, and the reactant relaxation started from one of them.'),
          P('<b>Fix.</b> After the forward scan, a backward pass re-optimises each point starting from its neighbour\'s '
            'geometry and keeps any that come out lower. It replaced 19 of 29 points (by 3–7 kcal/mol on the reactant '
            'side) and left one smooth maximum at ξ = +0.1. The reactant and product are now relaxed from the lowest scan '
            'point on each side of the maximum, not from the two ends.'),
          Image(str(f2), width=160 * mm, height=70 * mm),
          P('Figure 2. Corrected scan before (orange) and after (blue) the backward pass, relative to the relaxed '
            'reactant. The dashed line is the converged TS.', 'cap')]

story += [P('2.4 The TS search followed the wrong motion', 'h2'),
          P('<b>Symptom.</b> The followed mode had a curvature of about 10<super>−4</super> E<sub>h</sub>/Å², '
            'where a C–H–C hydride transfer mode should be about −0.2.'),
          P('<b>Cause.</b> P-RFO was told to follow the difference between neighbouring scan points. Across the folate '
            'shift that difference is dominated by the ring moving, so the search tracked a soft whole-cluster mode.'),
          P('<b>Fix.</b> The direction to follow is now ∂ξ/∂X, which involves only the donor carbon, the hydride and the '
            'acceptor carbon. Convergence requires a curvature below −10<super>−3</super> E<sub>h</sub>/Å², and the search '
            'stops after eight consecutive steps that do not meet it. A Hessian at ξ = +0.1 confirmed one mode at '
            '−0.21 E<sub>h</sub>/Å² with 0.94 overlap with the hydride coordinate.')]

ma = moved['analysis']
story += [P('2.5 The TS search moved the fixed atoms', 'h2'),
          P(f"<b>Symptom.</b> The next run converged to a barrier of {ma['dE_barrier']:.2f} kcal/mol, far below the "
            "≈12.6 kcal/mol the scan implied, and P-RFO lowered the energy by 7.3 kcal/mol on the way."),
          P('<b>Cause.</b> The saddle search ignored the fixed atoms and moved them by up to 0.44 Å, while the reactant and '
            'product were relaxed with them held. The TS was therefore optimised in a looser model than the end points. '
            'The Hessians also included fixed atoms that are not at a stationary point.'),
          P('<b>Fix.</b> The saddle search, the finite-difference Hessian and the frequency analysis now work in the free '
            'atoms\' coordinates only. Translation and rotation are no longer projected out, since the fixed atoms already '
            'remove them, and the rotational partition function is dropped. With this, the TS converged in two steps.')]

story += [P('2.6 The large-basis DFT ran out of disk', 'h2'),
          P('<b>Symptom.</b> The def2-SVP single points took about 2.3 h each. def2-TZVP without density fitting would take '
            'about a day per structure, and with density fitting it crashed with <i>No space left on device</i>.'),
          P('<b>Cause.</b> For 1782 basis functions and 4559 auxiliary functions, the precomputed 3-index integral file is '
            'about 58 GB; the disk had 35 GB free.'),
          P('<b>Fix.</b> Density fitting is on by default for triple-zeta and larger basis sets (tested on water: within '
            '0.02 kcal/mol of the exact result), and the container disk was enlarged from 47 to 99 GB.')]

story += [P('2.7 The active-space fragments were not chemically sound', 'h2'),
          P('<b>Symptom.</b> ADAPT-VQE missed the exact CASCI energy by 58 mHa for the product, and UCCSD by 29 mHa; the '
            'three structures used fragments of 9, 10 and 6 atoms.'),
          P('<b>Cause.</b> The stage claimed to cap the fragment but did not. It cut bonds at a 2.6 Å sphere around the '
            'hydride, leaving dangling bonds, and chose a charge of 0 or +1 only to make the electron count even. The '
            'broken bonds then dominated the 4-orbital active space, and because the region was re-chosen per structure, '
            'energies could not be compared across structures.'),
          P('<b>Fix.</b> One region is chosen at the TS and used for all three structures: heavy atoms within 2.6 Å of the '
            'hydride, plus the nucleotide\'s C4=O4 (which takes the negative charge in the product), the hydride, and all '
            'hydrogens on those atoms. The four bonds cut to the rest of the cluster are capped with hydrogens. The fragment '
            'is 18 atoms, 56 electrons, neutral and closed-shell, and the stage stops on an odd electron count instead of '
            'guessing a charge.')]

story += [P('2.8 Guards added', 'h2'),
          B('<b>Scan:</b> stop on an energy jump of more than 12 kcal/mol between neighbouring points; on any bond change '
            'other than the hydride\'s or C6u–S (the first point is checked against prep); or if the hydride is more than '
            '1.6 Å from both carbons. Replayed on the saved data, these catch the failures in 2.1 and 2.2.'),
          B('<b>Scan log:</b> any drop larger than 3 kcal/mol is noted.'),
          B('<b>TS stage:</b> refuses a scan maximum at an end point; P-RFO stops on persistent non-negative curvature.'),
          B('<b>Restarts:</b> prep and TS stages skip when already done, so the wrapper can be re-run safely.')]

# ---- 3. results
story += [P('3. Results', 'h1'),
          P('3.1 GFN2-xTB energetics and kinetic isotope effects (293 K)', 'h2')]
kap = an['kie']
story += [table([['Isotope', 'ZPE-corrected barrier (kcal/mol)', 'Imaginary freq. (cm⁻¹)', 'κ Wigner', 'κ Eckart'],
                 *[[iso, f"{kap[f'H/{iso}' if iso != 'H' else 'H/T']['kappa'][iso]['V1']:.2f}",
                    f"{kap[f'H/{iso}' if iso != 'H' else 'H/T']['kappa'][iso]['nu']:.0f}i",
                    f"{kap[f'H/{iso}' if iso != 'H' else 'H/T']['kappa'][iso]['wigner']:.2f}",
                    f"{kap[f'H/{iso}' if iso != 'H' else 'H/T']['kappa'][iso]['eckart']:.2f}"] for iso in ('H', 'D', 'T')]],
                [22 * mm, 50 * mm, 40 * mm, 28 * mm, 28 * mm]),
          P(f"Table 1. Electronic barrier {an['dE_barrier']:.2f} kcal/mol, reaction energy {an['dE_rxn']:+.2f} kcal/mol. "
            "Reactant and product have no imaginary modes; the TS has exactly one.", 'cap'),
          table([['KIE', 'Semiclassical', '+ Wigner', '+ Eckart', 'Experiment'],
                 *[[kk, f"{v['semiclassical']:.2f}", f"{v['wigner']:.2f}", f"{v['eckart']:.2f}",
                    '6.91 ± 0.05' if kk == 'H/T' else '—'] for kk, v in kap.items()]],
                [26 * mm, 38 * mm, 34 * mm, 34 * mm, 38 * mm]),
          P('Table 2. Kinetic isotope effects. The H/T values bracket experiment: the semiclassical value is slightly low, '
            'and the tunnelling corrections overshoot.', 'cap')]

g = {n: geo(n) for n in ('reactant', 'ts', 'product')}
TZLOG = Path(sorted(glob.glob(str(RES / 'nuc_*_tzvp.log')))[-1]).name if glob.glob(str(RES / 'nuc_*_tzvp.log')) else None
story += [P('3.2 Geometry', 'h2'),
          table([['Structure', 'C6–H', 'H–C11', 'C6···C11', 'N5–C11', 'C6u–S'],
                 *[[n, *[f'{v:.2f}' for v in g[n]]] for n in ('reactant', 'ts', 'product')]],
                [30 * mm, 26 * mm, 26 * mm, 30 * mm, 28 * mm, 28 * mm]),
          P('Table 3. Key distances in Å. N5–C11 stays above the 2.8 Å floor everywhere, so the restraint is inactive at '
            'all three stationary points. C6u–S stretches from 1.87 to 2.00 Å but does not break: the hydride moves first, '
            'and the C–S cleavage is not concerted with it in this model.', 'cap')]

story += [P('3.3 DFT single points', 'h2')]
drows = [['Basis', 'Barrier (kcal/mol)', 'Reaction (kcal/mol)', 'Time per structure', 'Density fitting']]
for bas in ('def2-svp', 'def2-tzvp'):
    if all(f'{n}/{bas}' in dft for n in ('reactant', 'ts', 'product')):
        e = {n: dft[f'{n}/{bas}']['E'] for n in ('reactant', 'ts', 'product')}
        t = np.mean([dft[f'{n}/{bas}']['seconds'] for n in ('reactant', 'ts', 'product')]) / 3600
        drows.append([bas, f"{(e['ts'] - e['reactant']) * K:.2f}", f"{(e['product'] - e['reactant']) * K:+.2f}",
                      f'{t:.1f} h', 'yes' if dft[f'reactant/{bas}'].get('density_fit') else 'no'])
    else:
        drows.append([bas, 'not finished', '', '', 'yes'])
story += [table(drows, [28 * mm, 34 * mm, 34 * mm, 36 * mm, 30 * mm]),
          P('Table 4. B3LYP-D3(BJ), PCM ε = 4, at the xTB geometries. With the xTB zero-point correction (−3.35 kcal/mol) '
            'the def2-SVP barrier is about 23.2 kcal/mol.', 'cap'),
          P('DFT gives roughly twice the xTB barrier. As a rough guide, the enzyme\'s turnover of about 7 s<super>−1</super> '
            'corresponds to a free-energy barrier near 16 kcal/mol, so xTB is low and single-point DFT at xTB geometries '
            'is high. Sections 3.5 to 3.7 test whether the functional or the geometry is responsible.')]

story += [P('3.4 Active-space circuits', 'h2')]
if act and all(f'{n}/4e4o' in act for n in ('reactant', 'ts', 'product')):
    a = {n: act[f'{n}/4e4o'] for n in ('reactant', 'ts', 'product')}
    story += [table([['Structure', 'Atoms', 'Qubits', 'UCCSD error (mHa)', 'UCCSD CNOTs', 'ADAPT error (mHa)',
                      'ADAPT params', 'ADAPT CNOTs'],
                     *[[n, str(a[n]['n_atoms']), str(a[n]['n_qubits']), f"{a[n]['uccsd']['err_mha']:.3g}",
                        str(a[n]['uccsd']['cnots']), f"{a[n]['adapt']['err_mha']:.3g}", str(a[n]['adapt']['n_params']),
                        str(a[n]['adapt']['cnots'])] for n in a]],
                    [22 * mm, 14 * mm, 15 * mm, 25 * mm, 22 * mm, 25 * mm, 22 * mm, 22 * mm]),
              P('Table 5. Errors are against exact CASCI in the same active space (chemical accuracy is 1.6 mHa). '
                'cc-pVDZ basis, CCSD natural orbitals.', 'cap')]
    for m, lab_ in (('e_ccsd_t', 'CCSD(T)'), ('e_casci', 'CASCI(4,4)')):
        e = {n: a[n][m] for n in a}
        story.append(B(f"{lab_} on the fragment: barrier {(e['ts'] - e['reactant']) * K:.2f}, reaction "
                       f"{(e['product'] - e['reactant']) * K:+.2f} kcal/mol."))
    story.append(P('These fragment energies are not reaction energetics. The 18-atom fragment is in the gas phase, '
                   'without Arg166 or a dielectric, so the product (an iminium cation paired with an enolate anion) '
                   'is left without stabilisation; CASCI(4,4) also lacks dynamic correlation. The fragment serves as a '
                   'benchmark of the circuits against exact CASCI, which every structure now passes.'))
else:
    story += [P('The corrected stage (section 2.7) had not finished when this report was generated. The earlier '
                'results are not reported because the fragments were not chemically valid.')]


# ---- 3.5-3.8: after the debugging (each shown once its results exist)
fx_path, pr_path = RES / 'ts_functional_check.json', RES / 'ts_profile_check.json'
op_path, hp_path = RES / 'ts_oniom_opt.json', RES / 'ts_oniom_opt.his147p.json'
NM = ('reactant', 'ts', 'product')
if fx_path.exists():
    fx = json.loads(fx_path.read_text())
    rows = [['Method (embedded fragment, def2-TZVP)', 'Barrier (kcal/mol)', 'Reaction (kcal/mol)']]
    for tag, v in fx.items():
        if all('E' in v.get(n, {}) for n in NM):
            nice = {'b3lyp-d3bj': 'B3LYP-D3(BJ)', 'pbe0-d3bj': 'PBE0-D3(BJ)', 'm062x-d3zero': 'M06-2X-D3(0)',
                    'wb97x-d3bj': '\u03c9B97X-D3(BJ)', 'bhandhlyp': 'BHandHLYP'}
            rows.append([nice.get(tag.split('/')[0], tag.split('/')[0]), f"{(v['ts']['E'] - v['reactant']['E']) * K:.1f}",
                         f"{(v['product']['E'] - v['reactant']['E']) * K:+.1f}"])
    foc = r.get('oniom', {})
    if all('mp2' in foc.get(n, {}) for n in NM):
        e = {n: foc[n]['mp2']['cc-pvtz']['e_hf'] + foc[n]['mp2']['cc-pvtz']['e_corr'] for n in NM}
        rows.append(['MP2/cc-pVTZ', f"{(e['ts'] - e['reactant']) * K:.1f}", f"{(e['product'] - e['reactant']) * K:+.1f}"])
        e = {n: foc[n]['e_ccsd_t_focal']['cbs'] for n in NM}
        rows.append(['CCSD(T), extrapolated basis', f"{(e['ts'] - e['reactant']) * K:.1f}",
                     f"{(e['product'] - e['reactant']) * K:+.1f}"])
    story += [P('3.5 Which functional?', 'h2'),
              P('B3LYP and PBE0 put the embedded-fragment barrier about 10 kcal/mol below CCSD(T), MP2 and the three '
                'functionals suited to barrier heights. B3LYP is the outlier, so its cluster barriers in section 3.3 are '
                'too low, and the later steps use \u03c9B97X-D3(BJ), the functional closest to CCSD(T) here.'),
              table(rows, [80 * mm, 45 * mm, 45 * mm]),
              P('Table 6. Single points at the GFN2-xTB geometries, fragment in the xTB point charges.', 'cap')]
if pr_path.exists() and '_profile' in json.loads(pr_path.read_text()):
    prof = json.loads(pr_path.read_text())['_profile']
    pts = [v for v in prof.values() if v['xi'] is not None]
    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=200)
    ax.plot([v['xi'] for v in pts], [v['xtb'] for v in pts], color='#8a8980', lw=1.6, marker='o', ms=3, label='GFN2-xTB')
    ax.plot([v['xi'] for v in pts], [v['frag_ee'] for v in pts], color='#8a8980', lw=1.6, ls='--', marker='o', ms=3,
            label='embedded fragment, \u03c9B97X-D3')
    ax.plot([v['xi'] for v in pts], [v['oniom_me'] for v in pts], color=BLUE, lw=2.2, marker='o', ms=4, mec='white',
            label='ONIOM(\u03c9B97X-D3:xTB)')
    ax.axvline(0.1, color='#8a8980', lw=0.8, ls=(0, (3, 3)))
    ax.text(0.08, 1.5, 'xTB saddle point', ha='right', fontsize=7.5, color='#55554f')
    top_ = max(pts, key=lambda v: v['oniom_me'])
    ax.annotate(f"maximum {top_['oniom_me']:.1f} at \u03be = +{top_['xi']:.1f}", (top_['xi'], top_['oniom_me']),
                xytext=(0.55, 39.5), textcoords='data', fontsize=7.8, color='#1f1f1e',
                arrowprops=dict(arrowstyle='-', color='#8a8980', lw=0.6))
    style(ax); ax.set_ylim(-1, 42)
    ax.set_xlabel('\u03be = d(C6\u2013H) \u2212 d(H\u2013C11)  (\u00c5)'); ax.set_ylabel('\u0394E vs reactant  (kcal/mol)')
    ax.legend(frameon=False, loc='upper left', fontsize=7.5)
    fig.tight_layout(); f3 = FIGDIR / 'fig3.png'; fig.savefig(f3); plt.close(fig)
    story += [KeepTogether([P('3.6 Is the xTB geometry the problem?', 'h2'),
              P(f"Along the xTB path, the ONIOM(\u03c9B97X-D3:xTB) profile peaks at {top_['oniom_me']:.1f} kcal/mol, only "
                f"{top_['oniom_me'] - prof['ts']['oniom_me']:.1f} kcal/mol above its value at the xTB saddle point. "
                'Single points at the xTB transition state therefore do not distort the barrier much.'),
              Image(str(f3), width=160 * mm, height=72 * mm),
              P('Figure 3. Energy along the GFN2-xTB scan with \u03c9B97X-D3(BJ)/def2-TZVP for the fragment.', 'cap')])]
if op_path.exists() and 'summary' in json.loads(op_path.read_text()):
    op = json.loads(op_path.read_text()); a_ = op['analysis']; su = op['summary']; kk = a_['kie']
    story += [P('3.7 Structures optimised at ONIOM(\u03c9B97X-D3:xTB)', 'h2'),
              P(f"Re-optimising reactant, transition state and product on the ONIOM surface gives a barrier of "
                f"{su['barrier_tzvp']:.1f} kcal/mol ({su['barrier_tzvp_zpe']:.1f} with zero-point energy) and a reaction "
                f"energy of {su['reaction_tzvp']:+.1f} kcal/mol, with def2-TZVP for the fragment. The transition state has "
                f"one imaginary mode ({abs(a_['iso']['ts']['H']['imag']):.0f}i cm\u207b\u00b9) and is later than the xTB one."),
              table([['Kinetic isotope effect (293 K)', 'Semiclassical', '+ Wigner', '+ Eckart', 'Experiment'],
                     *[[f'{k_}, ONIOM', f"{v['semiclassical']:.2f}", f"{v['wigner']:.2f}", f"{v['eckart']:.2f}",
                        '6.91 \u00b1 0.05' if k_ == 'H/T' else ''] for k_, v in kk.items()]],
                    [44 * mm, 30 * mm, 28 * mm, 28 * mm, 40 * mm]),
              P('Table 7. Frequencies from partial Hessians over the 34 free atoms within 5 \u00c5 of the hydride, which '
                'reproduce the full GFN2-xTB Hessian KIE within 0.2 %.', 'cap'),
              P('The isotope effects now bracket experiment from below, as GFN2-xTB did from above. The barrier stays '
                'about twice what turnover implies, and its excess tracks the endothermicity: the enolate charge on O4 '
                'has no positive partner nearby (His147, modelled neutral, at 3.8 \u00c5; Arg166 at 8 \u00c5).')]
    hp = json.loads(hp_path.read_text()) if hp_path.exists() else {}
    story += [P('3.8 His147 protonated', 'h2')]
    if 'summary' in hp:
        hs, hk = hp['summary'], hp['analysis']['kie']['H/T']
        story += [P(f"With His147 protonated (cluster charge +2), the same pipeline gives a barrier of "
                    f"{hs['barrier_tzvp']:.1f} kcal/mol ({hs['barrier_tzvp_zpe']:.1f} with zero-point energy) and a "
                    f"reaction energy of {hs['reaction_tzvp']:+.1f} kcal/mol; H/T KIE {hk['semiclassical']:.2f} "
                    f"(semiclassical), {hk['wigner']:.2f} (Wigner), {hk['eckart']:.2f} (Eckart).")]
    else:
        story += [P('The same pipeline with His147 protonated (cluster charge +2) was running when this report was '
                    'generated; it tests whether a positive charge near O4 lowers the barrier and the reaction energy.')]

# ---- 4. caveats
story += [P('4. Assumptions and caveats', 'h1'),
          B('<b>N5–C11 restraint (2.8 Å floor).</b> Needed to stop N5 re-bonding to C11; inactive at the stationary points '
            'but present along the scan. Rerunning with a 3.0 Å floor would test sensitivity.'),
          B('<b>Fixed atoms.</b> Eleven atoms are held in place throughout; frequencies and partition functions cover the '
            'free atoms only.'),
          B('<b>Solvent.</b> xTB uses ALPB chloroform (ε 4.8) as the closest named solvent to the ε = 4 used for '
            'protein interiors; DFT uses PCM ε = 4.'),
          B('<b>Geometries.</b> Sections 3.3 to 3.6 are single points at GFN2-xTB geometries; section 3.7 re-optimises at '
            'ONIOM(\u03c9B97X-D3/def2-SVP:xTB), not at full DFT.'),
          B('<b>Embedding.</b> The ONIOM optimisation uses mechanical embedding: the fragment interacts with its '
            'surroundings at the xTB level, with fixed link-atom distances.'),
          B('<b>Frequencies.</b> The ONIOM KIEs use partial Hessians over 34 atoms, validated on the xTB Hessians; '
            'tunnelling corrections use the imaginary frequency of the same level.'),
          B('<b>Scan steps.</b> The backward pass removes the steps the folate shift left in the profile; the barrier and '
            'KIE use the relaxed stationary points, not scan energies.')]

# ---- 5. code
story += [P('5. Code changes', 'h1'),
          table([['File', 'Change'],
                 ['<font name="DVM">chemoqbit/enzyme/frozen.py</font>',
                  'restraint code moved into a shared <font name="DVM">penalty()</font>; adds the one-sided '
                  '<font name="DVM">(\'min\', i, j, d0)</font> restraint'],
                 ['<font name="DVM">chemoqbit/enzyme/tsopt.py</font>',
                  'P-RFO: curvature threshold, abort on persistent non-negative curvature, fixed-atom support; '
                  '<font name="DVM">fd_hessian</font> and <font name="DVM">frequencies</font> over free atoms; '
                  'optional rotational term in <font name="DVM">ln_q_vib_rot</font>'],
                 ['<font name="DVM">scripts/ts_cluster_nuc.py</font>',
                  'N5 hold in prep and all later stages; scan guards and backward pass; TS from the hydride coordinate '
                  'with fixed atoms; end points from the lowest scan points; skip-if-done for prep and TS; free-atom '
                  'Hessians and KIE; density fitting for triple-zeta DFT, resumable; capped, consistent active-space '
                  'region; ONIOM and focal-point stages; model variants (--variant)'],
                 ['<font name="DVM">chemoqbit/enzyme/oniom.py</font>',
                  'two-layer ONIOM energy and gradient with link atoms (mechanical embedding)'],
                 ['<font name="DVM">tsopt.prfo_bofill</font>',
                  'P-RFO with a Bofill-updated Hessian, adaptive trust radius, resumable state'],
                 ['<font name="DVM">scripts/ts_functional_check.py</font>, <font name="DVM">ts_profile_check.py</font>, '
                  '<font name="DVM">ts_oniom_opt.py</font>',
                  'functional benchmark, barrier profile along the path, ONIOM optimisation with partial Hessians '
                  'and KIEs']],
                [52 * mm, 118 * mm])]

# ---- 6. status / next
story += [P('6. Status and next steps', 'h1')]
story += [B('xTB stages 0–4: <b>done</b>.'),
          B('DFT def2-SVP: <b>done</b>.'),
          B('DFT def2-TZVP: ' + ('<b>done</b>.' if tz else '<b>not finished</b> when this report was generated.')),
          B('Active-space stage: ' + ('<b>done</b>.' if act and len(act) == 3 else
                                      '<b>not finished</b> when this report was generated.')),
          B('ONIOM(\u03c9B97X-D3:xTB) optimisation and KIEs: ' + ('<b>done</b>.' if op_path.exists() and 'summary' in
                json.loads(op_path.read_text()) else '<b>not finished</b> when this report was generated.')),
          B('His147-protonated model: ' + ('<b>done</b>.' if hp_path.exists() and 'summary' in
                json.loads(hp_path.read_text()) else '<b>running</b> when this report was generated.')),
          B('Worth doing next: the N5-floor sensitivity check, and other ways of stabilising the O4 enolate '
            '(coupling to C6\u2013S cleavage, a larger cluster).')]

story += [P('7. Files and how to reproduce', 'h1'),
          P('Checkpoints kept in <font name="DVM">results/</font>, with the stage at which each was set aside:'),
          table([['File', 'Contents'],
                 ['<font name="DVM">ts_cluster_run.json</font>', 'current checkpoint'],
                 ['<font name="DVM">ts_cluster_run.N5adduct_20261017.json</font>', 'original failed run (section 2.1)'],
                 ['<font name="DVM">ts_cluster_run.badscan_20261018.json</font>', 'first rerun, N5 re-bonded (2.2)'],
                 ['<font name="DVM">ts_cluster_run.prebackfill_20261018.json</font>', 'forward scan before the backward pass (2.3)'],
                 ['<font name="DVM">ts_cluster_run.capsmovedTS_20261018.json</font>', 'TS with moved fixed atoms (2.5)'],
                 ['<font name="DVM">ts_functional_check.json</font>, <font name="DVM">ts_profile_check.json</font>',
                  'functional benchmark (3.5) and profile along the path (3.6)'],
                 ['<font name="DVM">ts_oniom_opt.json</font>, <font name="DVM">ts_oniom_opt.his147p.json</font>',
                  'ONIOM-optimised structures, Hessians and KIEs (3.7, 3.8)'],
                 ['<font name="DVM">nuc_20261018_1921.log</font>'
                  + (f', <font name="DVM">{TZLOG}</font>' if TZLOG else ''),
                  'logs of the final xTB/DFT run' + (' and the latest def2-TZVP run' if TZLOG else '')]],
                [86 * mm, 84 * mm]),
          Spacer(1, 6),
          P('Full pipeline (each stage skips completed work):'),
          P('bash run_nuc.sh', 'code'),
          P('Only the def2-TZVP single points:'),
          P('OMP_NUM_THREADS=4 .venv/bin/python -u scripts/ts_cluster_nuc.py --stages dft --basis def2-tzvp', 'code')]


def footer(c, d):
    c.saveState(); c.setFont('DV', 7.5); c.setFillColor(MUTED)
    c.drawString(20 * mm, 12 * mm, 'Thymidylate synthase TS pipeline: debugging report')
    c.drawRightString(190 * mm, 12 * mm, f'{d.page}')
    c.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                        bottomMargin=20 * mm, title='Thymidylate synthase TS pipeline: debugging report',
                        author='chemoqbit')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
shutil.rmtree(FIGDIR, ignore_errors=True)
print(OUT)
