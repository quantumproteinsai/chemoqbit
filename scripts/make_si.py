"""Supporting information for the two thymidylate synthase preprints, built from
the result files: results/SI_manuscript1.pdf, results/SI_manuscript2.pdf, and
the coordinates as .xyz files in results/si/. Re-run after new results (the
His147 variant appears once its checkpoint is complete).

    python scripts/make_si.py        # needs reportlab and matplotlib (fonts)
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
K = 627.509474
NAMES = ('reactant', 'ts', 'product')
LONG = {'reactant': 'reactant', 'ts': 'transition state', 'product': 'product'}

fdir = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
for name, f in (('DV', 'DejaVuSans.ttf'), ('DV-B', 'DejaVuSans-Bold.ttf'), ('DVM', 'DejaVuSansMono.ttf')):
    pdfmetrics.registerFont(TTFont(name, str(fdir / f)))
INK, INK2, RULE = colors.HexColor('#1f1f1e'), colors.HexColor('#55554f'), colors.HexColor('#d9d8d2')
S = dict(t=ParagraphStyle('t', fontName='DV-B', fontSize=15, leading=19, textColor=INK, spaceAfter=6),
         h=ParagraphStyle('h', fontName='DV-B', fontSize=11.5, leading=15, textColor=INK, spaceBefore=10, spaceAfter=4),
         p=ParagraphStyle('p', fontName='DV', fontSize=8.8, leading=12, textColor=INK, spaceAfter=4),
         c=ParagraphStyle('c', fontName='DV', fontSize=7.6, leading=9.6, textColor=INK),
         cb=ParagraphStyle('cb', fontName='DV-B', fontSize=7.6, leading=9.6, textColor=INK),
         m=ParagraphStyle('m', fontName='DVM', fontSize=6.6, leading=7.9, textColor=INK))
P = lambda t, s='p': Paragraph(t, S[s])


def table(rows, widths=None):
    data = [[Paragraph(str(c), S['cb' if i == 0 else 'c']) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([('LINEBELOW', (0, 0), (-1, 0), 0.7, INK2), ('LINEBELOW', (0, 1), (-1, -1), 0.25, RULE),
                           ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 2),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 2), ('LEFTPADDING', (0, 0), (-1, -1), 3),
                           ('RIGHTPADDING', (0, 0), (-1, -1), 3)]))
    return t


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


import ts_cluster_nuc as SN
from chemoqbit.enzyme import tsopt as T
from chemoqbit.enzyme.oniom import CAP

r = load('ts_cluster_run.json'); p = r['prep']; sym, lab, k = p['symbols'], p['labels'], p['key']
op, fx, prof = load('ts_oniom_opt.json'), load('ts_functional_check.json'), load('ts_profile_check.json')
fwd = load('ts_cluster_run.prebackfill_20261018.json')
hp, hpr = load('ts_oniom_opt.his147p.json'), load('ts_cluster_run.his147p.json')
atoms, cuts = SN.active_region(r)
free = SN._free(sym, p['frozen'])
XYZ = RES / 'si'; XYZ.mkdir(exist_ok=True)


def xyz_block(symbols, X, title):
    return f'{len(symbols)}\n{title}\n' + '\n'.join(f'{s:2s} {x:12.6f} {y:12.6f} {z:12.6f}' for s, (x, y, z) in zip(symbols, X)) + '\n'


structures = [(f'xtb_{n}', f'GFN2-xTB {LONG[n]}', sym, r[n]['coords']) for n in NAMES]
if op:
    structures += [(f'oniom_{n}', f'ONIOM(wB97X-D3/def2-SVP:GFN2-xTB) {LONG[n]}', sym, op[n]['X']) for n in NAMES]
if hp and 'summary' in hp:
    structures += [(f'his147p_oniom_{n}', f'His147-protonated, ONIOM {LONG[n]}', hpr['prep']['symbols'], hp[n]['X'])
                   for n in NAMES]
for key, title, s_, X in structures:
    (XYZ / f'{key}.xyz').write_text(xyz_block(s_, X, title))


def coords_section(story, start):
    story.append(P(f'{start} Cartesian coordinates (Å)', 'h'))
    story.append(P('Also provided as .xyz files. Fixed atoms are those listed in the cluster composition; '
                   'all coordinates refer to the same frame as PDB 1TLS.'))
    for key, title, s_, X in structures:
        story.append(P(title, 'p'))
        story.append(Preformatted(xyz_block(s_, X, key), S['m']))


def energies_rows():
    rows = [['Method', 'Reactant (Eh)', 'TS (Eh)', 'Product (Eh)', 'Barrier (kcal/mol)']]
    add = lambda m, e: rows.append([m, *[f'{e[n]:.6f}' for n in NAMES], f"{(e['ts'] - e['reactant']) * K:.2f}"])
    add('GFN2-xTB, cluster (ALPB)', {n: r[n]['E'] for n in NAMES})
    for b in ('def2-svp', 'def2-tzvp'):
        if all(f'{n}/{b}' in r['dft'] for n in NAMES):
            add(f'B3LYP-D3(BJ)/{b}, cluster, C-PCM', {n: r['dft'][f'{n}/{b}']['E'] for n in NAMES})
    o = r.get('oniom', {})
    if all(n in o for n in NAMES):
        add('B3LYP-D3(BJ)/def2-SVP, fragment in charges', {n: o[n]['e_dft_model'] for n in NAMES})
        add('HF/cc-pVDZ, fragment in charges', {n: o[n]['e_hf'] for n in NAMES})
        add('CCSD(T)/cc-pVDZ, fragment in charges', {n: o[n]['e_ccsd_t'] for n in NAMES})
        add('CASCI(4,4)/cc-pVDZ, fragment in charges', {n: o[n]['e_casci'] for n in NAMES})
        if all('mp2' in o[n] for n in NAMES):
            for b in ('cc-pvdz', 'cc-pvtz'):
                add(f'DF-MP2/{b}, fragment in charges', {n: o[n]['mp2'][b]['e_hf'] + o[n]['mp2'][b]['e_corr'] for n in NAMES})
            add('CCSD(T), extrapolated basis, fragment in charges', {n: o[n]['e_ccsd_t_focal']['cbs'] for n in NAMES})
    if fx:
        for tag, v in fx.items():
            if all('E' in v.get(n, {}) for n in NAMES):
                add(f'{tag}, fragment in charges', {n: v[n]['E'] for n in NAMES})
    if op and 'sp' in op:
        add('ONIOM(wB97X-D3(BJ)/def2-TZVP:xTB) at ONIOM geometries', {n: op['sp'][n]['E'] for n in NAMES})
    return rows


def build(path, title, story):
    def footer(c, d):
        c.saveState(); c.setFont('DV', 7); c.setFillColor(INK2)
        c.drawString(18 * mm, 10 * mm, title); c.drawRightString(192 * mm, 10 * mm, f'S{d.page}'); c.restoreState()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=title, author='Bertrand Mercier des Rochettes')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(path)


# ---------------- manuscript 1
s1 = [P('Supporting information: Embedded active-space quantum circuits for an enzymatic hydride transfer', 't'),
      P('Bertrand Mercier des Rochettes, Quantum Proteins AI'),
      P('S1 Fragment composition', 'h'),
      P(f'{len(atoms)} cluster atoms and {len(cuts)} link hydrogens; 56 electrons, charge 0, closed-shell.'),
      table([['Cluster atoms in the fragment'], [', '.join(lab[i] for i in atoms)]], [174 * mm]),
      Spacer(1, 4),
      table([['Cut bond (inside – outside)', 'Link H distance from inside atom (Å)'],
             *[[f'{lab[a]} – {lab[b]}', f'{CAP[sym[a]]:.2f}'] for a, b in cuts]], [110 * mm, 64 * mm]),
      P('S2 Circuit results', 'h')]
for key, title_ in (('oniom', 'fragment in the GFN2-xTB point charges (Table 1)'), ('active', 'capped fragment in the gas phase')):
    d = r.get(key, {}); rows = [['Structure', 'E(CASCI) (Eh)', 'Ansatz', 'E(VQE) (Eh)', 'Error (mHa)', 'Fixed-point error (mHa)',
                                 'Parameters', 'CNOTs']]
    for n in NAMES:
        v = d.get(n) if key == 'oniom' else d.get(f'{n}/4e4o')
        if not v:
            continue
        for a in ('uccsd', 'adapt'):
            rows.append([LONG[n], f"{v['e_casci']:.6f}", a.upper() if a == 'uccsd' else 'ADAPT', f"{v[a]['e_vqe']:.6f}",
                         f"{v[a]['err_mha']:.4f}", f"{(v[a]['e_vqe_fixed'] - v['e_casci']) * 1e3:.4f}",
                         str(v[a]['n_params']), str(v[a]['cnots'])])
    s1 += [P(title_.capitalize()), table(rows, [24 * mm, 25 * mm, 15 * mm, 25 * mm, 18 * mm, 25 * mm, 18 * mm, 18 * mm]), Spacer(1, 6)]
s1 += [P('Fixed-point energies: 18-bit amplitude words with 16 fractional bits.'),
       P('S3 Energies of all methods', 'h'), table(energies_rows(), [74 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm]),
       PageBreak()]
coords_section(s1, 'S4')
build(RES / 'SI_manuscript1.pdf', 'SI – Embedded active-space circuits, thymidylate synthase', s1)

# ---------------- manuscript 2
from chemoqbit.enzyme.cluster import FRAGMENTS, WATERS
s2 = [P('Supporting information: Hydride transfer in thymidylate synthase from a crystal-derived cluster model', 't'),
      P('Bertrand Mercier des Rochettes, Quantum Proteins AI'),
      P('S1 Composition of the cluster', 'h'),
      table([['Fragment', 'PDB residue', 'Atoms kept'],
             *[[n, f'{res} {num}', ', '.join(a)] for n, res, num, a in FRAGMENTS],
             ['waters', 'HOH', ', '.join(str(w) for w in WATERS)]], [32 * mm, 30 * mm, 112 * mm]),
      Spacer(1, 4),
      table([['Fixed atoms (11)'], [', '.join(lab[i] for i in p['frozen'])]], [174 * mm]),
      P(f"Total {len(sym)} atoms, charge {p['charge']}. Restraint: N5–C11 ≥ 2.8 Å (flat-bottom, k = 20 Eh/Å²)."),
      P('S2 GFN2-xTB scan', 'h'),
      P('Energies in kcal/mol relative to the relaxed reactant; forward = before the backward pass.')]
E_R = r['reactant']['E']; fw = {q['xi']: q['E'] for q in (fwd or {}).get('scan', [])}
rows = [['ξ (Å)', 'Forward', 'Final', 'C6–H (Å)', 'H–C11 (Å)', 'C6···C11 (Å)', 'C6–S (Å)']]
for q in r['scan']:
    rows.append([f"{q['xi']:+.1f}", f"{(fw[q['xi']] - E_R) * K:.2f}" if q['xi'] in fw else '', f"{(q['E'] - E_R) * K:.2f}",
                 f"{q['CdH']:.2f}", f"{q['HC11']:.2f}", f"{q['DAD']:.2f}", f"{q['C6uS']:.2f}"])
s2 += [table(rows, [20 * mm, 22 * mm, 22 * mm, 26 * mm, 26 * mm, 30 * mm, 26 * mm])]

s2 += [P('S3 Harmonic frequencies', 'h'),
       P('Imaginary mode and the ten lowest real frequencies (cm⁻¹) for each isotopologue (the transferring hydrogen as H, D or T). '
         'Full lists are in the dataset.')]
masses = {i: [T.AMU[s] for s in sym] for i in ('H', 'D', 'T')}
for i_ in ('D', 'T'):
    masses[i_][k['Hd']] = T.AMU[i_]
levels = [('GFN2-xTB', {n: (r[n]['coords'], r[n]['hessian']) for n in NAMES}, free)]
if op:
    levels.append(('ONIOM', {n: (op[n]['X'], op['hess'][n]['H']) for n in NAMES}, op['config']['hessian_region']))
rows = [['Level', 'Structure', 'Isotope', 'Imaginary', 'Ten lowest real']]
for lev, data, region in levels:
    for n in NAMES:
        X, H = data[n]
        for i_ in ('H', 'D', 'T'):
            nu = T.frequencies(np.array(X), np.array(H), masses[i_], free=region)
            im = nu[nu < -50]; re = nu[nu > 0][:10]
            rows.append([lev, LONG[n], i_, f'{abs(im.at(0) if hasattr(im, "at") else im[0]):.0f}i' if len(im) else '–',
                         ', '.join(f'{v:.0f}' for v in re)])
s2 += [table(rows, [20 * mm, 26 * mm, 14 * mm, 20 * mm, 94 * mm])]

s2 += [P('S4 Partial-Hessian validation', 'h'),
       P('H/T effect (transition-state theory) from sub-blocks of the full GFN2-xTB Hessians over free atoms within a radius '
         'of the hydride at the xTB transition state.')]
pos = {a: i for i, a in enumerate(free)}; Xt = np.array(r['ts']['coords'])


def kie_region(region):
    idx = np.array([3 * pos[a] + c for a in region for c in range(3)])
    out = {}
    for n in ('reactant', 'ts'):
        X = np.array(r[n]['coords']); H = np.array(r[n]['hessian'])[np.ix_(idx, idx)]
        for i_ in ('H', 'T'):
            nu = T.frequencies(X, H, masses[i_], free=region)
            out[(n, i_)] = T.ln_q_vib_rot(X, nu[nu > 0], masses[i_], 293.15, rot=False)
    return float(np.exp((out[('ts', 'H')] - out[('reactant', 'H')]) - (out[('ts', 'T')] - out[('reactant', 'T')])))


rows = [['Region', 'Atoms', 'H/T (TST)']]
for lab_, reg in [('all free atoms', free)] + [(f'within {rad:.0f} Å', [a for a in free if np.linalg.norm(Xt[a] - Xt[k['Hd']]) < rad])
                                               for rad in (3.0, 4.0, 5.0, 6.0)]:
    rows.append([lab_, str(len(reg)), f'{kie_region(reg):.3f}'])
s2 += [table(rows, [60 * mm, 30 * mm, 40 * mm])]

s2 += [P('S5 Energies of all methods', 'h'), table(energies_rows(), [74 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm])]
s2 += [P('S6 Isotope-effect ingredients', 'h'),
       P('Zero-point energies (kcal/mol), zero-point-corrected forward barrier V1 (kcal/mol), imaginary frequency ν (cm⁻¹) '
         'and tunnelling factors at 293.15 K.')]
rows = [['Level', 'Isotope', 'ZPE reactant', 'ZPE TS', 'V1', 'ν', 'κ Wigner', 'κ Eckart']]
for lev, an in (('GFN2-xTB', r['analysis']), ('ONIOM', op['analysis'] if op else None)):
    if not an:
        continue
    for i_ in ('H', 'D', 'T'):
        kap = an['kie']['H/D' if i_ == 'D' else 'H/T']['kappa'][i_]
        rows.append([lev, i_, f"{an['iso']['reactant'][i_]['zpe'] * K:.2f}", f"{an['iso']['ts'][i_]['zpe'] * K:.2f}",
                     f"{kap['V1']:.2f}", f"{kap['nu']:.0f}", f"{kap['wigner']:.3f}", f"{kap['eckart']:.3f}"])
s2 += [table(rows, [22 * mm, 16 * mm, 24 * mm, 22 * mm, 18 * mm, 16 * mm, 22 * mm, 22 * mm])]
if hp and 'summary' in hp:
    s2 += [P('S7 His147-protonated model', 'h'),
           P(f"Barrier {hp['summary']['barrier_tzvp']:.2f} kcal/mol ({hp['summary']['barrier_tzvp_zpe']:.2f} with ZPE), "
             f"reaction {hp['summary']['reaction_tzvp']:+.2f} kcal/mol at ONIOM(ωB97X-D3/def2-TZVP:xTB).")]
s2 += [PageBreak()]
coords_section(s2, 'S8' if hp and 'summary' in hp else 'S7')
build(RES / 'SI_manuscript2.pdf', 'SI – Thymidylate synthase hydride transfer, cluster model', s2)
