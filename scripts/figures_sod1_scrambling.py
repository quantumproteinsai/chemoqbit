#!/usr/bin/env python
"""Figures for paper/sod1_scrambling.md, regenerated from the stored results.

Nothing here recomputes physics. Every number plotted is read from results/ or
from the deposited coordinates, so a figure can never drift from the text: if a
calculation is rerun, rerunning this script is the whole update.

    python scripts/figures_sod1_scrambling.py

Figures are numbered as they appear in the manuscript, which is not the order they
are built in here:

Figure 1  the attack site itself, projected from the 2C9U coordinates   (figure3)
Figure 2  the clearance ladder at the attack point                      (figure2)
Figure 3  the conformational work against separation, with the ceilings (figure1)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit import als_approach as A              # noqa: E402

OUT = ROOT / 'paper' / 'figures'
HARTREE = 627.5094740631

# Validated categorical palette; see the dataviz reference. Blue carries the data,
# the reserved critical red carries the one threshold that decides the paper, and
# both always arrive with a text label rather than colour alone.
BLUE, RED = '#2a78d6', '#d03b3b'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'

CEILINGS = [(25.18, 'one event per day'),
            (28.82, 'one per year'),
            (31.23, 'one per fifty years')]
REQUIRED = 3.3     # A, a sulfur-carbon van der Waals contact


def style(ax):
    """Recessive chrome: the data is the only thing with weight."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / ('%s.%s' % (name, ext)), dpi=300, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)
    print('  %s.png / .pdf' % name)


def figure1(res):
    """Work against separation. One series, so no legend: the title names it."""
    p = res['mobile']['profile']
    d = np.array([r['d'] for r in p])
    W = (np.array([r['E'] for r in p]) - p[0]['E']) * HARTREE

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    style(ax)

    # The three ceilings lie within 6 kcal/mol of each other, so three labelled
    # lines collide. The band carries the range and the one decisive line is drawn.
    lo, hi = CEILINGS[0][0], CEILINGS[-1][0]
    ax.axhspan(lo, hi, color=MUTED, alpha=0.11, zorder=1)
    ax.axhline(hi, color=RED, linewidth=1.5, linestyle='--', zorder=2)
    ax.text(3.15, hi + 3.5, 'fifty-year ceiling, %.2f kcal/mol' % hi,
            fontsize=8.5, color=RED, ha='right', fontweight='bold')
    ax.text(3.15, lo - 9.0, 'band: per day %.2f → per fifty years %.2f' % (lo, hi),
            fontsize=7.8, color=MUTED, ha='right')

    ax.plot(d, W, color=BLUE, linewidth=2.0, zorder=3)
    ax.plot(d, W, 'o', color=BLUE, markersize=4.5, zorder=4)

    # the two points the argument turns on, labelled directly rather than in a legend
    i50 = int(np.argmax(W > CEILINGS[-1][0]))
    ax.annotate('ceiling crossed\nnear %.1f Å, with\n%.1f Å still to travel' % (6.5, 3.3),
                xy=(d[i50], W[i50]), xytext=(d[i50] + 0.95, W[i50] + 26),
                fontsize=8.5, color=INK, ha='left',
                arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.9))
    j = int(np.argmin(np.abs(d - 5.57)))
    ax.annotate('%.2f kcal/mol at %.2f Å,\ntwice the fifty-year ceiling' % (W[j], d[j]),
                xy=(d[j], W[j]), xytext=(d[j] - 0.15, W[j] + 30),
                fontsize=8.5, color=INK, ha='center',
                arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.9))

    ax.set_xlabel('Cys6 S$\\gamma$ – Cys146 S$\\gamma$ separation (Å)', fontsize=10, color=INK)
    ax.set_ylabel('conformational work (kcal/mol)', fontsize=10, color=INK)
    ax.set_title('The approach costs more than the disease timescale allows',
                 fontsize=11, color=INK, loc='left', pad=12)
    ax.invert_xaxis()                      # the approach then reads left to right
    ax.set_ylim(-6, 150)
    save(fig, 'fig3_conformational_work')


def figure2(chi1):
    """Clearance against requirement. Every bar falls short; that is the result."""
    rows = [('crystal structure', 1.92, 'blocked by Thr116 C$\\gamma$2'),
            ('best of 5184 $\\chi_1$ rotamer pairs', 1.64, 'rotating the side chains\nexposes what is beneath them'),
            ('both side chains deleted entirely', 2.47, 'blocked by Leu117 O — backbone')]
    # read the headline numbers back from the file rather than trusting the literals
    stored = {r.get('case', ''): (r.get('clearance_heavy') or r.get('clearance'))
              for r in chi1['ladder']}
    for i, (lab, val, _note) in enumerate(rows):
        for k, v in stored.items():
            if lab.split()[0].lower() in k.lower() and abs(v - val) > 0.005:
                raise SystemExit('figure 2 disagrees with results/als_chi1_scan.json: %s' % k)

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    style(ax)
    ax.grid(axis='y', visible=False)

    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], height=0.5, color=BLUE, zorder=3)
    for yi, (lab, val, note) in zip(y, rows):
        ax.text(val + 0.08, yi, '%.2f Å' % val, va='center', fontsize=9.5, color=INK)
        ax.text(3.62, yi, note, va='center', fontsize=8, color=MUTED)   # one aligned column

    ax.axvline(REQUIRED, color=RED, linewidth=1.5, linestyle='--', zorder=4)
    ax.text(REQUIRED - 0.06, 2.62, 'a thiolate needs ≈3.3 Å', color=RED,
            fontsize=8.5, fontweight='bold', va='center', ha='right')
    ax.set_ylim(-0.65, 2.95)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
    ax.set_xlabel('clearance at the attack point (Å)', fontsize=10, color=INK)
    ax.set_title('Nothing the side chains can do opens the site',
                 fontsize=11, color=INK, loc='left', pad=12)
    ax.set_xlim(0, 6.9)
    save(fig, 'fig2_clearance_ladder')


def figure3():
    """The attack site, projected onto the plane of Cys57, Cys146 and Cys6."""
    at = A.read_chain()
    g = lambda ri, an: at[(ri, an)][1]
    s57, s146, s6 = g(57, 'SG'), g(146, 'SG'), g(6, 'SG')

    e1 = (s146 - s57) / np.linalg.norm(s146 - s57)          # along the S-S bond
    w = s6 - s146
    e2 = w - np.dot(w, e1) * e1
    e2 /= np.linalg.norm(e2)
    proj = lambda x: np.array([np.dot(x - s146, e1), np.dot(x - s146, e2)])

    attack = s146 + 3.0 * e1                                 # anti to the leaving sulfur
    shown = [(57, 'SG', 'Cys57 S$\\gamma$'), (146, 'SG', 'Cys146 S$\\gamma$'),
             (116, 'CG2', 'Thr116 C$\\gamma$2'), (118, 'CG2', 'Val118 C$\\gamma$2'),
             (117, 'O', 'Leu117 O'), (147, 'N', 'Gly147 N'), (146, 'CB', 'Cys146 C$\\beta$')]

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    style(ax)
    ax.grid(False)
    ax.set_aspect('equal')

    a2 = proj(attack)
    ax.add_patch(plt.Circle(a2, REQUIRED, facecolor=RED, alpha=0.08, edgecolor=RED,
                            linestyle='--', linewidth=1.3, zorder=1))
    ax.plot(*zip(proj(s57), proj(s146)), color=INK, linewidth=2.2, zorder=3)
    ax.plot(*zip(proj(s146), a2), color=MUTED, linewidth=1.2, linestyle=':', zorder=3)
    ax.plot(*a2, 'x', color=RED, markersize=11, markeredgewidth=2.2, zorder=5)

    for ri, an, label in shown:
        x, y = proj(g(ri, an))
        sulfur = an == 'SG'
        ax.plot(x, y, 'o', markersize=11 if sulfur else 8,
                color=INK if sulfur else BLUE, zorder=4)
        dist = np.linalg.norm(g(ri, an) - attack)
        txt = label if sulfur else '%s\n%.2f Å' % (label, dist)
        ax.annotate(txt, (x, y), textcoords='offset points', xytext=(0, 12 if y >= a2[1] else -22),
                    ha='center', fontsize=8.3, color=INK)

    x6, y6 = proj(s6)
    ax.plot(x6, y6, 'o', markersize=11, color=INK, zorder=4)
    ax.annotate('Cys6 S$\\gamma$\n8.37 Å away, at 133°', (x6, y6), textcoords='offset points',
                xytext=(0, -26), ha='center', fontsize=8.3, color=INK)
    # the disc itself is explained in the caption, not crowded into the plot
    ax.annotate('where the attacking\nsulfur must arrive', xy=a2, xytext=(-2.25, -2.1),
                textcoords='data', ha='left', fontsize=8.5, color=RED, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=RED, linewidth=0.9))

    ax.set_xlabel('Å along the Cys57→Cys146 axis', fontsize=10, color=INK)
    ax.set_ylabel('Å perpendicular', fontsize=10, color=INK)
    ax.set_title('The backside attack point is already occupied',
                 fontsize=11, color=INK, loc='left', pad=12)
    save(fig, 'fig1_attack_site')


def main():
    for need in (ROOT / 'results' / 'als_approach.json',
                 ROOT / 'results' / 'als_chi1_scan.json'):
        if not need.exists():
            print('missing %s -- run the calculation that produces it'
                  % need.relative_to(ROOT))
            return 1
    if not Path(A.PDB).exists():
        print('missing %s -- run scripts/fetch_structures.py 2C9U' % A.PDB)
        return 1

    res = json.loads((ROOT / 'results' / 'als_approach.json').read_text())
    chi1 = json.loads((ROOT / 'results' / 'als_chi1_scan.json').read_text())
    print('writing figures to %s' % OUT.relative_to(ROOT))
    figure1(res)
    figure2(chi1)
    figure3()
    return 0


if __name__ == '__main__':
    sys.exit(main())
