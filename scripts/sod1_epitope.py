#!/usr/bin/env python
"""Which SOD1 surface could a conformation-specific antibody use?

An antibody against misfolded SOD1 has to clear a bar that is easy to state and
easy to forget: the epitope must be exposed on the aggregate AND BURIED on the
healthy native dimer. One that is exposed on both binds normal SOD1 in every
cell that makes it, which is every cell. So the quantity wanted is not
accessibility but the DIFFERENCE in accessibility between the two states.

Both halves have to be measured in the right context or the difference is an
artefact of the model rather than of the protein:

  native    chain A of 1HL5 with chain B PRESENT. SOD1 is an obligate dimer, and
            a monomer calculation reports the whole dimer interface as exposed
            when in the cell it never is.
  aggregate the MIDDLE layer of a three-layer stack of the 7VZF fibril, so the
            measured chain has neighbours above and below as it would inside a
            filament. A single layer reports both stacking faces as surface.

Shrake-Rupley, summed per residue, over the residues present in both.

The result that was not sought: Thr116, Val118 and His120 are the three residues
that occlude the Cys146 backside attack trajectory in the folded protein, and
all three are buried natively and exposed on the fibril. The surface that blocks
the chemistry in the native state is the surface that appears on aggregation.
Those are the same measurement made twice for unrelated reasons, and they agree.

The second: His46 and His120 are both copper ligands and both become exposed, so
the surface reports the apo state. An antibody raised against it selects for
metal-free protein.

ALSO TESTED, because a published antibody rests on it. Sangwan et al. (bioRxiv
2025.05.05.652290) raised the monoclonal CSAb against residues 28-38, the
"corkscrew" segment, reasoning that Val29, Val31, Gly33 and Ile35 are buried in
native SOD1 and become exposed on misfolding, and that this difference could be
used to make a conformation-specific antibody. The native half is confirmed here
(Val29, Val31 and Ile35 are all at 0.0 A^2 in the dimer). On the fibril only
Val29 opens. That is not a contradiction: CSAb detects OLIGOMERS and not
fibrils, by their own result, and the corkscrew is an out-of-register oligomer
conformation rather than the fibril fold -- so the segment failing to surface in
a FIBRIL map is what their specificity predicts. Measuring it makes that
prediction quantitative instead of assumed.

    python scripts/sod1_epitope.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit import sanity                      # noqa: E402
from chemoqbit.sasa import sasa                   # noqa: E402

OUT = ROOT / 'results' / 'sod1_epitope_map.json'
NATIVE = ROOT / 'data' / 'rcsb_1hl5' / '1HL5.pdb'
FIBRIL = ROOT / 'data' / 'sod1_fibrils' / '7VZF.pdb'

BURIED = 10.0      # A^2 on the native dimer: below this a residue is not on the surface
EXPOSED = 30.0     # A^2 on the fibril: above this there is enough area for a contact
CORKSCREW = range(28, 39)
SANGWAN = (29, 31, 33, 35)


def read(path, chains):
    """Heavy atoms of the given chains, highest-occupancy conformer only.

    Summing alternate conformers would double-count a side chain and is the way
    a residue area goes over its physical maximum."""
    best, out = {}, []
    for line in open(path):
        if not line.startswith('ATOM'):
            continue
        chain = line[21]
        if chain not in chains:
            continue
        element = (line[76:78].strip() or line[12:16].strip()[0]).capitalize()
        if element == 'H':
            continue
        name, alt = line[12:16].strip(), line[16]
        key = (chain, int(line[22:26]), name)
        occ = float(line[54:60] or 1.0)
        if key in best and (best[key][0] >= occ or alt == ' '):
            continue
        best[key] = (occ, (line[17:20].strip(), element,
                           (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    for (chain, num, _name), (_occ, (resname, element, xyz)) in best.items():
        out.append((chain, num, resname, element, xyz))
    return out


def per_residue(path, chains, measured):
    """Residue accessibility of one chain, computed in the presence of the others."""
    atoms = read(path, chains)
    area = sasa([a[3] for a in atoms], np.array([a[4] for a in atoms]))
    acc, name = {}, {}
    for (chain, num, resname, _el, _xyz), a in zip(atoms, area):
        if chain != measured:
            continue
        acc[num] = acc.get(num, 0.0) + float(a)
        name[num] = resname
    return acc, name


def main():
    for path in (NATIVE, FIBRIL):
        if not path.exists():
            print('missing %s -- run scripts/fetch_structures.py' % path.relative_to(ROOT))
            return 1

    nat, nat_name = per_residue(str(NATIVE), {'A', 'B'}, 'A')
    fib, fib_name = per_residue(str(FIBRIL), {'A', 'B', 'C'}, 'B')
    shared = sorted(set(nat) & set(fib))

    res = dict(
        question='A conformation-specific antibody against misfolded SOD1 needs an epitope '
                 'accessible on the aggregate and BURIED on the healthy native dimer. An '
                 'epitope exposed on both binds normal protein in every cell that makes SOD1. '
                 'Which surface satisfies that?',
        method='Per-residue Shrake-Rupley accessibility of chain A inside the physiological '
               'dimer (1HL5, holo, both chains present) against the same residues in the '
               'middle layer of a three-layer stack of the wild-type fibril (7VZF), where '
               'that chain has neighbours above and below as inside a filament.',
        thresholds=dict(buried_native=BURIED, exposed_fibril=EXPOSED, unit='A^2'),
        n_comparable=len(shared))

    for num in shared:
        sanity.record(res, 'residue_sasa', nat[num], label='native %s%d' % (nat_name[num], num))
        sanity.record(res, 'residue_sasa', fib[num], label='fibril %s%d' % (fib_name[num], num))

    cand = [dict(residue='%s%d' % (nat_name[n], n), number=n,
                 native=round(nat[n], 1), fibril=round(fib[n], 1))
            for n in shared if nat[n] < BURIED and fib[n] > EXPOSED]
    cand.sort(key=lambda c: c['fibril'] - c['native'], reverse=True)
    res['candidates'] = cand
    res['unusable_exposed_on_both'] = [n for n in shared
                                       if nat[n] > EXPOSED and fib[n] > EXPOSED]

    print('%d residues comparable; %d buried natively and exposed on the fibril\n'
          % (len(shared), len(cand)))
    print('%-9s %9s %9s' % ('residue', 'native', 'fibril'))
    for c in cand:
        print('  %-7s %9.1f %9.1f' % (c['residue'], c['native'], c['fibril']))

    # the corkscrew segment, against the rationale CSAb was built on
    rows = []
    for n in CORKSCREW:
        if n in nat and n in fib:
            rows.append(dict(number=n, residue='%s%d' % (nat_name[n], n),
                             native=round(nat[n], 1), fibril=round(fib[n], 1),
                             named_by_sangwan=n in SANGWAN,
                             opens=bool(nat[n] < BURIED and fib[n] > EXPOSED)))
    opened = [r['residue'] for r in rows if r['named_by_sangwan'] and r['opens']]
    res['corkscrew_28_38'] = dict(
        reference='Sangwan et al., bioRxiv 2025.05.05.652290, monoclonal CSAb raised '
                  'against residues 28-38',
        their_rationale='Val29, Val31, Gly33 and Ile35 are buried in native folded SOD1 '
                        'and become solvent-exposed on misfolding',
        rows=rows,
        native_burial_confirmed=[r['residue'] for r in rows
                                 if r['named_by_sangwan'] and r['native'] < BURIED],
        opens_on_the_fibril=opened,
        segment_mean_native=round(float(np.mean([r['native'] for r in rows])), 1),
        segment_mean_fibril=round(float(np.mean([r['fibril'] for r in rows])), 1),
        reading='Their native-burial claim holds. On the FIBRIL only Val29 opens, and the '
                'segment is on average less accessible than natively. This is consistent '
                'with their own central result rather than against it: CSAb detects '
                'oligomers and not fibrils, and the corkscrew is an out-of-register '
                'oligomer conformation, not the fibril fold. A fibril accessibility map is '
                'therefore a structural negative control for that specificity. Testing it '
                'properly needs an oligomer coordinate model, which we do not have.')

    print('\nthe corkscrew segment 28-38 (Sangwan et al., CSAb)')
    print('%-9s %9s %9s' % ('residue', 'native', 'fibril'))
    for r in rows:
        print('  %-7s %9.1f %9.1f  %s%s'
              % (r['residue'], r['native'], r['fibril'],
                 'opens' if r['opens'] else '     ',
                 '  <- named by Sangwan et al.' if r['named_by_sangwan'] else ''))
    print('  segment mean: native %.1f, fibril %.1f'
          % (res['corkscrew_28_38']['segment_mean_native'],
             res['corkscrew_28_38']['segment_mean_fibril']))
    print('  of the four named, these open on the fibril: %s' % (', '.join(opened) or 'none'))

    # the stretch that looks like the answer and is not
    gap = [n for n in sorted(nat) if n not in fib]
    buried_in_gap = [n for n in gap if nat[n] < BURIED]
    res['disordered_in_the_fibril'] = dict(
        residues=[gap[0], gap[-1]] if gap else [],
        count=len(gap),
        mean_native_sasa=round(float(np.mean([nat[n] for n in gap])), 1) if gap else None,
        buried_natively=['%s%d' % (nat_name[n], n) for n in buried_in_gap],
        reading='Residues missing from the fibril model are flexible in the aggregate and '
                'look like an attractive target. They are not: most are already exposed on '
                'the healthy dimer, so they cannot be selective however accessible they '
                'become, and the few that are buried are glycine-rich and small, which '
                'makes poor epitopes.')
    if gap:
        print('\ndisordered in the fibril: %d residues, %d-%d, mean native SASA %.1f A^2; '
              'only %d of them buried natively'
              % (len(gap), gap[0], gap[-1], res['disordered_in_the_fibril']['mean_native_sasa'],
                 len(buried_in_gap)))

    res['two_convergences'] = [
        'Thr116, Val118 and His120 are the three residues that occlude the Cys146 backside '
        'attack trajectory in the folded protein. All three are buried natively and exposed '
        'on the fibril. The surface that blocks the chemistry in the native state is the '
        'surface that appears when the protein aggregates, and those are the same '
        'measurement made twice for unrelated reasons.',
        'His46 and His120 are both copper ligands and both become exposed, so the aggregate '
        'surface reports the apo state. An antibody raised against it selects for metal-free '
        'protein, which is the species the maturation argument identifies as vulnerable.']
    res['limitations'] = [
        'These are in-vitro fibrils. The authors state that their structures may not be '
        'identical to those in patients and that they intend to collect patient material.',
        'The H46R and G85R fibril cores comprise only residues 85-153 and 82-153, so their '
        'maps differ from the wild-type one. If SOD1 strains are real -- and prion-like '
        'transmission of two distinct strains has been reported -- one conformation-specific '
        'antibody may not cover them.',
        'Accessibility is necessary and not sufficient. An antibody recognises a '
        'conformational surface, not solvent-accessible area, and nothing here models a '
        'paratope.',
        'A three-layer stack approximates the lateral surface of a filament; a real one is '
        'longer and may bundle.']

    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print('\nsaved %s' % OUT.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
