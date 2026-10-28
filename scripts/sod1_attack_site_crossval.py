#!/usr/bin/env python
"""Does the Cys6 attack geometry hold in more than one crystal of SOD1?

results/als_sod1_attack_site.json measures the backside attack site on Cys146 in
ONE chain of ONE structure (2C9U chain F conformer B, 1.24 A). An exclusion that
rests on a single chain is an exclusion that a reviewer can attribute to crystal
packing, to a refinement choice, or to the conformer we happened to pick. So the
same measurement is repeated here across every chain of every wild-type SOD1
crystal in data/, and the structures are treated as replicates rather than as one
preferred model plus some others.

7FB9 (Baek et al., Commun Biol 2022) is what makes this worth doing: a different
space group from 2C9U, thirteen disulfide-bonded chains in the asymmetric unit,
and deposited after the proximity argument it tests.

THE MEASUREMENT, per chain, is the one the single-chain file defines:

  angle     S6...S146-S57. An S_N2 at sulfur proceeds by backside attack, so the
            nucleophile, the attacked sulfur and the leaving sulfur must be close
            to collinear. The sanity bound for a reacting system is >= 150 deg.
  site      the point 3.3 A beyond S146 along the direction AWAY from S57, which
            is where the attacking sulfur has to arrive. Asking what is AT that
            point is sharper than asking what lies on the line between the two
            sulfurs, and gives a different answer.

WHAT REPLICATES AND WHAT DOES NOT, which is the reason this file exists:

  replicates      the angle never reaches 150 deg, in any chain of any structure;
                  the attack site is occupied in every chain; and backbone is part
                  of what occupies it every time.
  does NOT        WHICH residue occupies the site. In 2C9U conformer B it is the
                  gamma-methyls of Thr116 and Val118; in conformer A of the same
                  chain, and in most chains of 7FB9, it is the Gly147 backbone.
                  So the occupant is a property of the conformer and the occupancy
                  is a property of the fold. Only the second belongs in a claim.

7FB9 is 2.70 A and its S-S distances are refinement-restrained to 2.03 A in all
thirteen chains, so it is good evidence about the angle and the backbone and weak
evidence about which side chain sits where. 2C9U at 1.24 A stays the structure for
that, and this file does not overturn it -- it bounds what can be said from it.

NOT RECORDED as a sanity violation: chalcogen_attack_angle. Every chain here is
below its 150 deg floor, which is the RESULT rather than a fault, and recording it
would put fourteen false positives in the auditor for a finding the file is about.
The bound is checked against the disulfide bond length instead, where a violation
would be real.

    python scripts/sod1_attack_site_crossval.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit import sanity                      # noqa: E402

OUT = ROOT / 'results' / 'als_attack_site_crossval.json'

# wild-type, disulfide-intact SOD1 crystals. The fibrils are not here: they are a
# different state and the question does not apply to them.
STRUCTURES = [
    dict(pdb='2C9U', path=ROOT / 'data' / 'rcsb_2c9u' / '2C9U.pdb', resolution=1.24,
         note='Strange et al., JMB 2006; the structure the single-chain measurement '
              'was made on, chain F conformer B'),
    dict(pdb='1HL5', path=ROOT / 'data' / 'rcsb_1hl5' / '1HL5.pdb', resolution=1.80,
         note='Strange et al., JMB 2003; eighteen chains, and the S-S distances vary '
              'from 2.04 to 2.21 A rather than sitting at one restrained value, so the '
              'disulfide geometry here is refined per chain and not imposed'),
    dict(pdb='7FB9', path=ROOT / 'data' / 'rcsb_7fb9' / '7FB9.pdb', resolution=2.70,
         note='Baek et al., Commun Biol 2022; P3_2, an independent crystal form'),
]

REACH = 3.3        # A, where the attacking sulfur has to arrive, and the radius
                   # searched around it -- a sulfur-carbon van der Waals contact
BACKBONE = {'N', 'CA', 'C', 'O', 'OXT'}
BACKSIDE_FLOOR = 150.0     # deg; chemoqbit.sanity.BOUNDS['chalcogen_attack_angle']


def conformers(path):
    """Every (chain, altloc) of a PDB file as its own set of heavy atoms.

    Alternate conformers are kept APART rather than merged or silently resolved to
    the highest occupancy: the whole point here is that the answer depends on which
    one is taken, so collapsing them would destroy the finding."""
    chains = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        if not line.startswith('ATOM'):
            continue
        element = (line[76:78].strip() or line[12:16].strip()[0]).capitalize()
        if element == 'H':
            continue
        chain, alt = line[21], line[16]
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        chains[chain][alt].append((int(line[22:26]), line[17:20].strip(),
                                   line[12:16].strip(), xyz))
    out = []
    for chain, by_alt in sorted(chains.items()):
        shared = by_alt.get(' ', [])
        letters = sorted(a for a in by_alt if a != ' ')
        if not letters:
            out.append((chain, ' ', shared))
            continue
        for alt in letters:
            out.append((chain, alt, shared + by_alt[alt]))
    return out


def sulfur(atoms, num):
    return next((x for n, _rn, an, x in atoms if n == num and an == 'SG'), None)


def measure(atoms):
    """One chain: the backside angle, and what sits at the attack site."""
    a, b, c = sulfur(atoms, 6), sulfur(atoms, 146), sulfur(atoms, 57)
    if a is None or b is None or c is None:
        return None
    ss = float(np.linalg.norm(b - c))
    axis = (b - c) / ss                       # S57 -> S146, extended past S146
    site = b + REACH * axis
    v1, v2 = a - b, c - b
    angle = float(np.degrees(np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))))
    occupants = sorted(
        (dict(atom='%s%d:%s' % (rn, n, an), residue='%s%d' % (rn, n),
              d=round(float(np.linalg.norm(x - site)), 2),
              backbone=an in BACKBONE)
         for n, rn, an, x in atoms if not (n == 146 and an == 'SG')),
        key=lambda o: o['d'])
    # Cys6...Cys111 is the control on the proximity argument rather than a
    # by-product: that is the displacement Cys6 is actually known to perform in
    # SOD1 (Anzai et al. 2020, on a sulfenylated Cys111), and it is the LONGER
    # of the two approaches. If separation decided this chemistry the ranking
    # would be the other way round.
    d111 = sulfur(atoms, 111)
    return dict(disulfide_SS=round(ss, 2),
                d_S6_S146=round(float(np.linalg.norm(b - a)), 2),
                d_S6_S111=(None if d111 is None
                           else round(float(np.linalg.norm(d111 - a)), 2)),
                attack_angle=round(angle, 1),
                travel=round(float(np.linalg.norm(a - site)), 2),
                occupants=[o for o in occupants if o['d'] < REACH])


def free_cysteine_routes(atoms):
    """Every intramolecular attack on the Cys57-Cys146 bond that SOD1 could mount.

    This exists because of an objection with a real precedent. A buried disulfide
    is not always unreachable: in pI258 arsenate reductase the buried Cys10-Cys15
    bond is relayed outward by a THIRD cysteine, Cys82, until thioredoxin can
    reach it (Messens et al., JMB 2004). So the question is not only whether Cys6
    can attack, but whether any cysteine in SOD1 can.

    SOD1 has exactly four. Two of them ARE the bond. So the candidate relays are
    Cys6 and Cys111 and nothing else, and both ends of the bond have to be tested
    because a relay may attack either sulfur."""
    sg = {n: sulfur(atoms, n) for n in (6, 57, 111, 146)}
    if any(v is None for v in sg.values()):
        return None
    out = {}
    for att in (6, 111):
        for tgt, leaving in ((146, 57), (57, 146)):
            v1, v2 = sg[att] - sg[tgt], sg[leaving] - sg[tgt]
            out['Cys%d_to_Cys%d' % (att, tgt)] = dict(
                d=round(float(np.linalg.norm(sg[att] - sg[tgt])), 2),
                angle=round(float(np.degrees(np.arccos(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))), 1))
    return out


def main():
    missing = [s['pdb'] for s in STRUCTURES if not s['path'].exists()]
    if missing:
        print('missing %s -- run scripts/fetch_structures.py %s'
              % (', '.join(missing), ' '.join(missing)))
        return 1

    res = dict(
        question='The backside attack site on Cys146 was measured in one chain of one '
                 'SOD1 crystal. Does the geometry that excludes the reaction hold across '
                 'every chain of every wild-type crystal available, or is it a property '
                 'of that chain?',
        method='Per chain and per alternate conformer: the S6...S146-S57 angle, and the '
               'occupants within %.1f A of the point %.1f A beyond S146 along the '
               'direction away from S57. Alternate conformers are kept separate rather '
               'than resolved by occupancy, because the occupant depends on which is '
               'taken.' % (REACH, REACH),
        backside_floor_deg=BACKSIDE_FLOOR,
        reach=REACH, structures=[])

    rows, angles = [], []
    occupied_by = defaultdict(set)       # residue -> {(pdb, chain, alt)}
    for s in STRUCTURES:
        per_chain = []
        for chain, alt, atoms in conformers(s['path']):
            m = measure(atoms)
            if m is None:
                continue
            key = '%s %s%s' % (s['pdb'], chain, '' if alt == ' ' else '/' + alt)
            m.update(chain=chain, altloc=alt.strip() or None,
                     approaches=free_cysteine_routes(atoms))
            per_chain.append(m)
            angles.append(m['attack_angle'])
            rows.append((key, m))
            for o in m['occupants']:
                occupied_by[o['residue']].add(key)
            sanity.record(res, 'chalcogen_bond', m['disulfide_SS'],
                          label='%s Cys57-Cys146' % key)
        res['structures'].append(dict(
            pdb=s['pdb'], resolution=s['resolution'], note=s['note'],
            n_chains_measured=len(per_chain), chains=per_chain))
        print('%s  %.2f A  %d chains with all three cysteine sulfurs modelled'
              % (s['pdb'], s['resolution'], len(per_chain)))

    n = len(rows)
    print('\n%-12s %8s %9s %8s   %s'
          % ('chain', 'S-S', 'S6-S146', 'angle', 'at the attack site'))
    for key, m in rows:
        print('%-12s %8.2f %9.2f %8.1f   %s'
              % (key, m['disulfide_SS'], m['d_S6_S146'], m['attack_angle'],
                 ', '.join('%s %.2f' % (o['atom'], o['d']) for o in m['occupants'][:4])
                 or 'CLEAR'))

    clear = [k for k, m in rows if not m['occupants']]
    aligned = [k for k, m in rows if m['attack_angle'] >= BACKSIDE_FLOOR]
    backbone_in = [k for k, m in rows if any(o['backbone'] for o in m['occupants'])]
    # 2C9U chain A carries the SSBOND record but models the bond at 2.75 A, and
    # conformer A of chain F at 2.58 A. Those are not formed disulfides, so they are
    # counted apart rather than folded into the replicate count -- the conclusion
    # should not need them, and stating the smaller number is the honest claim.
    formed = [k for k, m in rows if 1.95 <= m['disulfide_SS'] <= 2.45]
    res['summary'] = dict(
        n_chains=n,
        n_chains_with_a_formed_disulfide=len(formed),
        chains_with_a_formed_disulfide=formed,
        attack_angle=dict(mean=round(float(np.mean(angles)), 1),
                          min=round(min(angles), 1), max=round(max(angles), 1)),
        chains_reaching_backside_floor=aligned,
        chains_with_a_clear_attack_site=clear,
        chains_with_backbone_at_the_site=len(backbone_in),
        occupancy_by_residue={r: sorted(k) for r, k in
                              sorted(occupied_by.items(), key=lambda kv: -len(kv[1]))})
    res['reading'] = (
        'Across %d chains (%d of them with a physically formed disulfide) the attack angle '
        'spans %.1f-%.1f deg and reaches the %.0f deg '
        'backside floor in none of them, the attack site is occupied in all of them, and '
        'backbone is among the occupants in %d. What does NOT replicate is which residue '
        'occupies the site: it is the Thr116 and Val118 side chains in the 2C9U conformer '
        'the single-chain measurement used, and Gly147 backbone in most chains of 7FB9 and '
        'in the other 2C9U conformer. The occupancy is a property of the fold and the '
        'occupant is a property of the conformer, so a claim about the obstruction should '
        'rest on the first. 7FB9 is 2.70 A with its S-S distances refinement-restrained, '
        'which makes it strong evidence on the angle and the backbone and weak evidence on '
        'side-chain identity.'
        % (n, len(formed), min(angles), max(angles), BACKSIDE_FLOOR, len(backbone_in)))

    print('\n%d chains over %d crystals, %d with a formed disulfide'
          % (n, len(STRUCTURES), len(formed)))
    print('  attack angle          %.1f deg (%.1f-%.1f)'
          % (res['summary']['attack_angle']['mean'], min(angles), max(angles)))
    print('  reaching %.0f deg       %s' % (BACKSIDE_FLOOR, ', '.join(aligned) or 'none'))
    print('  with a clear site     %s' % (', '.join(clear) or 'none'))
    print('  with backbone at it   %d of %d' % (len(backbone_in), n))
    print('\nresidues at the attack site, by how many chains:')
    for r, keys in res['summary']['occupancy_by_residue'].items():
        print('   %-10s %2d/%d' % (r, len(keys), n))

    # the two approaches Cys6 could make, per structure
    approaches = {}
    for s in res['structures']:
        for key, field in (('to_Cys146', 'd_S6_S146'), ('to_Cys111', 'd_S6_S111')):
            vals = [c[field] for c in s['chains'] if c.get(field) is not None]
            if vals:
                approaches.setdefault(key, {})[s['pdb']] = dict(
                    mean=round(float(np.mean(vals)), 2),
                    min=round(min(vals), 2), max=round(max(vals), 2), n=len(vals))
    res['cys6_approaches'] = dict(
        distances=approaches,
        reading='Cys6 is FURTHER from Cys111 than from Cys146 in every structure, and the '
                'displacement on Cys111 is the one that has been demonstrated (Anzai et al. '
                '2020, on the sulfenylated residue) while the approach to Cys146 has not. '
                'Separation does not rank these two reactions correctly, so it is not what '
                'decides them.')
    # every intramolecular attack available to SOD1, over the chains whose
    # disulfide is physically formed -- the malformed ones distort the angle at
    # the attacked sulfur, which is the quantity in question
    relay = {}
    for key, m in rows:
        if not (1.95 <= m['disulfide_SS'] <= 2.45) or not m.get('approaches'):
            continue
        for route, v in m['approaches'].items():
            relay.setdefault(route, {'d': [], 'angle': []})
            relay[route]['d'].append(v['d'])
            relay[route]['angle'].append(v['angle'])
    res['intramolecular_routes'] = dict(
        n_chains=len(relay['Cys6_to_Cys146']['d']) if relay else 0,
        routes={r: dict(d_mean=round(float(np.mean(v['d'])), 2),
                        d_min=round(min(v['d']), 2), d_max=round(max(v['d']), 2),
                        angle_mean=round(float(np.mean(v['angle'])), 1),
                        angle_min=round(min(v['angle']), 1),
                        angle_max=round(max(v['angle']), 1))
                for r, v in relay.items()},
        reading='SOD1 has four cysteines and two of them are the bond, so Cys6 and Cys111 '
                'are the only possible attackers and there is no third cysteine left to '
                'act as the relay that reaches the buried disulfide in pI258 arsenate '
                'reductase (Messens et al., JMB 2004). Both ends are tested because a '
                'relay may attack either sulfur. All four routes fail, for two different '
                'reasons: the approaches to Cys57 are far and nowhere near collinear, and '
                'the approaches to Cys146 are either close and misaligned (Cys6) or well '
                'aligned and twice as far (Cys111).')
    print('\nevery intramolecular attack on the Cys57-Cys146 bond '
          '(%d chains, formed disulfide only):' % res['intramolecular_routes']['n_chains'])
    print('   %-20s %18s %16s' % ('route', 'd (A)', 'angle at target'))
    for r, v in res['intramolecular_routes']['routes'].items():
        print('   %-20s %7.2f (%.2f-%.2f) %7.1f (%.0f-%.0f)'
              % (r.replace('_to_', ' -> '), v['d_mean'], v['d_min'], v['d_max'],
                 v['angle_mean'], v['angle_min'], v['angle_max']))

    print('\nthe two approaches Cys6 could make (A, mean over chains):')
    print('   %-8s %12s %12s' % ('', 'to Cys146', 'to Cys111'))
    for s in res['structures']:
        a = approaches.get('to_Cys146', {}).get(s['pdb'])
        b = approaches.get('to_Cys111', {}).get(s['pdb'])
        print('   %-8s %12s %12s'
              % (s['pdb'],
                 '%.2f' % a['mean'] if a else '-',
                 '%.2f' % b['mean'] if b else '-'))

    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print('\nsaved %s' % OUT.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
