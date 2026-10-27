#!/usr/bin/env python
"""Download the crystal structures the enzyme and folding pipelines need.

The PDB files are not in the repository. They are third-party data, they are
large, and they are better fetched from the authoritative source than vendored
into git -- so `data/` is gitignored. The consequence, which is easy to miss, is
that a fresh clone cannot run any of the cluster work until this has been run.

    python scripts/fetch_structures.py            # everything the code references
    python scripts/fetch_structures.py 2C9U 5WMJ  # just these

Each entry goes where the code expects to find it: 1TLS flat in data/, the rest
under data/rcsb_<id>/<ID>.pdb, which is the layout the RCSB web download produces
and which chemoqbit.pocket, chemoqbit.zipper and chemoqbit.als_approach assume.
"""
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
URL = 'https://files.rcsb.org/download/{}.pdb'

# Every entry referenced from chemoqbit/ or scripts/, with what uses it.
STRUCTURES = {
    '1TLS': 'thymidylate synthase with dUMP and a folate analogue; the 96-atom '
            'active-site cluster of the first two applications',
    '2C9U': 'human Cu/Zn superoxide dismutase at 1.24 A; the SOD1 accessibility, '
            'attack-geometry and Cys6-approach work. Use chain F conformer B: chain A '
            'carries the Cys57-Cys146 SSBOND record but models it at 2.75 A',
    '2ON9': 'five-fluorouracil-related structure for the tautomerism application',
    '4EY7': 'acetylcholinesterase with donepezil; the AChE pocket alanine scan',
    '4NIP': 'GVIGIAQ, SOD1 residues 147-153, 1.90 A; benchmark for the sod1ct zipper',
    '4RIL': 'GAVVTGVTAVA amyloid segment, 1.43 A; a zipper benchmark',
    '4ZNN': 'GVVHGVTTVA from the A53T alpha-synuclein mutant, 1.41 A (MicroED)',
    '5WMJ': 'KVWGSI, SOD1 residues 30-35, 1.40 A; benchmark for the sod1 zipper. '
            'Note 1.40 and not the 1.45 some secondary sources give -- read from '
            'REMARK 2 of this file',
    '7SVD': 'CFTR NBD1 with lumacaftor (VX-809); the cftr pocket alanine scan',
    '1HL5': 'human SOD1 holo dimer at 1.8 A; the NATIVE half of the epitope map -- '
            'accessibility is measured on chain A with chain B present, because a '
            'monomer would report the dimer interface as exposed when it is not',
    '7VZF': 'cryo-EM fibril of full-length wild-type human SOD1 (Wang et al., Nat '
            'Commun 2022); the AGGREGATE half of the epitope map. Chains A, B and C '
            'are three stacked layers and the middle one is the one measured',
    '8IHV': 'SOD1 H46R fibril (Wang et al., Sci Adv 2024); core 85-153 only',
    '8IHU': 'SOD1 G85R fibril (Wang et al., Sci Adv 2024); core 82-153 only',
}

FLAT = {'1TLS'}          # the one entry the code expects directly under data/
# the fibrils live together rather than one directory each, which is where the
# epitope map looks for them
FOLDER = {'7VZF': 'sod1_fibrils', '8IHV': 'sod1_fibrils', '8IHU': 'sod1_fibrils'}


def destination(pdb_id):
    if pdb_id in FOLDER:
        return DATA / FOLDER[pdb_id] / f'{pdb_id}.pdb'
    if pdb_id in FLAT:
        return DATA / f'{pdb_id}.pdb'
    return DATA / f'rcsb_{pdb_id.lower()}' / f'{pdb_id}.pdb'


def fetch(pdb_id):
    out = destination(pdb_id)
    if out.exists() and out.stat().st_size > 0:
        print(f'  {pdb_id}  already present ({out.stat().st_size // 1024} kB)')
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(URL.format(pdb_id), timeout=60) as r:
            body = r.read()
    except (urllib.error.URLError, TimeoutError) as e:
        print(f'  {pdb_id}  FAILED: {e}')
        return False
    if not body.startswith(b'HEADER'):
        print(f'  {pdb_id}  FAILED: the response is not a PDB file')
        return False
    out.write_bytes(body)
    print(f'  {pdb_id}  {len(body) // 1024} kB -> {out.relative_to(ROOT)}')
    return True


def main(argv):
    wanted = [a.upper() for a in argv] or sorted(STRUCTURES)
    unknown = [w for w in wanted if w not in STRUCTURES]
    if unknown:
        print(f'not referenced by this codebase: {", ".join(unknown)}')
        print(f'known entries: {", ".join(sorted(STRUCTURES))}')
        return 2
    print(f'fetching {len(wanted)} structure(s) from RCSB into {DATA}')
    ok = [fetch(w) for w in wanted]
    bad = sum(1 for o in ok if not o)
    print()
    if bad:
        print(f'{bad} of {len(wanted)} failed; rerun to retry just those')
        return 1
    print(f'all {len(wanted)} present. The cluster pipelines can now run.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
