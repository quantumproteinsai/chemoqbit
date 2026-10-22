"""Bundle the thymidylate synthase cluster results as a Zenodo dataset.

Collects the checkpoint, the archived intermediate checkpoints, the side
files, the run logs and the PDF report into dist/, with a README that
describes every file and states which stages are complete, a SHA-256
manifest, and zenodo_dataset.json holding the record metadata to enter on
zenodo.org. Nothing is uploaded.

    python scripts/make_dataset.py            # re-run after stages finish

The notifier's files are left out: they hold the recipient's address.
"""
import datetime, hashlib, importlib.metadata as md, json, platform, subprocess, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
VERSION = json.loads((ROOT / '.zenodo.json').read_text())['version']
K = 627.509474

ROLES = [  # (glob, description) -- first match wins
    ('ts_cluster_run.json', 'current checkpoint: every stage\'s results (see "Checkpoint layout")'),
    ('ts_cluster_run.N5adduct_*.json', 'original failed run: N5-C11 bond re-formed, scan cliff at xi = +1.2'),
    ('ts_cluster_run.badscan_*.json', 'first rerun: N5 re-bonded to C11 at the first scan point'),
    ('ts_cluster_run.prebackfill_*.json', 'forward scan before the backward pass (folate shifts visible)'),
    ('ts_cluster_run.capsmovedTS_*.json', 'TS search that moved the frozen atoms (barrier 5.3, invalid)'),
    ('ts_cluster_run.preactivemerge_*.json', 'checkpoint before the corrected active-space results were merged'),
    ('ts_cluster_run.preoniommerge_*.json', 'checkpoint before the ONIOM results were merged'),
    ('ts_functional_check.json', 'embedded-fragment barrier with five functionals at def2-TZVP'),
    ('ts_cluster_run.his147p.json', 'His147-protonated model (charge +2): xTB stages, same pipeline'),
    ('ts_oniom_opt.his147p.json', 'His147-protonated model: ONIOM(wB97X-D3:xTB) optimisation, Hessians, KIEs'),
    ('ts_oniom_opt.json', 'ONIOM(wB97X-D3:xTB) optimised reactant/TS/product, partial Hessians, KIEs, def2-TZVP energies'),
    ('ts_profile_check.json', 'wB97X-D3(BJ)/def2-TZVP barrier profile along the xTB path (fragment and ONIOM(DFT:xTB))'),
    ('ts_cluster_run.o4h.json', 'O4-protonated model (charge +2): stopped by the scan guard, proton passes to water 327'),
    ('ts_o4h_diagnostic.json', 'O4-protonated model: geometric diagnosis of why it is not viable'),
    ('ts_cs2d.json', 'two-dimensional surface over xi and C6-S: GFN2-xTB geometries, ONIOM(wB97X-D3/def2-SVP:xTB) energies'),
    ('ts_thermo.json', 'free-energy corrections (harmonic and quasi-harmonic) at 293 and 298 K, both His147 states'),
    ('ts_kie_temperature.json', 'H/T effects at 278-313 K and Arrhenius parameters, both levels and His147 states'),
    ('ts_qmsize.json', 'ONIOM barrier and reaction energy for high-level regions of 18, 36, 64 and 96 atoms'),
    ('SI_manuscript*.pdf', 'supporting information of the two preprints'),
    ('ts_active_*.json', 'corrected active-space stage (capped 18-atom fragment), side file'),
    ('ts_oniom_*.json', 'ONIOM / electrostatic-embedding stage (and focal-point data if merged), side file'),
    ('ts_focal_*.json', 'ONIOM stage plus DF-MP2 cc-pVDZ/cc-pVTZ basis-set correction, side file'),
    ('ts_pipeline_report.pdf', 'report: what failed, why, how it was fixed, and the results'),
    ('nuc_*_tzvp.log', 'run log, def2-TZVP DFT single points (several attempts; see README)'),
    ('nuc_*.log', 'run log, xTB stages and def2-SVP DFT'),
    ('ts_*_2026*.log', 'run log of a side job (active space, ONIOM, focal point or functional check)'),
]


def role(name):
    from fnmatch import fnmatch
    return next((d for g, d in ROLES if fnmatch(name, g)), None)


def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def status(r):
    have = lambda *ks: all(k in r for k in ks)
    dft = r.get('dft', {})
    full = lambda b: all(f'{n}/{b}' in dft for n in ('reactant', 'ts', 'product'))
    rows = [('0-2 prep, scan, ts', have('prep', 'scan', 'ts', 'reactant', 'product')),
            ('3-4 hess, kie', have('analysis')),
            ('5 dft def2-SVP', full('def2-svp')),
            ('5 dft def2-TZVP', full('def2-tzvp')),
            ('6 active', len(r.get('active', {})) == 3),
            ('7 oniom', len(r.get('oniom', {})) == 3),
            ('8 focal', all('mp2' in v for v in r.get('oniom', {}).values()) and len(r.get('oniom', {})) == 3),
            ('profile check', (RES / 'ts_profile_check.json').exists()
             and '_profile' in json.loads((RES / 'ts_profile_check.json').read_text()))]
    return rows


def key_numbers(r):
    out = []
    a = r.get('analysis')
    if a:
        out.append(f"xTB barrier {a['dE_barrier']:.2f}, reaction {a['dE_rxn']:+.2f} kcal/mol; TS "
                   f"{abs(a['iso']['ts']['H']['imag']):.0f}i cm-1")
        for k, v in a['kie'].items():
            out.append(f"KIE {k}: semiclassical {v['semiclassical']:.2f}, Wigner {v['wigner']:.2f}, "
                       f"Eckart {v['eckart']:.2f}")
    for b in ('def2-svp', 'def2-tzvp'):
        d = r.get('dft', {})
        if all(f'{n}/{b}' in d for n in ('reactant', 'ts', 'product')):
            e = {n: d[f'{n}/{b}']['E'] for n in ('reactant', 'ts', 'product')}
            out.append(f"B3LYP-D3(BJ)/{b}, PCM eps 4: barrier {(e['ts'] - e['reactant']) * K:.2f}, "
                       f"reaction {(e['product'] - e['reactant']) * K:+.2f} kcal/mol")
    return out


def main():
    today = datetime.date.today().isoformat()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'chemoqbit', 'scripts'], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    files = sorted(p for p in RES.iterdir() if p.is_file() and role(p.name))
    r = json.loads((RES / 'ts_cluster_run.json').read_text())
    vers = {}
    for pkg in ('numpy', 'scipy', 'pyscf', 'tblite', 'rdkit', 'geometric'):
        try:
            vers[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            vers[pkg] = 'not installed'
    st = status(r)

    readme = [f'# chemoqbit thymidylate synthase cluster: results ({today})', '',
              'Results of the hydride-transfer pipeline in `scripts/ts_cluster_nuc.py` of chemoqbit '
              f'{VERSION}: a 96-atom active-site cluster from PDB 1TLS, GFN2-xTB scan and saddle search, '
              'H/D/T kinetic isotope effects, B3LYP-D3(BJ) single points, and an electrostatically embedded '
              'active space with ONIOM(CCSD(T):B3LYP) and UCCSD/ADAPT-VQE circuits.', '',
              f'Code: chemoqbit commit `{commit}`' + (' (with uncommitted changes)' if dirty else '') + '.',
              'Software DOI: _add the chemoqbit 1.1.0 DOI once the release is archived_.', '',
              '## Status of the stages', '', '| Stage | Complete |', '|---|---|',
              *[f'| {s} | {"yes" if ok else "**no**"} |' for s, ok in st], '']
    kn = key_numbers(r)
    if kn:
        readme += ['## Key numbers', '', *[f'- {x}' for x in kn],
                   '- Experimental H/T KIE: 6.91 +- 0.05 at 293 K', '',
                   'See `ts_pipeline_report.pdf` for the context and caveats; in particular, all DFT and '
                   'ONIOM energies are single points at the GFN2-xTB geometries.', '']
    readme += ['## Files', '', '| File | Size | Contents |', '|---|---|---|',
               *[f'| `{p.name}` | {p.stat().st_size / 1e6:.2f} MB | {role(p.name)} |' for p in files], '',
               'Several `nuc_*_tzvp.log` files are attempts that ended early (out-of-memory kills, a session '
               'teardown, a reboot); only the last one carries results, and it is complete only if the '
               'status table says so.', '',
               '## Checkpoint layout (`ts_cluster_run.json`)', '',
               'Units: angstrom, hartree (Eh), cm-1; kcal/mol only where a key says so.', '',
               '- `prep`: symbols, coords, labels, charge (+1), frozen atom indices, restraints, key atom indices',
               '- `scan`: list of points with `xi`, `E`, `coords` and key distances; `backfilled` marks points '
               'replaced by the backward pass',
               '- `reactant`, `ts`, `product`: `coords`, `E` (xTB), `hessian` (free atoms only), `ts.prfo`',
               '- `analysis`: barriers, frequencies, ZPE, partition functions and KIEs per isotopologue',
               '- `dft`: `<structure>/<basis>` -> energy, convergence, timing, density fitting',
               '- `active`: capped 18-atom fragment, `<structure>/4e4o` -> HF, CCSD(T), CASCI, VQE results',
               '- `oniom`: embedded fragment per structure -> DFT model energy, CCSD(T), CASCI, VQE results, '
               'point-charge summary, and `mp2`/`e_ccsd_t_focal` from the focal stage', '',
               '## Software', '', f'Python {platform.python_version()} on {platform.system()} {platform.release()}',
               '', *[f'- {k} {v}' for k, v in vers.items()], '',
               '## Licence', '', 'CC BY 4.0. The cluster coordinates derive from PDB entry 1TLS.', '']

    meta = dict(upload_type='dataset',
                title='Thymidylate synthase hydride transfer: cluster-model scans, saddle points, kinetic isotope '
                      'effects and embedded active-space results (chemoqbit)',
                description='Checkpoints, run logs and a report for the thymidylate synthase hydride-transfer '
                            'pipeline of chemoqbit: a 96-atom active-site cluster from PDB 1TLS, GFN2-xTB reaction '
                            'scan and P-RFO saddle point, H/D/T kinetic isotope effects with Wigner and Eckart '
                            'tunnelling, B3LYP-D3(BJ)/PCM single points, and an electrostatically embedded 18-atom '
                            'fragment with ONIOM(CCSD(T):B3LYP), a DF-MP2 basis-set correction and UCCSD/ADAPT-VQE '
                            'circuits. Includes the intermediate checkpoints of the failure modes documented in the '
                            'report.',
                creators=json.loads((ROOT / '.zenodo.json').read_text())['creators'],
                license='cc-by-4.0', access_right='open', version=today,
                keywords=['thymidylate synthase', 'hydride transfer', 'kinetic isotope effect', 'cluster model',
                          'GFN2-xTB', 'transition state', 'ONIOM', 'electrostatic embedding', 'ADAPT-VQE'],
                notes='Add a related identifier "is supplement to" the chemoqbit 1.1.0 software DOI.')

    out = ROOT / 'dist'; out.mkdir(exist_ok=True)
    z = out / f'chemoqbit-ts-cluster-data-{today}.zip'
    base = f'chemoqbit-ts-cluster-data-{today}'
    manifest = '\n'.join(f'{sha256(p)}  data/{p.name}' for p in files) + '\n'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f'{base}/README.md', '\n'.join(readme))
        zf.writestr(f'{base}/MANIFEST.sha256', manifest)
        zf.writestr(f'{base}/zenodo_dataset.json', json.dumps(meta, indent=2) + '\n')
        for p in files:
            zf.write(p, f'{base}/data/{p.name}')
    (out / 'zenodo_dataset.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(z, f'{z.stat().st_size / 1e6:.1f} MB,', len(files), 'files')
    for s, ok in st:
        print(f'  {s:22s} {"done" if ok else "NOT DONE"}')


if __name__ == '__main__':
    main()
