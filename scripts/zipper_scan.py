"""What holds an amyloid steric zipper together, residue by residue.

Three segments, each a microcrystal structure good enough for a side-chain
argument -- unlike the cryo-EM fibrils of full-length Abeta or tau, which carry
the familial mutations but sit at 3.4-4.0 A:

  tau       tau PHF6, 306-VQIVYK-311 (2ON9, 1.51 A; Sawaya et al., Nature 2007)
  asyn47    alpha-synuclein 47-56, GVVHGVTTVA (4ZNN, 1.41 A; Rodriguez et al.,
            Nature 2015) -- the A53T FAMILIAL PARKINSON mutant as crystallised.
            Truncating Thr53 to alanine restores the wild-type sequence, so the
            pathogenic mutation is a difference between two models that differ
            by one side chain, at a geometry good enough to mean something.
  nacore    alpha-synuclein 68-78, GAVVTGVTAVA (4RIL, 1.43 A), the toxic core

The mating sheet is the partner that buries the most area with the largest
side-chain share (chemoqbit.zipper.mating_sheet), which is how Sawaya et al.
define a zipper: facing side chains interdigitating in a dry interface.
Counting close contacts, as a first attempt did, scores an edge-on packing
contact just as highly and picked the wrong partner.

  E_int  = E(both sheets) - E(sheet A) - E(sheet B), at one geometry
  ddE    = E_int(residue truncated to alanine in both sheets) - E_int(wild type)

Truncation happens in the relaxed geometry, so every other atom stays put.
Hydrogens are relaxed once with the heavy atoms at their crystal positions.

    python scripts/zipper_scan.py <system> [stage ...]     # wt, scan, dft, core
Resumable: results/zipper_<system>.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemoqbit.zipper import build, protonate
from chemoqbit.enzyme.frozen import optimise_frozen
from chemoqbit import sanity
from chemoqbit.enzyme.xtb import energy_grad

K = 627.509474
SOLVENT = 'water'
NEIGHBOURS = 0            # one strand per sheet; 1 adds a stacked strand either side
XC, BASIS = 'wb97x-d3bj', 'def2-svp'
SYSTEMS = {
    'tau':    dict(pdb='rcsb_2on9/2ON9.pdb', mate='B', note='tau PHF6 306-VQIVYK-311'),
    'asyn47': dict(pdb='rcsb_4znn/4ZNN.pdb', mate=None,
                   note='alpha-synuclein 47-56, the A53T familial Parkinson mutant; '
                        'truncating Thr53 to Ala restores the wild-type sequence'),
    'nacore': dict(pdb='rcsb_4ril/4RIL.pdb', mate=None, note='alpha-synuclein NACore 68-78'),
    'sod1':   dict(pdb='rcsb_5wmj/5WMJ.pdb', mate='B',
                   note='SOD1 30-35 KVWGSI, the ALS-linked oligomer-forming segment '
                        '(Charcot disease). Sangwan et al., Protein Sci 2018, 1.40 A'),
    'sod1ct': dict(pdb='rcsb_4nip/4NIP.pdb', mate=None,
                   note='SOD1 147-153 GVIGIAQ, a C-terminal amyloid segment; '
                        'Ivanova et al., PNAS 2013, 1.90 A'),
}
# Which model the DFT cross-check runs on. None means the WHOLE sheet pair, a
# list means that contiguous residue window with the waters removed.
#
# wb97x-d3bj is range separated at omega = 0.3, so it needs TWO integral files,
# not one (see cderi_bytes). At 788 GB of NVMe, 743 GiB free, the 70 % budget is
# 520 GiB and both SOD1 pairs fit whole:
#
#                             atoms  charge  nao   peak GiB   budget 520.2
#   sod1     WHOLE PAIR        262     +2    2246    403.1       fits
#   sod1ct   WHOLE PAIR        246      0    2112    335.6       fits
#   tau      3-6               190     +2    1616    149.6       (whole: 529.7)
#   asyn47   51-56             184      0    1604    147.8       (whole: 945.7)
#   nacore   72-78             208      0    1814    213.8       (whole: 865.9)
#
# The two SOD1 systems being whole is the point: the check now applies to the
# same model as stage_wt, so its DFT interaction energy is directly comparable to
# the E_int the paper quotes, and the xTB number the stage recomputes should
# reproduce res['wt']['E_int'] -- a free consistency test that a reduced model
# could not give. Earlier windows could not do this. Lys1 carries NZ+ on each
# strand, so the whole sod1 pair is a dication where every window of it was
# neutral, and the two were never the same quantity.
#
# tau misses by 9.5 GiB. Not worth loosening the budget for: the budget protects
# als-approach and the OS, which share this filesystem, and tau has not run its
# wt or scan stages yet. The three remaining windows keep the side chains the
# xTB scan found to hold each interface together.
#
# These are fixed here rather than computed at run time on purpose -- choosing
# from whatever disk happens to be free would make the model, and so the number,
# depend on the machine.
CORE = {'sod1':   None,
        'sod1ct': None,
        'tau':    [3, 4, 5, 6],
        'asyn47': [51, 52, 53, 54, 55, 56],
        'nacore': [72, 73, 74, 75, 76, 77, 78]}

S = {}
NAME = None
PDB = None
OUT = None


def save(d):
    tmp = OUT.with_suffix('.json.tmp'); tmp.write_text(json.dumps(d, indent=1)); tmp.replace(OUT)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def pieces(to_alanine=(), keep_residues=None, seed=None):
    c = build(str(PDB), mate=S['mate'], neighbours=NEIGHBOURS, to_alanine=to_alanine,
              keep_residues=keep_residues)
    if c.get('cut_c_terminus'):
        raise ValueError('the window does not reach the deposited C-terminus, so there '
                         'is no OXT: the carboxylate charge and the cap would both be '
                         'invented. Extend the window to the last residue.')
    sym, X, lab, q, sheet, hmap = protonate(c, seed=seed)
    heavy = [i for i, s in enumerate(sym) if s != 'H']
    X, _ = optimise_frozen(sym, X, charge=q, frozen=heavy, solvent=SOLVENT,
                           maxiter=300, gtol=5e-4)
    X = np.array(X)
    # the relaxed hydrogen network, by the heavy atom that carries it, so the
    # mutants can start from the wild type's basin instead of their own
    hyd = {l: [X[i].tolist() for i in idx] for l, idx in hmap.items()}
    return sym, X, q, sheet, lab, hyd


def interaction(sym, X, q, sheet):
    """E(both) - E(sheet 0) - E(sheet 1); each sheet carries half the charge."""
    E = lambda s, x, qq: float(energy_grad(s, x, qq, solvent=SOLVENT)[0])
    half = q // 2
    assert 2 * half == q, f'charge {q} does not split evenly between the sheets'
    e_all = E(sym, X, q)
    out = {}
    for s_ in (0, 1):
        i = np.flatnonzero(sheet == s_)
        out[s_] = E([sym[k] for k in i], X[i], half)
    return (e_all - out[0] - out[1]) * K, dict(E_both=e_all, E_sheet0=out[0], E_sheet1=out[1],
                                               E_sheet_asymmetry=(out[0] - out[1]) * K)


def check_sheets(res, parts, label):
    """A symmetry-generated mate must give the two fragments the same energy.

    Not a diagnostic nicety: the interaction energy is a difference of numbers two
    hundred times its own size, so a per-sheet difference passes into it whole. In
    4NIP it was 36 kcal/mol against an interaction energy of -3.4, which is how two
    alanine ddE values reached -31 and -28 kcal/mol.
    """
    if isinstance(S['mate'], str):
        return True            # a second deposited chain is a different fragment
    return sanity.record(res, 'sheet_equivalence', abs(parts['E_sheet_asymmetry']),
                         label=label)


def residues(lab):
    """(number, three-letter name) of each residue of the first sheet, in order.

    It used to return the label with its number still attached ("GLY4"), which
    broke two things at once: the display name came out "GLY44" because the number
    was then appended again, and the Gly/Ala skip below compared "GLY4" against
    "GLY" and never matched. SOD1 30-35 is KVWGSI, so its fourth residue is a
    glycine, and mutating a glycine to alanine ADDS a methyl rather than removing
    one. It ran anyway and gave ddE = -55.37 kcal/mol, which the audit caught.
    """
    out = {}
    for l in lab:
        strand, res, _nm = l.split(':')
        n = int(''.join(c for c in res if c.isdigit()))
        out[n] = ''.join(c for c in res if not c.isdigit())
    return sorted(out.items())


def stage_wt(res):
    if 'wt' in res:
        return
    t0 = time.time()
    from chemoqbit.zipper import mating_sheet
    if S['mate'] is None:
        _R, _t, desc = mating_sheet(str(PDB))
        res['interface'] = desc
        print(f"  interface: symmetry op {desc['op']} cell {desc['cell']}, buries "
              f"{desc['buried']:.1f} A^2, side-chain fraction {desc['side_fraction']:.2f}, "
              f"interfacial waters {desc['interfacial_waters']}", flush=True)
    sym, X, q, sheet, lab, hyd = pieces()
    ei, parts = interaction(sym, X, q, sheet)
    check_sheets(res, parts, f'{NAME} wild type sheet pair')
    res['wt'] = dict(atoms=len(sym), charge=q, neighbours=NEIGHBOURS, E_int=ei,
                     residues=[f'{r}{n}' for n, r in residues(lab)],
                     seconds=time.time() - t0, **parts)
    res['geometry'] = dict(labels=lab, hydrogens=hyd)
    save(res)
    sanity.record(res, 'interaction_energy', ei, label='zipper wild type', fatal=True)
    print(f'wild type: {len(sym)} atoms, charge {q:+d}, E_int {ei:+.2f} kcal/mol '
          f'({time.time() - t0:.0f}s)', flush=True)


def stage_scan(res):
    res.setdefault('alanine', {})
    nums = [n for n, r in residues(res['geometry']['labels'])]
    names = dict(residues(res['geometry']['labels']))
    for n in nums:
        key = f'{names[n]}{n}'
        # HOH has no side chain at all, and mutate_to_alanine on a water deletes
        # atoms that are the whole molecule: it returned ddE -9.26 kcal/mol for
        # HOH102, which is a water being removed from the interface and not an
        # alanine substitution. Gly and Ala have no side chain beyond CB.
        if key in res['alanine'] or names[n] in ('GLY', 'ALA', 'HOH'):
            if names[n] == 'HOH':
                res['alanine'].setdefault(key, dict(skipped='a water, not a residue'))
            elif names[n] in ('GLY', 'ALA'):
                res['alanine'].setdefault(key, dict(skipped='no side chain beyond CB'))
            continue
        t0 = time.time()
        try:
            sym, X, q, sheet, lab, _hyd = pieces(
                to_alanine=[n], seed=(res['geometry'].get('hydrogens') or None))
            ei, parts = interaction(sym, X, q, sheet)
            check_sheets(res, parts, f'{key} -> Ala sheet pair')
            res['alanine'][key] = dict(E_int=ei, ddE=ei - res['wt']['E_int'], charge=q,
                                       atoms=len(sym), seconds=time.time() - t0, **parts)
            sanity.record(res, 'alanine_ddE', ei - res['wt']['E_int'], label=f'{key} -> Ala')
            print(f"  {key} -> Ala  E_int {ei:+8.2f}  ddE {ei - res['wt']['E_int']:+7.2f} kcal/mol "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            res['alanine'][key] = dict(failed=f'{type(e).__name__}: {e}')
            print(f'  {key} failed: {type(e).__name__}: {e}', flush=True)
        save(res)
    rows = sorted(((k, v['ddE']) for k, v in res['alanine'].items() if 'ddE' in v),
                  key=lambda r: r[1], reverse=True)
    print('side chains whose loss costs most: ' + ', '.join(f'{k} {v:+.1f}' for k, v in rows),
          flush=True)


def cderi_bytes(mol, xc=None):
    """Peak scratch a density-fitted SCF needs for this system, and why it doubles.

    One integral file is naux x nao(nao+1)/2 doubles. This is the check whose
    absence cost about forty CPU-hours: stage_dft called .density_fit() on the
    whole 262-atom sheet pair at def2-svp, which asks for 201 GiB of scratch on a
    99 GiB disk, so every attempt wrote until the filesystem filled at 82.5 GiB,
    died with Errno 28, deleted the file and was restarted by systemd. Seven
    times. Nothing about the failure said "disk": the journal reported a 14 G
    memory peak, which reads like an OOM.

    A RANGE-SEPARATED functional needs TWO such files, not one. wb97x-d3bj is
    range separated at omega = 0.3, and df.DF.range_coulomb() builds a second,
    separate DF object for the long-range operator with its own _dataname
    ('...-lr/<omega>'); both are live through the SCF. The first version of this
    function counted one file, so it passed a core model needing 96.8 GiB as
    needing 48.4, and the run filled the disk a second time -- caught at 70 GiB,
    which is the first file complete plus 21.6 GiB of the second.
    """
    from pyscf import df, dft
    auxmol = df.addons.make_auxmol(mol, df.addons.make_auxbasis(mol, mp2fit=False))
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    one = naux * (nao * (nao + 1) // 2) * 8
    files = 1
    if xc:
        omega, _alpha, _beta = dft.libxc.rsh_coeff(xc)
        if omega:
            files = 2
    return one * files, nao, naux


def scratch_free_bytes():
    """Free space where PySCF will write, which is TMPDIR and not the repo."""
    import os, shutil, tempfile
    return shutil.disk_usage(os.environ.get('TMPDIR') or tempfile.gettempdir()).free


def check_fits(mol, tag, headroom=0.70, xc=None):
    """Refuse a DF calculation that cannot finish, with the arithmetic in the message."""
    xc = XC if xc is None else xc
    need, nao, naux = cderi_bytes(mol, xc=xc)
    free = scratch_free_bytes()
    g = 1024 ** 3
    msg = (f'{tag}: {mol.natm} atoms, nao {nao}, naux {naux} -> cderi '
           f'{need / g:.1f} GiB peak, scratch free {free / g:.1f} GiB')
    if need > headroom * free:
        raise RuntimeError(msg + f' -- refusing: needs more than {headroom:.0%} of free '
                           f'scratch. This cannot finish; it is not worth starting.')
    return msg



SCF_MAX_CYCLE = 60        # the default 50 was reached by the sod1 core SCF after about
                          # 43 iterations and 21 hours. More cycles alone is not the fix,
                          # which is why the ladder below exists, but stopping one cycle
                          # short of a convergence that was nearly there is pure waste.
SCF_VERBOSE = 4           # print every SCF cycle. At verbose=0 a run that is converging
                          # and a run going nowhere look identical from outside: both are
                          # a live process reading the integral file at 200 MB/s. That is
                          # exactly what the stall check could not distinguish, and why a
                          # job was reported as "working" for 21 hours while it drifted
                          # towards max_cycle.


def run_scf(mol, tag, chkdir=None):
    """One SCF, with somewhere to go when it does not converge.

    stage_core and stage_dft set conv_tol and nothing else, so a hard case -- the
    sod1 core model is a dication with six waters and a charged interface -- had
    no help at all: plain DIIS, and a refusal after fifty cycles. Three attempts
    now, each starting from the density the last one reached:

      1. DIIS as before.
      2. A level shift with damping, which separates the frontier orbitals and
         stops the oscillation that keeps a charged, water-rich system cycling.
         The shift is released for a final stretch so the answer is the true
         stationary point and not the shifted one.
      3. Second-order (Newton) SCF, which converges where DIIS will not, at a
         higher cost per step.

    Returns (energy, converged, how). The chkfile means a killed job resumes from
    the last density rather than from the atomic guess.
    """
    from pyscf import dft          # pyscf is imported lazily by the stages that use it
    mol.verbose = SCF_VERBOSE
    mol.stdout = sys.stdout

    def base():
        m = dft.RKS(mol, xc=XC).density_fit()
        m.conv_tol = 1e-8
        m.max_cycle = SCF_MAX_CYCLE
        if chkdir:
            m.chkfile = str(Path(chkdir) / f'scf_{tag}.chk')
        return m

    mf = base()
    dm = None
    if chkdir and Path(str(mf.chkfile)).exists():
        try:
            dm = mf.from_chk(str(mf.chkfile))
            print(f'  scf {tag}: resuming from {mf.chkfile}', flush=True)
        except Exception as e:
            print(f'  scf {tag}: stored density unusable ({e}); starting fresh', flush=True)
            dm = None
    e = float(mf.kernel(dm0=dm))
    if mf.converged:
        return e, True, 'diis'

    print(f'  scf {tag}: DIIS did not converge in {SCF_MAX_CYCLE}; '
          f'retrying with a level shift', flush=True)
    dm = mf.make_rdm1()
    mf = base()
    mf.level_shift, mf.damp = 0.4, 0.3
    e = float(mf.kernel(dm0=dm))
    # release the shift: a shifted SCF converges to the shifted problem
    dm = mf.make_rdm1()
    mf = base()
    e = float(mf.kernel(dm0=dm))
    if mf.converged:
        return e, True, 'level-shift then released'

    print(f'  scf {tag}: still not converged; going second-order', flush=True)
    dm = mf.make_rdm1()
    mf = base().newton()
    mf.max_cycle = 100
    e = float(mf.kernel(dm0=dm))
    return e, bool(mf.converged), 'newton' if mf.converged else 'failed'


def stage_dft(res):
    """A DFT check of the wild-type interface energy."""
    if 'dft' in res:
        return
    from pyscf import gto, dft
    # size first, on composition alone: relaxing for two hours and only then
    # discovering the integral file does not fit is how this stage wasted about
    # forty CPU-hours
    sym0, X0, _l, q0, _sh, _hm = core_model(None)
    probe = gto.M(atom=[(sym0[i], X0[i]) for i in range(len(sym0))], basis=BASIS,
                  charge=int(q0), unit='Angstrom', verbose=0)
    print('  ' + check_fits(probe, 'both (before relaxing)'), flush=True)
    sym, X, q, sheet, lab, _hyd = pieces(seed=(res.get('geometry') or {}).get('hydrogens'))
    out = {}
    half = q // 2
    for tag, idx, qq in (('both', np.arange(len(sym)), q),
                         ('sheet0', np.flatnonzero(sheet == 0), half),
                         ('sheet1', np.flatnonzero(sheet == 1), half)):
        t0 = time.time()
        mol = gto.M(atom=[(sym[i], X[i]) for i in idx], basis=BASIS, charge=int(qq),
                    unit='Angstrom', verbose=0)
        print('  ' + check_fits(mol, tag), flush=True)
        e, ok, how = run_scf(mol, tag, chkdir=OUT.parent)
        out[tag] = dict(E=e, converged=ok, how=how, atoms=len(idx),
                        seconds=time.time() - t0)
        res['dft'] = out; save(res)
        print(f"  dft {tag}: {len(idx)} atoms, E {out[tag]['E']:.6f} ({time.time() - t0:.0f}s)",
              flush=True)
    if not all(out[t]['converged'] for t in ('both', 'sheet0', 'sheet1')):
        # stage_core has always refused here; this stage recorded the flag and used
        # the energy anyway, which is the one case where a fragment energy is
        # arbitrary rather than merely imprecise
        out['note'] = 'at least one SCF did not converge; no interaction energy reported'
        res['dft'] = out; save(res)
        print('an SCF did not converge -- not reporting an interaction energy', flush=True)
        return
    out['E_int'] = (out['both']['E'] - out['sheet0']['E'] - out['sheet1']['E']) * K
    out['E_sheet_asymmetry'] = (out['sheet0']['E'] - out['sheet1']['E']) * K
    check_sheets(res, out, f'{NAME} wild type sheet pair, DFT')
    res['dft'] = out; save(res)
    print(f"DFT E_int {out['E_int']:+.2f} kcal/mol (gas phase)", flush=True)


def core_model(keep):
    """The reduced model's composition, without relaxing anything.

    Basis size depends only on which elements are present, so the size check can
    run before the hydrogen relaxation rather than after it. stage_dft got this
    the wrong way round: it relaxed for two hours and only then discovered that
    the integral file would not fit.
    """
    c = build(str(PDB), mate=S['mate'], neighbours=NEIGHBOURS, keep_residues=keep)
    if c.get('cut_c_terminus'):
        raise ValueError('window does not reach the deposited C-terminus: no OXT, so '
                         'the carboxylate charge and the cap would both be invented')
    return protonate(c)


def stage_core(res):
    """xTB against wB97X-D3BJ on a core small enough that the DF file fits the disk.

    Resumable per SCF. Each of the three energies is hours of work and about 48 GiB
    of integrals, and this job has a history of being killed part way through, so
    the geometry is stored with the result: a restart reuses it instead of relaxing
    again, which also guarantees the DFT and the xTB numbers refer to the same
    coordinates. Comparing them across two relaxations would not be a comparison.
    """
    from pyscf import gto, dft
    if NAME not in CORE:
        print('no DFT cross-check model defined for %s, skipping' % NAME, flush=True)
        return
    keep = CORE[NAME]          # None means the whole sheet pair
    out = res.get('core') or {}
    if out.get('dft_E_int') is not None:
        return
    if out and out.get('window', 'absent') != keep:
        print('stored cross-check used model %s, now %s -- starting over'
              % (out.get('window'), keep), flush=True)
        out = {}

    # size first, on composition alone, before anything expensive
    sym0, X0, lab0, q0, sheet0, _hm = core_model(keep)
    probe = gto.M(atom=[(sym0[i], X0[i]) for i in range(len(sym0))], basis=BASIS,
                  charge=int(q0), unit='Angstrom', verbose=0)
    print('  ' + check_fits(probe, 'both (before relaxing)'), flush=True)

    if 'xtb' in out and out.get('coords'):
        sym, X, q = out['symbols'], np.array(out['coords']), out['charge']
        sheet, lab = np.array(out['sheet']), out['labels']
        ei_xtb = out['xtb']['E_int']
        print('core %s: resuming on the stored geometry, %d atoms' % (NAME, len(sym)),
              flush=True)
    else:
        t0 = time.time()
        sym, X, q, sheet, lab, _hyd = pieces(
            keep_residues=keep, seed=(res.get('geometry') or {}).get('hydrogens'))
        names = dict(residues(lab))
        ei_xtb, parts_xtb = interaction(sym, X, q, sheet)
        check_sheets(res, parts_xtb, f'{NAME} core sheet pair')
        out = dict(window=keep, residues=[f'{names[n]}{n}' for n in sorted(names)],
                   atoms=len(sym), charge=int(q), symbols=list(sym),
                   coords=[list(map(float, r)) for r in X], sheet=[int(v) for v in sheet],
                   labels=list(lab), xtb=dict(E_int=ei_xtb, **parts_xtb))
        res['core'] = out; save(res)
        print('core %s: %d atoms, charge %+d, residues %s'
              % (NAME, len(sym), q, ' '.join(out['residues'])), flush=True)
        print('  xTB  E_int %+8.2f kcal/mol (%.0fs)' % (ei_xtb, time.time() - t0), flush=True)

    half = q // 2
    E = dict(out.get('dft') or {})
    for tag, idx, qq in (('both', np.arange(len(sym)), q),
                         ('sheet0', np.flatnonzero(np.array(sheet) == 0), half),
                         ('sheet1', np.flatnonzero(np.array(sheet) == 1), half)):
        if tag in E and E[tag].get('converged'):
            print('  dft %-6s already done, E %.6f' % (tag, E[tag]['E']), flush=True)
            continue
        t1 = time.time()
        mol = gto.M(atom=[(sym[i], X[i]) for i in idx], basis=BASIS, charge=int(qq),
                    unit='Angstrom', verbose=0)
        print('  ' + check_fits(mol, tag), flush=True)
        e, ok, how = run_scf(mol, tag, chkdir=OUT.parent)
        E[tag] = dict(E=e, converged=ok, how=how,
                      atoms=len(idx), seconds=time.time() - t1)
        out['dft'] = E; res['core'] = out; save(res)
        print('  dft %-6s %d atoms  E %.6f  converged %s  (%.0fs)'
              % (tag, len(idx), E[tag]['E'], E[tag]['converged'], time.time() - t1),
              flush=True)

    if not all(E.get(t, {}).get('converged') for t in ('both', 'sheet0', 'sheet1')):
        out['note'] = 'at least one SCF did not converge; no interaction energy reported'
        res['core'] = out; save(res)
        print('an SCF did not converge -- not reporting an interaction energy', flush=True)
        return
    ei_dft = (E['both']['E'] - E['sheet0']['E'] - E['sheet1']['E']) * K
    out['dft_E_int'] = ei_dft
    out['dft_E_sheet_asymmetry'] = (E['sheet0']['E'] - E['sheet1']['E']) * K
    check_sheets(res, dict(E_sheet_asymmetry=out['dft_E_sheet_asymmetry']),
                 f'{NAME} core sheet pair, DFT')
    out['xtb_minus_dft'] = ei_xtb - ei_dft
    # the full sheet pair for contrast: the core is a neutral 4-residue dry model and
    # the full model is a dication with six waters, so these are not the same quantity
    # and the core must not be read as a validation of the full number
    if 'wt' in res:
        out['full_model_xtb_E_int'] = res['wt']['E_int']
        out['core_fraction_of_full'] = ei_xtb / res['wt']['E_int'] if res['wt']['E_int'] else None
    res['core'] = out; save(res)
    sanity.record(res, 'interaction_energy', ei_dft, label=f'{NAME} core DFT')
    print('', flush=True)
    print('  xTB          E_int %+8.2f kcal/mol' % ei_xtb, flush=True)
    print('  wB97X-D3BJ   E_int %+8.2f kcal/mol  (%s, gas phase)' % (ei_dft, BASIS), flush=True)
    print('  xTB - DFT          %+8.2f kcal/mol' % (ei_xtb - ei_dft), flush=True)


STAGES = dict(wt=stage_wt, scan=stage_scan, dft=stage_dft, core=stage_core)


def main(system, names):
    global S, PDB, OUT, NAME
    NAME = system
    S = SYSTEMS[system]
    PDB = ROOT / 'data' / S['pdb']
    OUT = ROOT / 'results' / f'zipper_{system}.json'
    print(f"system {system}: {S['note']}", flush=True)
    res = load()
    for n in names:
        print(f'=== {n}', flush=True)
        STAGES[n](res)
        res['_stages_done'] = sorted(set(res.get('_stages_done', [])) | {n})
        save(res)


if __name__ == '__main__':
    # 'dft' is kept reachable by name but is no longer a default: on this machine it
    # cannot run for ANY of the five systems, which is why it is not in this list.
    main(sys.argv[1], sys.argv[2:] or ['wt', 'scan', 'core'])
