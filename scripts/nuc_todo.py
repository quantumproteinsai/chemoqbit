"""Exit 0 if a NUC job still has work to do, 1 if it is complete.

Used as ExecCondition= in the chemoqbit-*.service user units, so a finished
job is skipped at boot instead of re-running (and the notifier does not mail
"finished" again on every boot).

    python scripts/nuc_todo.py tzvp|xc|profile|oniomopt|his147p|cs2d|o4h|qmsize|abstraction|prolyl|pocket_cftr|pocket_ache|zipper_tau|zipper_asyn47|zipper_nacore|any
"""
import json, sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / 'results'
NAMES = ('reactant', 'ts', 'product')


def tzvp():
    r = json.loads((RES / 'ts_cluster_run.json').read_text())
    return not all(f'{n}/def2-tzvp' in r.get('dft', {}) for n in NAMES)


def xc():
    sys.path.insert(0, str(Path(__file__).parent))
    from ts_functional_check import XC, BASIS
    p = RES / 'ts_functional_check.json'
    res = json.loads(p.read_text()) if p.exists() else {}
    return not all(n in res.get(f'{x}/{BASIS}', {}) for x in XC for n in NAMES)


def profile():
    sys.path.insert(0, str(Path(__file__).parent))
    from ts_profile_check import OUT
    return not (OUT.exists() and '_profile' in json.loads(OUT.read_text()))


def oniomopt():
    p = RES / 'ts_oniom_opt.json'
    return not (p.exists() and 'summary' in json.loads(p.read_text()))


def his147p():
    p = RES / 'ts_oniom_opt.his147p.json'
    return not (p.exists() and 'summary' in json.loads(p.read_text()))


def cs2d():
    p = RES / 'ts_cs2d.json'
    return not (p.exists() and 'summary' in json.loads(p.read_text()))


def o4h():
    # complete when the ONIOM summary exists, or when the variant was diagnosed as
    # not viable (ts_o4h_diagnostic.json) and deliberately not pursued
    if (RES / 'ts_o4h_diagnostic.json').exists():
        return False
    p = RES / 'ts_oniom_opt.o4h.json'
    return not (p.exists() and 'summary' in json.loads(p.read_text()))


def qmsize():
    p = RES / 'ts_qmsize.json'
    return not (p.exists() and json.loads(p.read_text()).get('_done'))


STAGES_ABSTRACTION = ('prep', 'acceptor', 'cleave', 'band')


def abstraction():
    p = RES / 'ts_abstraction.json'
    if not p.exists():
        return True
    done = json.loads(p.read_text()).get('_stages_done', [])
    return not all(s in done for s in STAGES_ABSTRACTION)


def prolyl():
    p = RES / 'prolyl.json'
    if not p.exists():
        return True
    done = json.loads(p.read_text()).get('_stages_done', [])
    return not all(x in done for x in ('scan', 'points', 'dft'))


def pocket(system):
    p = RES / f'pocket_{system}.json'
    if not p.exists():
        return True
    done = json.loads(p.read_text()).get('_stages_done', [])
    return not all(x in done for x in ('wt', 'scan', 'dft'))


def zipper(system):
    p = RES / f'zipper_{system}.json'
    if not p.exists():
        return True
    done = json.loads(p.read_text()).get('_stages_done', [])
    return not all(x in done for x in ('wt', 'scan', 'dft'))


if __name__ == '__main__':
    job = sys.argv[1]
    todo = {'tzvp': tzvp, 'xc': xc, 'profile': profile, 'oniomopt': oniomopt, 'his147p': his147p, 'cs2d': cs2d, 'o4h': o4h, 'qmsize': qmsize, 'abstraction': abstraction, 'prolyl': prolyl,
            'pocket_cftr': lambda: pocket('cftr'), 'pocket_ache': lambda: pocket('ache'), 'zipper_tau': lambda: zipper('tau'), 'zipper_asyn47': lambda: zipper('asyn47'),
            'zipper_nacore': lambda: zipper('nacore'),
            'any': lambda: tzvp() or oniomopt() or his147p() or cs2d() or o4h() or qmsize() or abstraction() or prolyl()
                           or pocket('cftr') or pocket('ache') or zipper('tau') or zipper('asyn47') or zipper('nacore')}[job]()
    print(f'{job}: {"work remaining" if todo else "complete, skipping"}')
    sys.exit(0 if todo else 1)
