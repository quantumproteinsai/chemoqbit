"""Exit 0 if a NUC job still has work to do, 1 if it is complete.

Used as ExecCondition= in the chemoqbit-*.service user units, so a finished
job is skipped at boot instead of re-running (and the notifier does not mail
"finished" again on every boot).

    python scripts/nuc_todo.py tzvp|xc|profile|oniomopt|any
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


if __name__ == '__main__':
    job = sys.argv[1]
    todo = {'tzvp': tzvp, 'xc': xc, 'profile': profile, 'oniomopt': oniomopt, 'any': lambda: tzvp() or oniomopt()}[job]()
    print(f'{job}: {"work remaining" if todo else "complete, skipping"}')
    sys.exit(0 if todo else 1)
