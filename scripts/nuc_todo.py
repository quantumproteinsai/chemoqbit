"""Exit 0 if a NUC job still has work to do, 1 if it is complete.

Used as ExecCondition= in the chemoqbit-*.service user units, so a finished
job is skipped at boot instead of re-running (and the notifier does not mail
"finished" again on every boot).

    python scripts/nuc_todo.py tzvp|xc|any
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


if __name__ == '__main__':
    job = sys.argv[1]
    todo = {'tzvp': tzvp, 'xc': xc, 'any': lambda: tzvp() or xc()}[job]()
    print(f'{job}: {"work remaining" if todo else "complete, skipping"}')
    sys.exit(0 if todo else 1)
