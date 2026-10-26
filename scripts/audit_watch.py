"""Run the audit on a timer and mail when a new violation appears.

A reviewer needs one bad number to reject a paper, so the results are checked
while they are being produced rather than when someone thinks to look. This
runs scripts/audit.py, compares the violations with the last run's, and mails
only what is new -- so a standing problem does not arrive every quarter hour.

    python scripts/audit_watch.py [--to ADDRESS] [--quiet]
Driven by chemoqbit-audit.timer every 15 minutes.
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
STATE = ROOT / 'results' / '.audit_seen.json'


def main():
    ap = argparse.ArgumentParser()
    # no default: a personal address hardcoded here would be published with the
    # repository. The systemd unit passes it, and the notifier's credentials
    # already live outside the repo in ~/.config/chemoqbit/smtp.ini.
    ap.add_argument('--to', required=True,
                    help='recipient for violation mail')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    # the checks are themselves under test: if a known failure stops firing, or a
    # verified-good result starts firing, the checker is broken and says so
    t = subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/test_checks.py')],
                       capture_output=True, text=True)
    r = subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/audit.py')],
                       capture_output=True, text=True)
    if t.returncode:
        bad = [l for l in t.stdout.splitlines() if 'WRONG' in l]
        r.stdout = ('SANITY the checker itself failed its own test cases:\n'
                    + '\n'.join(bad) + '\n\n') + r.stdout
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith('SANITY')]
    seen = set(json.loads(STATE.read_text())) if STATE.exists() else set()
    new = [l for l in lines if l not in seen]
    STATE.write_text(json.dumps(sorted(set(lines) | seen)))
    if not a.quiet:
        print(r.stdout.rstrip() or '(no output)')
    if new:
        body = ('The periodic audit found results outside their physical bounds.\n\n'
                + '\n\n'.join(new)
                + '\n\nFull audit output:\n\n' + r.stdout)
        try:
            from notify_watch import send
            send(a.to, f'chemoqbit audit: {len(new)} new violation(s)', body)
            print(f'mailed {len(new)} new violation(s) to {a.to}')
        except Exception as e:
            print(f'could not mail ({type(e).__name__}: {e}); violations were:\n' + '\n'.join(new))
    return 0          # the mail is the alert; a nonzero exit would only mark the timer failed


if __name__ == '__main__':
    sys.exit(main())
