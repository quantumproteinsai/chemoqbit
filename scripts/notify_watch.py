"""Mail a notification when a long-running pipeline step finishes or fails.

Watches log files for step-completion lines and a set of processes for exit,
and sends one mail per event through Gmail's submission server
(smtp.gmail.com:587, STARTTLS). Port 25 is blocked on the NUC's network, so
direct delivery is not an option.

Credentials: ~/.config/chemoqbit/smtp.ini (keep it chmod 600)

    [smtp]
    user = you@gmail.com
    password = <16-character Google app password>

Until that file exists, events are kept in results/notify_pending.jsonl and
sent as soon as it appears.

    python scripts/notify_watch.py --to you@gmail.com \
        --watch focal:results/ts_focal_X.log:ts-focal \
        --watch tzvp:results/nuc_X_tzvp.log:ts-tzvp

Each --watch is name:logfile:unit, where unit is the systemd --user unit the
job runs in; its journal says whether it ended cleanly or was killed.
"""
import argparse, configparser, json, re, smtplib, subprocess, time
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PENDING = ROOT / 'results' / 'notify_pending.jsonl'
CRED = Path.home() / '.config' / 'chemoqbit' / 'smtp.ini'
STEP = re.compile(r'^(DFT |focal |ONIOM\(|oniom |active |barrier |KIE |reactant: |product: |'
                  r'\S+: barrier |\S+: scan maximum |ONIOM-OPT |\S+: minimax |qmsize |.*(Error|Traceback|failed))')


def send(to, subject, body):
    if not CRED.exists():
        return False
    c = configparser.ConfigParser(); c.read(CRED)
    m = EmailMessage()
    m['From'], m['To'], m['Subject'] = c['smtp']['user'].strip().strip('"\''), to, subject
    m.set_content(body)
    with smtplib.SMTP('smtp.gmail.com', 587, timeout=30) as s:
        # tolerate a quoted value: configparser keeps quotes as part of it
        pw = c['smtp']['password'].strip().strip('"\'')
        s.starttls(); s.login(c['smtp']['user'].strip().strip('"\''), pw); s.send_message(m)
    return True


def queue(to, subject, body):
    with PENDING.open('a') as f:
        f.write(json.dumps(dict(to=to, subject=subject, body=body, t=time.strftime('%F %T'))) + '\n')


def flush():
    if not PENDING.exists() or not CRED.exists():
        return
    left = []
    for line in PENDING.read_text().splitlines():
        e = json.loads(line)
        try:
            send(e['to'], e['subject'], f"(queued {e['t']})\n\n{e['body']}")
        except Exception as ex:                       # keep it for the next attempt
            left.append(line); print('send failed:', ex, flush=True)
    PENDING.write_text('\n'.join(left) + ('\n' if left else ''))


def alive(unit):
    out = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True)
    return out.stdout.strip() in ('active', 'activating', 'deactivating', 'reloading')


def ended_badly(unit):
    out = subprocess.run(['journalctl', '--user', '-u', unit, '-n', '20', '--no-pager'],
                         capture_output=True, text=True).stdout
    last_start = out.rfind('Started ')
    return re.search(r'Failed with result|oom-kill|status=9/KILL', out[last_start:]) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--to', required=True)
    ap.add_argument('--watch', action='append', required=True, help='name:logfile:systemd-user-unit')
    ap.add_argument('--every', type=int, default=60)
    a = ap.parse_args()
    jobs = []
    for w in a.watch:
        name, log, unit = w.split(':', 2)
        p = ROOT / log
        jobs.append(dict(name=name, log=p, unit=unit, pos=p.stat().st_size if p.exists() else 0, done=False))

    def notify(subject, body):
        print(subject, flush=True)
        try:
            if send(a.to, subject, body):
                return
        except Exception as ex:
            print('send failed, queued:', ex, flush=True)
        queue(a.to, subject, body)

    while not all(j['done'] for j in jobs):
        flush()
        for j in jobs:
            if j['done']:
                continue
            new = []
            if j['log'].exists():
                with j['log'].open() as f:
                    f.seek(j['pos']); txt = f.read(); j['pos'] = f.tell()
                new = [l for l in txt.splitlines() if STEP.match(l) and 'Warning' not in l]
            for l in new:
                notify(f"chemoqbit {j['name']}: {l[:90]}", l)
            if not alive(j['unit']):
                j['done'] = True
                tail = j['log'].read_text().splitlines()[-15:] if j['log'].exists() else []
                ok = not ended_badly(j['unit']) and not any(re.search('Error|Traceback', l) for l in tail)
                notify(f"chemoqbit {j['name']}: job {'finished' if ok else 'STOPPED - check log'}",
                       f"log: {j['log']}\n\nlast lines:\n" + '\n'.join(tail))
        time.sleep(a.every)
    flush()


if __name__ == '__main__':
    main()
