"""Send one plain-text note through the notifier's credentials, tagged so replies come back.

The subject is prefixed with [chemoqbit] unless it already carries it, because
read_replies.py filters on that tag by default and a reply keeps it under "Re:".
That is what lets a mail answer a mail without opening the rest of the mailbox.

    python scripts/send_note.py --to you@gmail.com --subject "charge ladder" --body note.txt
    echo "short message" | python scripts/send_note.py --to you@gmail.com --subject hello

Outgoing mail is stamped X-Chemoqbit-Origin: agent. Sender and recipient are the same
address here, so without that header read_replies.py cannot tell this script's own
output from a human reply, and a polling loop would read its own words back as
instructions.

The password is read from ~/.config/chemoqbit/smtp.ini and never printed.
"""
import argparse, configparser, smtplib, sys
from email.message import EmailMessage
from pathlib import Path

CRED = Path.home() / '.config' / 'chemoqbit' / 'smtp.ini'
TAG = '[chemoqbit]'
ORIGIN = 'X-Chemoqbit-Origin'   # stamped on outgoing mail; the reader skips what it finds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--to', required=True)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--body', help='file to send; omit to read stdin')
    ap.add_argument('--no-tag', action='store_true', help='leave the subject alone')
    a = ap.parse_args()

    if not CRED.exists():
        sys.exit(f'no credentials at {CRED}')
    c = configparser.ConfigParser(); c.read(CRED)
    unquote = lambda s: s.strip().strip('"\'')
    user, pw = unquote(c['smtp']['user']), unquote(c['smtp']['password'])

    subject = a.subject if (a.no_tag or TAG in a.subject) else f'{TAG} {a.subject}'
    text = Path(a.body).read_text(encoding='utf-8') if a.body else sys.stdin.read()

    m = EmailMessage()
    m['From'], m['To'], m['Subject'] = user, a.to, subject
    # This account mails itself: sender and recipient are the same address, so the
    # reader cannot tell Claude's own mail from a reply by address alone. Stamp it.
    m[ORIGIN] = 'agent'
    m.set_content(text)
    with smtplib.SMTP('smtp.gmail.com', 587, timeout=60) as s:
        s.starttls(); s.login(user, pw); s.send_message(m)
    print(f'sent to {a.to}: {len(text)} characters, subject {subject!r}')


if __name__ == '__main__':
    main()
