"""Read replies sent back to the notifier's own mailbox, so a mail can answer a mail.

notify_watch.py sends through Gmail's submission server with an app password. The same
app password works for IMAP, so replies can be read back without a second credential
and without any OAuth flow.

    Credentials: ~/.config/chemoqbit/smtp.ini   (the same file notify_watch.py uses)

The search is deliberately narrow. It never lists the mailbox: it asks for messages
from one address, newer than a few days, optionally filtered by subject, and prints
only those. Nothing is marked read, deleted or moved -- the mailbox is opened readonly,
so running this twice gives the same answer.

    python scripts/read_replies.py --from bartmercier@gmail.com            # [chemoqbit] only
    python scripts/read_replies.py --from bartmercier@gmail.com --subject omics
    python scripts/read_replies.py --from bartmercier@gmail.com --any --days 2

A subject filter is on by default and defaults to the [chemoqbit] tag that outgoing
mail carries, so an ordinary inbox is never swept. --any turns the filter off and has
to be asked for: that address receives personal mail which is none of this script's
business.

Two things Gmail does that are easy to trip over. A filter routes this traffic into
the 'chemoqbit' mailbox rather than leaving it in INBOX, which is why that is the
default here. And Gmail's SEARCH ignores punctuation, so the [chemoqbit] tag matches
any subject containing the bare word -- the notifier's own 'chemoqbit audit:' mails
included. --replies-only cuts it down to actual answers.

Without --full each message is cut at --chars characters, which keeps a long quoted
thread from burying the part that is actually new.
"""
import argparse, configparser, email, imaplib, json, sys
from email.header import decode_header, make_header
from pathlib import Path

CRED = Path.home() / '.config' / 'chemoqbit' / 'smtp.ini'
TAG = '[chemoqbit]'          # outgoing subjects carry it, so replies keep it under "Re:"
BOX = 'chemoqbit'            # a Gmail filter routes this traffic out of INBOX into here
SEEN = Path(__file__).resolve().parents[1] / 'results' / 'replies_seen.json'
QUOTED = ('>', 'On ', '-----Original', '________')


def creds():
    if not CRED.exists():
        sys.exit(f'no credentials at {CRED}')
    c = configparser.ConfigParser(); c.read(CRED)
    unquote = lambda s: s.strip().strip('"\'')
    return unquote(c['smtp']['user']), unquote(c['smtp']['password'])


def body_of(msg):
    """The plain-text part, preferring text/plain over a stripped text/html."""
    if not msg.is_multipart():
        return msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', 'replace')
    for want in ('text/plain', 'text/html'):
        for part in msg.walk():
            if part.get_content_type() == want and 'attachment' not in str(part.get('Content-Disposition')):
                t = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', 'replace')
                if want == 'text/html':
                    import re as _re
                    t = _re.sub(r'<[^>]+>', '', t)
                return t
    return ''


def new_part(text):
    """Drop the quoted original, which is usually most of a reply."""
    out = []
    for line in text.splitlines():
        if line.strip().startswith(QUOTED):
            break
        out.append(line)
    return '\n'.join(out).strip() or text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='sender', required=True, help='only mail from this address')
    ap.add_argument('--subject', default=TAG, help=f'substring the subject must contain (default {TAG!r})')
    ap.add_argument('--any', action='store_true', help='no subject filter: reads unrelated mail too')
    ap.add_argument('--days', type=int, default=7)
    ap.add_argument('--chars', type=int, default=4000, help='truncate each body (0 = no limit)')
    ap.add_argument('--full', action='store_true', help='keep the quoted thread too')
    ap.add_argument('--mailbox', default=BOX, help=f'default {BOX!r}; use INBOX if no filter routes it')
    ap.add_argument('--replies-only', action='store_true', help='only subjects starting with Re:')
    ap.add_argument('--new-only', action='store_true',
                    help='print only messages not seen before, and remember them')
    a = ap.parse_args()

    seen = set()
    if a.new_only and SEEN.exists():
        seen = set(json.loads(SEEN.read_text()))

    user, pw = creds()
    import datetime
    since = (datetime.date.today() - datetime.timedelta(days=a.days)).strftime('%d-%b-%Y')
    crit = ['FROM', f'"{a.sender}"', 'SINCE', since]
    subject = '' if a.any else a.subject
    if subject:
        crit += ['SUBJECT', f'"{subject}"']

    m = imaplib.IMAP4_SSL('imap.gmail.com', 993)
    try:
        m.login(user, pw)
        m.select(a.mailbox, readonly=True)          # readonly: nothing is marked or moved
        typ, data = m.search(None, *crit)
        ids = data[0].split()
        shown = []
        print(f'{len(ids)} message(s) from {a.sender} in {a.mailbox} since {since}'
              + (f' matching subject ~ {subject!r}' if subject else ' (NO subject filter)'))
        for i in ids:
            typ, raw = m.fetch(i, '(BODY.PEEK[])')   # PEEK: does not set \Seen
            msg = email.message_from_bytes(raw[0][1])
            subj = str(make_header(decode_header(msg.get('Subject', '(no subject)'))))
            if a.replies_only and not subj.lower().startswith('re:'):
                continue
            mid = msg.get('Message-ID') or f'{msg.get("Date")}|{subj}'
            if a.new_only:
                if mid in seen:
                    continue
                seen.add(mid)
                shown.append(mid)
            text = body_of(msg)
            if not a.full:
                text = new_part(text)
            if a.chars and len(text) > a.chars:
                text = text[:a.chars] + f'\n[... {len(text) - a.chars} more characters]'
            print('\n' + '=' * 78)
            print(f'date   : {msg.get("Date")}')
            print(f'subject: {subj}')
            print('=' * 78)
            print(text)
        if a.new_only:
            SEEN.parent.mkdir(parents=True, exist_ok=True)
            SEEN.write_text(json.dumps(sorted(seen), indent=0))
            print(f'\n{len(shown)} new message(s) this run; {len(seen)} known in total')
    finally:
        try:
            m.close()
        except Exception:
            pass
        m.logout()


if __name__ == '__main__':
    main()
