#!/usr/bin/env python
"""Render one of this project's Markdown manuscripts as a PDF.

Written because pandoc, wkhtmltopdf, weasyprint and a LaTeX engine are all absent
here and installing any of them needs root, while reportlab installs from pip.
It handles the subset of Markdown these manuscripts actually use: YAML front
matter, ATX headings, paragraphs, bold/italic/code spans, fenced code, pipe
tables, bullet and numbered lists, horizontal rules and inline links. It is not a
general converter and does not pretend to be one.

    python scripts/md2pdf.py paper/tdp43_threshold.md out.pdf
"""
import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, KeepTogether, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

# Unicode super- and subscripts are not in reportlab's default font encoding and
# render as black boxes: "0.0047 h-1" came out as a filled square. They are
# translated to reportlab's own <super>/<sub> markup instead, which the built-in
# fonts do support. A handful of maths symbols outside Latin-1 go to ASCII.
SUP = dict(zip('\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b\u207a',
                '0123456789-+'))
SUB = dict(zip('\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u208b'
               '\u1d2e\u1d3a',
                '0123456789-' 'BN'))
PLAIN = {'\u2212': '-', '\u2248': '~', '\u2264': '<=', '\u2265': '>=',
         '\u2032': "'", '\u2033': '"', '\u00d7': 'x',
         '\u2010': '-', '\u2500': '\u2014'}

# Everything else outside Latin-1 -- Greek, arrows, angle brackets, the dotted e
# of Leinartaite -- has no glyph in the built-in Times faces and comes out as a
# filled square. Twenty-three of those squares were the gamma of "S-gamma" in the
# SOD1 manuscript, in a sentence about which sulfur attacks which. They are drawn
# from DejaVu Serif instead, which is a close enough serif to sit in a Times line
# and is the only locally installed family that carries the glyphs. It has no
# italic, so an italicised run of them comes out upright; none of the manuscripts
# does that today, and convert() says so if one ever starts.
_DEJAVU = '/usr/share/fonts/truetype/dejavu/'
FALLBACK = 'DejaVuSerif'
try:
    pdfmetrics.registerFont(TTFont(FALLBACK, _DEJAVU + 'DejaVuSerif.ttf'))
    pdfmetrics.registerFont(TTFont(FALLBACK + '-Bold', _DEJAVU + 'DejaVuSerif-Bold.ttf'))
    pdfmetrics.registerFontFamily(FALLBACK, normal=FALLBACK, bold=FALLBACK + '-Bold',
                                  italic=FALLBACK, boldItalic=FALLBACK + '-Bold')
except Exception as exc:                                    # pragma: no cover
    print('warning: no fallback face (%s); exotic characters will be boxes' % exc)
    FALLBACK = None

DRAWN = set()           # what convert() had to borrow the fallback face for


def _latin1(ch):
    try:
        ch.encode('cp1252')
        return True
    except UnicodeEncodeError:
        return False


def _fallback(t):
    """Wrap every run of non-Latin-1 characters in the fallback face.

    The markup already inserted by _scripts is pure ASCII, so it is never
    swallowed by a run."""
    if FALLBACK is None:
        return t
    out, i, n = [], 0, len(t)
    while i < n:
        if _latin1(t[i]):
            out.append(t[i]); i += 1
            continue
        run = ''
        while i < n and not _latin1(t[i]):
            run += t[i]; i += 1
        DRAWN.update(run)
        out.append('<font face="%s">%s</font>' % (FALLBACK, run))
    return ''.join(out)


def _scripts(t):
    """Turn runs of unicode super/subscripts into <super>/<sub> runs."""
    for tbl, tag in ((SUP, 'super'), (SUB, 'sub')):
        out, i = [], 0
        while i < len(t):
            if t[i] in tbl:
                run = ''
                while i < len(t) and t[i] in tbl:
                    run += tbl[t[i]]; i += 1
                out.append('<%s>%s</%s>' % (tag, run, tag))
            else:
                out.append(t[i]); i += 1
        t = ''.join(out)
    for k, v in PLAIN.items():
        t = t.replace(k, v)
    return _fallback(t)


SS = getSampleStyleSheet()
BODY = ParagraphStyle('body', parent=SS['BodyText'], fontName='Times-Roman',
                      fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
H1 = ParagraphStyle('h1', parent=BODY, fontName='Times-Bold', fontSize=16,
                    leading=20, spaceBefore=4, spaceAfter=10, alignment=0)
H2 = ParagraphStyle('h2', parent=BODY, fontName='Times-Bold', fontSize=12.5,
                    leading=16, spaceBefore=14, spaceAfter=6, alignment=0)
H3 = ParagraphStyle('h3', parent=BODY, fontName='Times-Italic', fontSize=11,
                    leading=14, spaceBefore=10, spaceAfter=4, alignment=0)
META = ParagraphStyle('meta', parent=BODY, fontName='Times-Italic', fontSize=9.5,
                      leading=13, alignment=0, spaceAfter=3)
CODE = ParagraphStyle('code', parent=BODY, fontName='Courier', fontSize=8.5,
                      leading=11, alignment=0, leftIndent=10, spaceAfter=8,
                      backColor=colors.Color(.96, .96, .96))
LI = ParagraphStyle('li', parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
CELL = ParagraphStyle('cell', parent=BODY, fontSize=8.5, leading=11, alignment=0,
                      spaceAfter=0)
CELLH = ParagraphStyle('cellh', parent=CELL, fontName='Times-Bold')


def spans(t):
    """Markdown inline -> reportlab inline markup.

    Code spans are lifted out FIRST and put back LAST. Without that, an asterisk
    inside a code span -- scripts/ts_* in the software paper -- is seen by the
    italic matcher, which then closes an <i> across a <font> boundary and the
    parser rejects the whole paragraph."""
    t = _scripts(html.escape(t))
    held = []

    def stash(m):
        held.append(m.group(1))
        return '\x00%d\x00' % (len(held) - 1)

    t = re.sub(r'`([^`]+)`', stash, t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', t)
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 <font size="8">&lt;\2&gt;</font>', t)
    for k, v in enumerate(held):
        t = t.replace('\x00%d\x00' % k, '<font name="Courier" size="8.5">%s</font>' % v)
    return t


def table(rows):
    body = [[Paragraph(spans(c), CELLH if i == 0 else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(body, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.25, colors.Color(.75, .75, .75)),
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(.93, .93, .93)),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    return t


def figure(path, src, maxw=16.6 * cm, maxh=19 * cm, dpi=300.0):
    """An embedded figure, fitted to the text block but never enlarged past `dpi`.

    The manuscripts write their figure paths relative to themselves
    (figures/fig1_...png beside the .md), so that is what they resolve against.
    A missing file becomes a visible note rather than a traceback: a manuscript
    should still render when a figure has not been regenerated yet, and a line
    saying so in the PDF is harder to overlook than an error the author has
    learned to rerun past.

    The third term of the scale is the one that matters. getSize() returns
    PIXELS, so a 1.0 cap compares pixels against points and never binds; every
    figure was being stretched to the full text width regardless of what it was
    rendered at. The SOD1 figures are written by matplotlib at 300 dpi, and
    blowing an 1339 px panel up to 16.6 cm served it at 205 ppi -- below what a
    preprint server asks for, from a file that already met it. Capping the scale
    at 72/dpi points per pixel draws each figure at the size it was designed
    for, which for those three is 11.3 cm rather than 16.6 cm.
    """
    p = Path(src).parent / path
    if not p.exists():
        return Paragraph('<i>[missing figure: %s]</i>' % html.escape(str(path)), BODY)
    iw, ih = ImageReader(str(p)).getSize()
    k = min(maxw / iw, maxh / ih, 72.0 / dpi)
    return Image(str(p), iw * k, ih * k)


def convert(src, dst):
    lines = Path(src).read_text().split('\n')
    flow, i, n = [], 0, len(lines)
    DRAWN.clear()

    # YAML front matter, as the JOSS paper carries
    if lines and lines[0].strip() == '---':
        j = next((k for k in range(1, n) if lines[k].strip() == '---'), None)
        if j:
            for ln in lines[1:j]:
                m = re.match(r'^(title|authors?|date|tags|bibliography):\s*(.*)$', ln)
                if m and m.group(2).strip():
                    v = m.group(2).strip().strip('"\'')
                    flow.append(Paragraph(spans(v), H1 if m.group(1) == 'title' else META))
                elif re.match(r'^\s+-?\s*name:\s*(.+)$', ln):
                    flow.append(Paragraph(spans(re.sub(r'^\s+-?\s*name:\s*', '', ln)), META))
            flow.append(Spacer(1, 8))
            i = j + 1

    while i < n:
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1; continue
        if s.startswith('```'):
            buf = []
            i += 1
            while i < n and not lines[i].strip().startswith('```'):
                buf.append(lines[i]); i += 1
            i += 1
            # the spaces go first: _scripts emits <font face="..."> tags, whose own
            # spaces must not become &nbsp; or the parser rejects the block
            txt = '<br/>'.join(_scripts(html.escape(b).replace(' ', '&nbsp;')) for b in buf)
            flow.append(Paragraph(txt, CODE)); continue
        if re.match(r'^(---|\*\*\*|___)\s*$', s):
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width='100%', thickness=0.5,
                                   color=colors.Color(.7, .7, .7)))
            flow.append(Spacer(1, 6)); i += 1; continue
        m = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)\s*$', s)
        if m:
            flow.append(Spacer(1, 6))
            flow.append(figure(m.group(2), src))
            flow.append(Spacer(1, 4)); i += 1; continue
        m = re.match(r'^(#{1,4})\s+(.*)$', s)
        if m:
            lvl = len(m.group(1))
            flow.append(Paragraph(spans(m.group(2)), H1 if lvl == 1 else H2 if lvl == 2 else H3))
            i += 1; continue
        if s.startswith('|') and i + 1 < n and re.match(r'^\|[\s:|-]+\|?\s*$', lines[i + 1].strip()):
            rows = []
            while i < n and lines[i].strip().startswith('|'):
                r = lines[i].strip().strip('|')
                if not re.match(r'^[\s:|-]+$', r):
                    rows.append([c.strip() for c in r.split('|')])
                i += 1
            flow.append(KeepTogether([table(rows), Spacer(1, 8)])); continue
        m = re.match(r'^[-*]\s+(.*)$', s)
        if m:
            flow.append(Paragraph(spans(m.group(1)), LI, bulletText='•')); i += 1; continue
        m = re.match(r'^(\d+)\.\s+(.*)$', s)
        if m:
            flow.append(Paragraph(spans(m.group(2)), LI, bulletText=m.group(1) + '.'))
            i += 1; continue
        buf = []
        while i < n and lines[i].strip() and not re.match(
                r'^(#{1,4}\s|[-*]\s|\d+\.\s|\||```|!\[|---\s*$)', lines[i].strip()):
            buf.append(lines[i].strip()); i += 1
        flow.append(Paragraph(spans(' '.join(buf)), BODY))

    SimpleDocTemplate(str(dst), pagesize=A4, topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                      leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                      title=Path(src).stem).build(flow)
    if DRAWN:
        print('    %d character(s) borrowed from %s: %s'
              % (len(DRAWN), FALLBACK, ' '.join(sorted(DRAWN))))
    return dst


if __name__ == '__main__':
    out = convert(sys.argv[1], sys.argv[2])
    print('  %s -> %s (%d kB)' % (sys.argv[1], out, Path(out).stat().st_size // 1024))
