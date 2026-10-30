#!/usr/bin/env python
"""Screen a manuscript the way a preprint server screens it, before submitting.

Written after BIORXIV/2026/762551 was declined at screening as "not a complete research
manuscript". Two defects produced that reading and both were mechanical: the
availability statement said the model scripts were not in the public repository when
they had been there since the previous day, and the abstract claimed three falsification
experiments where the paper gives four. Neither needed judgement to catch. This catches
them.

    python scripts/prescreen.py paper.md [--repo DIR] [--tarball FILE]

FATAL  a factual error a screener can verify against the repository or the deposit
WARN   wording or structure that reads as hypothesis rather than finished work
INFO   worth a glance

Exit status is the number of FATAL findings, so it can gate a submission script.
"""
import argparse
import re
import subprocess
import sys
import tarfile
from pathlib import Path

NUMWORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# Phrases a screener reads as "not finished". Each is legitimate in its place; the
# point is to see them deliberately rather than by accident.
HEDGE = [
    (r"\billustrative\b", "calls its own parameters illustrative"),
    (r"\bnot yet\b", "'not yet' - check it is not describing the work's own state"),
    (r"\bpreliminary\b", "'preliminary'"),
    (r"\bfuture work\b", "'future work'"),
    (r"\bremains? to be (seen|determined|shown)\b", "'remains to be ...'"),
    (r"\bwe (speculate|hypothesi[sz]e)\b", "'we speculate' / 'we hypothesise'"),
    (r"\bthis (note|commentary|perspective|opinion)\b", "calls itself a note or commentary"),
    (r"\bproof[- ]of[- ]concept\b", "'proof of concept'"),
]
# An optional adjective between the article and the genre word: ": a threshold model"
# is the case that prompted all this, and a pattern anchored on "a model" misses it.
TITLE_GENRE = [(r":\s*an?\s+(?:\w+\s+){0,2}"
                r"(model|hypothesis|perspective|commentary|review|proposal|note|framework)\b",
                "the subtitle names a genre rather than a finding")]
PLACEHOLDER = [r"XXXX+", r"\bTODO\b", r"\bTBD\b", r"\[citation needed\]", r"\bFIXME\b",
               r"\bplaceholder\b"]


def sections(md):
    out, cur, buf = {}, "(preamble)", []
    for line in md.split("\n"):
        if line.startswith("## ") or line.startswith("# "):
            out[cur] = "\n".join(buf); cur, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    out[cur] = "\n".join(buf)
    return out


def check_references(md, add):
    body, _, refs = md.partition("## References")
    if not refs:
        add("WARN", "no '## References' section found"); return
    refs = re.split(r"\n## ", refs)[0]
    have = {int(m) for m in re.findall(r"^(\d+)\.", refs, re.M)}
    cited = set()
    for grp in re.findall(r"\[([\d,\s–-]+)\]", body):
        for tok in (t.strip() for t in grp.split(",")):
            if tok.isdigit():
                cited.add(int(tok))
            elif re.match(r"^\d+\s*[–-]\s*\d+$", tok):
                a, b = re.split(r"[–-]", tok); cited.update(range(int(a), int(b) + 1))
    if not have:
        add("WARN", "reference list parsed as empty"); return
    for c in sorted(cited - have):
        add("FATAL", f"citation [{c}] has no entry in the reference list")
    for h in sorted(have - cited):
        add("WARN", f"reference {h} is listed but never cited")
    for g in sorted(set(range(1, max(have) + 1)) - have):
        add("INFO", f"reference numbering skips {g}")


def check_counts(md, add):
    """The abstract says three; the section heading says four."""
    secs = sections(md)
    abstract = secs.get("Abstract", "")
    heads = [h for h in secs if re.match(r"^\d+\.", h)]
    for h in heads:
        m = re.match(r"^\d+\.\s+(\w+)\s+(\w+)", h)
        if not m or m.group(1).lower() not in NUMWORD:
            continue
        n_head, noun = NUMWORD[m.group(1).lower()], m.group(2).lower().rstrip("s")
        for am in re.finditer(r"\b(" + "|".join(NUMWORD) + r")\s+(\w+)", abstract, re.I):
            if am.group(2).lower().rstrip("s") != noun:
                continue
            n_abs = NUMWORD[am.group(1).lower()]
            if n_abs != n_head:
                add("FATAL", f"abstract says {am.group(1)} {am.group(2)}, "
                             f"but section heading says {m.group(1)} ({h})")


def check_availability(md, repo, tarball, add):
    """Every file the paper names must exist where the paper says it does."""
    secs = sections(md)
    avail = next((v for k, v in secs.items() if "availability" in k.lower()), "")
    if not avail:
        add("WARN", "no data/code availability section found"); return
    named = set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:py|json|md|cff|toml|tsv|csv))`", avail))
    if not named:
        add("INFO", "availability section names no files in backticks")
    for claim, why in ((r"does not yet carry", "says the repository does NOT carry files"),
                       (r"\bin the deposit only\b", "says files are deposit-only")):
        if re.search(claim, avail):
            add("WARN", f"availability statement {why} - verify that is still true")
    if repo:
        tracked = set(subprocess.run(["git", "-C", str(repo), "ls-files"],
                                     capture_output=True, text=True).stdout.split())
        base = {Path(t).name for t in tracked}
        for f in sorted(named):
            if Path(f).name not in base:
                add("FATAL", f"availability names `{f}` but it is not tracked in {repo}")
    if tarball and Path(tarball).exists():
        with tarfile.open(tarball) as t:
            inside = {Path(n).name for n in t.getnames()}
        for f in sorted(named):
            if Path(f).name not in inside:
                add("FATAL", f"availability names `{f}` but it is not in {Path(tarball).name}")
    for doi in set(re.findall(r"zenodo\.(\d+|X+)", avail)):
        if set(doi) == {"X"}:
            add("FATAL", "availability still contains a placeholder DOI")


def check_wording(md, add):
    title = md.split("\n")[0].lstrip("# ").strip()
    for pat, why in TITLE_GENRE:
        if re.search(pat, title, re.I):
            add("WARN", f"title: {why} - '{title}'")
    secs = sections(md)
    for name, text in secs.items():
        if name.lower().startswith(("competing", "ai usage", "funding", "references")):
            continue
        for pat, why in HEDGE:
            for m in re.finditer(pat, text, re.I):
                line = text[:m.start()].count("\n")
                ctx = text.split("\n")[line].strip()[:84]
                add("WARN", f"[{name}] {why}: \"{ctx}\"")
    for pat in PLACEHOLDER:
        for m in re.finditer(pat, md, re.I):
            add("FATAL", f"placeholder left in text: {m.group(0)!r}")


def check_abstract(md, add):
    secs = sections(md)
    a = secs.get("Abstract", "").strip()
    if not a:
        add("WARN", "no abstract section found"); return
    w = len(a.split())
    if w > 400:
        add("WARN", f"abstract is {w} words; most servers cap near 350")
    elif w < 100:
        add("INFO", f"abstract is only {w} words")
    # A results abstract either says so in words or carries numbers. Requiring both
    # produced a false alarm on an abstract containing "62.31 kcal/mol", and a screener
    # that cries wolf on a results-dense abstract trains its author to ignore it.
    verbs = re.search(r"\bwe (show|find|report|measure|derive|put|give|exclude|price|"
                      r"calculate|estimate|demonstrate|observe)\b|\bhere we\b", a, re.I)
    numbers = len(re.findall(r"\d+(?:\.\d+)?\s*(?:%|\u00c5|kcal|nm|\u00b5m|h\u207b\u00b9|"
                             r"-fold|fold|\u00b0C|Da|kDa)", a))
    if not verbs and numbers < 3:
        add("WARN", f"abstract states no result in words and carries {numbers} quantities "
                    f"- a screener looks for a finding")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manuscript")
    ap.add_argument("--repo", help="public git repository the paper points readers to")
    ap.add_argument("--tarball", help="the deposited archive the paper cites")
    a = ap.parse_args()
    md = Path(a.manuscript).read_text(encoding="utf-8")

    found = []
    add = lambda lvl, msg: found.append((lvl, msg))
    check_references(md, add)
    check_counts(md, add)
    check_availability(md, a.repo, a.tarball, add)
    check_wording(md, add)
    check_abstract(md, add)

    print(f"prescreen: {Path(a.manuscript).name}")
    print(f"  title: {md.split(chr(10))[0].lstrip('# ').strip()}\n")
    n = {"FATAL": 0, "WARN": 0, "INFO": 0}
    for lvl in ("FATAL", "WARN", "INFO"):
        rows = [m for l, m in found if l == lvl]
        n[lvl] = len(rows)
        for m in rows:
            print(f"  {lvl:<6} {m}")
    print(f"\n  {n['FATAL']} fatal, {n['WARN']} warnings, {n['INFO']} notes")
    if not n["FATAL"]:
        print("  No verifiable factual defect found. Warnings are for the author to judge.")
    return n["FATAL"]


if __name__ == "__main__":
    sys.exit(main())
