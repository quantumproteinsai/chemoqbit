"""The three files that describe a release have to agree with each other.

Both of the bugs this guards against were found in a published archive rather
than before it. CITATION.cff still read 1.3.0 inside the 1.4.0 tarball, so the
deposit's own citation metadata named a version that was two releases old. And
the keyword pass that made the SOD1 work findable edited .zenodo.json alone, so
the two lists drifted by six terms without anything noticing.

Keywords are how someone who read a paper finds the archive behind it, and the
version is how they know which one they have. Neither is exercised by any other
test, because nothing imports them.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _cff():
    return (ROOT / 'CITATION.cff').read_text()


def _zenodo():
    return json.loads((ROOT / '.zenodo.json').read_text())


def _cff_keywords():
    return re.findall(r'^  - (.+)$', _cff().split('keywords:')[1], re.M)


def test_version_agrees_across_the_three_files():
    pyproject = re.search(r'^version = "(.+?)"$', (ROOT / 'pyproject.toml').read_text(),
                          re.M).group(1)
    cff = re.search(r'^version: (.+)$', _cff(), re.M).group(1).strip()
    assert cff == pyproject, 'CITATION.cff says %s, pyproject says %s' % (cff, pyproject)
    assert _zenodo()['version'] == pyproject, (
        '.zenodo.json says %s, pyproject says %s' % (_zenodo()['version'], pyproject))


def test_keywords_agree_between_citation_and_zenodo():
    cff, zen = _cff_keywords(), _zenodo()['keywords']
    assert sorted(cff) == sorted(zen), (
        'only in CITATION.cff: %s; only in .zenodo.json: %s'
        % ([k for k in cff if k not in zen], [k for k in zen if k not in cff]))


def test_the_applications_are_findable():
    """A reader of one of the papers has to be able to search their way here."""
    kw = [k.lower() for k in _cff_keywords()]
    for term in ('sod1', 'tdp-43', 'amyotrophic lateral sclerosis', 'thymidylate synthase'):
        assert any(term in k for k in kw), 'no keyword covers %r' % term


def test_author_identity_agrees():
    cff = _cff()
    creator = _zenodo()['creators'][0]
    assert creator['orcid'] in cff
    assert creator['affiliation'] in cff
    family, given = creator['name'].split(', ')
    assert family in cff and given in cff
