import pytest
from pathlib import Path

fastapi = pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

RESULTS = Path(__file__).resolve().parents[1] / 'results'
pytestmark = pytest.mark.skipif(not (RESULTS / 'formamide.json').exists(),
                                reason='run scripts/formamide_case.py first')


def client():
    from service.app import app
    return TestClient(app)


def test_cases_and_embed():
    c = client()
    rows = c.get('/api/cases').json()
    fa = next(r for r in rows if r['tag'] == 'formamide')
    assert fa['available'] and 30 < fa['barrier_ccsd_t'] < 70
    html = c.get('/embed/formamide').text
    assert '/*__DATA__*/null' not in html and 'CCSD(T)' in html


def test_circuit_words_decode_and_reproduce_energy():
    import json, numpy as np
    from chemoqbit.isa import decode
    from chemoqbit.circuit import Ansatz, FloatSim
    c = client()
    txt = c.get('/api/cases/formamide/circuit', params=dict(space='4e4o')).text
    words = np.array([int(w, 16) for w in txt.split()], dtype=np.uint32)
    assert words[-1] >> 28 == 15
    d = json.loads((RESULTS / 'formamide.json').read_text())['ts']['spaces']['4e4o_adapt']
    ans = Ansatz.from_excitations(d['n_qubits'], d['nelec'], [tuple(o) for o in d['operators']])
    ref = ans.state(np.array(d['theta']))
    psi = FloatSim(d['n_qubits']).run(ans.hf_state(), decode(words))
    assert 1 - abs(np.vdot(ref, psi)) ** 2 < 1e-6
