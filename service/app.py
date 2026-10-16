"""chemoqbit module for the oncology research platform.

Small FastAPI service, meant to run in its own LXC container next to the
oncology app and be reverse-proxied under /chemoqbit/.

    GET  /api/cases               list of case studies with headline numbers
    GET  /api/cases/{tag}         full result JSON (geometries, energies, rates)
    GET  /api/cases/{tag}/circuit ISA words for the qsim-pynq overlay (hex)
    GET  /embed/{tag}             self-contained HTML panel (iframe-able)
    GET  /health

Results are produced offline by ``scripts/*_case.py`` into ``results/``;
the service only reads them, so it stays light enough for a small container.
Educational use only: not for clinical decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
TEMPLATE = (Path(__file__).parent / 'embed.html').read_text()

app = FastAPI(title='chemoqbit', version='0.1.0')

CASES = {
    'formamide': dict(title='Formamide → formamidic acid',
                      blurb='Minimal amide–iminol model of the 5-FU lactam–lactim shift.'),
    'fivefu': dict(title='5-Fluorouracil N3-H → O4 lactim',
                   blurb='Minor tautomer proposed (alongside ionised forms) in 5-FU mispairing.'),
}


def _load(tag):
    f = RESULTS / f'{tag}.json'
    if tag not in CASES or not f.exists():
        raise HTTPException(404, f'no results for {tag!r}')
    return json.loads(f.read_text())


@app.get('/health')
def health():
    return {'ok': True}


@app.get('/api/cases')
def cases():
    out = []
    for tag, meta in CASES.items():
        f = RESULTS / f'{tag}.json'
        row = dict(tag=tag, **meta, available=f.exists())
        if f.exists():
            s = json.loads(f.read_text())['summary']
            row['barrier_ccsd_t'] = s['CCSD(T)']['dE_barrier']
            best = max((k for k in s if k.startswith('ADAPT ') and 'fixed' not in k),
                       key=lambda k: int(k.split()[1][:-1]))
            row['best_vqe'] = best
            row['barrier_vqe'] = s[best]['dE_barrier']
            comp = 'CCSD(T)+static 12q (ADAPT)'
            if comp in s:
                row['barrier_composite'] = s[comp]['dE_barrier']
        out.append(row)
    return out


@app.get('/api/cases/{tag}')
def case(tag: str):
    return _load(tag)


@app.get('/api/cases/{tag}/circuit', response_class=PlainTextResponse)
def circuit(tag: str, geometry: str = 'ts', space: str = '6e6o', method: str = 'adapt'):
    """Re-compile the optimised circuit and return one 32-bit word per line."""
    import numpy as np
    from chemoqbit.circuit import Ansatz
    from chemoqbit.isa import encode, peephole
    d = _load(tag)
    sp_ = d[geometry]['spaces'].get(f'{space}_{method}')
    if sp_ is None or 'theta' not in sp_:
        raise HTTPException(404, 'circuit parameters not stored for this run')
    if method == 'uccsd':
        ans = Ansatz.uccsd(sp_['n_qubits'], sp_['nelec'])
    else:
        ans = Ansatz.from_excitations(sp_['n_qubits'], sp_['nelec'],
                                      [tuple(o) for o in sp_['operators']])
    words = encode(peephole(ans.gates(np.array(sp_['theta']))))
    return '\n'.join(f'{w:08x}' for w in words)


@app.get('/embed/{tag}', response_class=HTMLResponse)
def embed(tag: str):
    d = _load(tag)
    payload = dict(meta=CASES[tag], summary=d['summary'], imag_cm=d['imag_cm'],
                   T=d.get('T', 310.15),
                   spaces={g: d[g]['spaces'] for g in ('reactant', 'ts', 'product')})
    return TEMPLATE.replace('/*__DATA__*/null', json.dumps(payload, default=float))
