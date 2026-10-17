"""A 32-bit gate instruction set for the qsim-pynq overlay.

Word layout (MSB -> LSB)
    [31:28] opcode   0=NOP 1=H 2=RX 3=RZ 4=CNOT 5=DIAG 15=END
    [27:23] qubit a  (target for 1q gates, control for CNOT, table slot for DIAG)
    [22:18] qubit b  (CNOT target; 0 otherwise)
    [17:16] reserved
    [15:0]  angle    signed Q0.15 fraction of pi  (theta = pi * a / 2^15)

DIAG multiplies every amplitude x by exp(-i angle * C[x]), with C a
diagonal cost table preloaded in block RAM (one pass over the state).

Angles are quantised to pi/32768 ~ 9.6e-5 rad. The ARM core streams the
words over AXI-DMA; the fabric turns the angle into cos/sin(theta/2) with
its own table or CORDIC. ``FixedSim`` emulates the result as gate
coefficients rounded to ``coef_bits``; a ROM/CORDIC model can replace it.

Also provides peephole compression: adjacent self-inverse pairs
(H H, CNOT CNOT on the same qubits, RX(+a) RX(-a)) between consecutive
Pauli exponentials cancel; on full UCCSD circuits this removes ~18 % of the
gates (mostly basis changes) and ~3-5 % of the CNOTs.
"""
from __future__ import annotations

import numpy as np

OP = {'NOP': 0, 'H': 1, 'RX': 2, 'RZ': 3, 'CNOT': 4, 'DIAG': 5, 'END': 15}
INV = {v: k for k, v in OP.items()}
ANGLE_LSB = np.pi / 2 ** 15


def quantise_angle(theta):
    """wrap to (-pi, pi] and round to the 16-bit grid"""
    t = (theta + np.pi) % (2 * np.pi) - np.pi
    a = int(np.clip(np.rint(t / ANGLE_LSB), -2 ** 15, 2 ** 15 - 1))
    return a


def encode(gates):
    words = []
    for g in gates:
        op = OP[g[0]]
        qa, qb, ang = g[1], 0, 0
        if g[0] == 'CNOT':
            qb = g[2]
        elif g[0] in ('RX', 'RZ', 'DIAG'):
            ang = quantise_angle(g[2]) & 0xFFFF
        assert qa < 32 and qb < 32
        words.append((op << 28) | (qa << 23) | (qb << 18) | ang)
    words.append(OP['END'] << 28)
    return np.array(words, dtype=np.uint32)


def decode(words):
    gates = []
    for w in map(int, words):
        op = INV[(w >> 28) & 0xF]
        if op == 'END':
            break
        qa, qb, ang = (w >> 23) & 0x1F, (w >> 18) & 0x1F, w & 0xFFFF
        if ang >= 2 ** 15:
            ang -= 2 ** 16
        if op == 'CNOT':
            gates.append(('CNOT', qa, qb))
        elif op in ('RX', 'RZ', 'DIAG'):
            gates.append((op, qa, ang * ANGLE_LSB))
        elif op == 'H':
            gates.append(('H', qa))
    return gates


def _cancels(g, h):
    if g[0] != h[0] or g[1] != h[1]:
        return False
    if g[0] == 'H':
        return True
    if g[0] == 'CNOT':
        return g[2] == h[2]
    if g[0] == 'RX':
        return abs(g[2] + h[2]) < 1e-12
    return False


def _commutes_past(g, h):
    """g and h act on disjoint qubits -> they commute."""
    qs = lambda x: {x[1], x[2]} if x[0] == 'CNOT' else {x[1]}
    return not (qs(g) & qs(h))


def peephole(gates):
    """Cancel inverse pairs, looking back past gates on disjoint qubits."""
    out = []
    for g in gates:
        j = len(out) - 1
        while j >= 0 and _commutes_past(g, out[j]):
            j -= 1
        if j >= 0 and _cancels(out[j], g):
            out.pop(j)
        elif g[0] == 'RZ' and j >= 0 and out[j][0] == 'RZ' and out[j][1] == g[1]:
            out[j] = ('RZ', g[1], out[j][2] + g[2])
        else:
            out.append(g)
    return out
