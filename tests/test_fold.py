import itertools
import numpy as np

from chemoqbit.fold.lattice import (T, coords, analyse, cost_table, pair_energies,
                                    decode_turns, n_qubits, load_mj)


def test_bond_geometry():
    X = coords([1, 0, 2, 3, 1, 2])
    d2 = np.sum(np.diff(X, axis=0) ** 2, axis=1)
    assert np.all(d2 == 3)
    for i in range(1, len(X) - 1):
        a, b = X[i - 1] - X[i], X[i + 1] - X[i]
        assert a @ b == -1          # cos = -1/3  -> 109.47 deg


def test_self_avoiding_walk_counts_match_diamond_lattice():
    # OEIS A001394 (https://oeis.org/A001394): SAWs on the diamond lattice
    known = {1: 4, 2: 12, 3: 36, 4: 108, 5: 324, 6: 948, 7: 2796, 8: 8196}
    for n, c in known.items():
        count = 0
        for turns in itertools.product(range(4), repeat=n):
            if any(turns[i] == turns[i - 1] for i in range(1, n)):
                continue
            X = coords(list(turns))
            count += len({tuple(x) for x in X}) == len(X)
        assert count == c, (n, count, c)


def test_vectorised_table_matches_scalar():
    seq = 'GYDPETGTW'
    E, st = cost_table(seq)
    Ep = pair_energies(seq)
    rng = np.random.default_rng(0)
    for b in rng.integers(0, len(E), 400):
        s, e, _ = analyse(decode_turns(int(b), len(seq)), Ep)
        assert {'valid': 0, 'backtrack': 1, 'overlap': 2}[s] == st[b]
        if s == 'valid':
            assert abs(e - E[b]) < 1e-12


def test_valid_count_matches_saw_count_with_fixed_turns():
    # fixing the first two turns divides the SAW count by 4*3
    for N, saw in [(6, 324), (7, 948), (8, 2796), (9, 8196), (11, 70188)]:
        _, st = cost_table('A' * N)
        assert int(np.sum(st == 0)) * 12 == saw


def test_mj_symmetric_and_known_entry():
    aa, M = load_mj()
    assert np.allclose(M, M.T)
    assert abs(M[aa.index('L'), aa.index('L')] - (-7.37)) < 1e-9
