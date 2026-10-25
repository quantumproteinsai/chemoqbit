# Contributing to chemoqbit

Thanks for your interest. This is a small project maintained by one person, so
please open an issue before starting substantial work — it avoids duplicated
effort and lets us agree on scope first.

## Reporting a problem

Open an issue with:

- what you ran (the exact command, or the Python snippet),
- what you expected and what happened,
- the output of `python -c "import chemoqbit, pyscf, numpy; print(chemoqbit.__file__, pyscf.__version__, numpy.__version__)"`,
- the `results/*.json` checkpoint if a long calculation produced a wrong number.

Numerical bugs are the ones worth reporting most. If a barrier, a frequency or a
circuit energy looks wrong, say what you believe the right answer is and why —
an expected value turns a bug report into a test case.

## Setting up

```sh
git clone <repository>
cd chemoqbit
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The test suite must pass before and after your change. It checks H2 and LiH
against full configuration interaction, analytic against numerical gradients,
the FPGA instruction-set round trip, Eckart transmission, the folding encoding
and QAOA, and the web service.

## Making a change

1. Branch from `main`.
2. Add a test that fails without your change. For numerical work, pin the
   expected value to an independent source — an analytic result, a different
   method, or a published measurement — rather than to the code's own output.
3. Keep the physical sanity checks intact. `chemoqbit/sanity.py` holds bounds
   with the reason each exists written into the file; `scripts/test_checks.py`
   exercises every bound in both directions. **If a check fires on a correct
   result, fix the check rather than widening the bound**, and record why in the
   same place. A bound widened to make a run pass is how a wrong number reaches a
   manuscript.
4. Run `pytest -q` and `python scripts/test_checks.py`.
5. Open a pull request describing what changed and what you verified.

## Style

Follow the surrounding code. Comments explain *why* a choice was made, especially
where it encodes something learned the hard way — several in this codebase record
a specific past failure, and those are worth more than a restatement of what the
line does.

## Scope

In scope: electronic structure and active-space construction, circuit compilation
and the fixed-point instruction stream, reaction energetics, and the enzyme and
folding applications.

Out of scope: general-purpose quantum simulation, gate-model hardware backends,
and anything claiming quantum advantage. The package is explicit that it offers
none.

## Code of conduct

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
