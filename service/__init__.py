"""FastAPI service exposing the cases and their FPGA instruction streams.

This file exists because the package did not install `service` at all: the
pyproject `packages.find` include list named only `chemoqbit*`, so the module the
JOSS paper describes as part of the software was absent from any install. The
tests caught it only on CI, because `pytest` run bare does not put the working
directory on sys.path while `python -m pytest` does -- which is why it passed
locally and failed on the runner three times.
"""
