# A reproducible environment for chemoqbit.
#
# The point of this file is that "it runs on my machine" is not a claim a reviewer
# can check. Every version below is the one the published results were produced
# with, read from pip freeze rather than guessed, so a reviewer gets the same
# numbers rather than approximately the same numbers.
#
#   docker build -t chemoqbit .
#   docker run --rm chemoqbit                      # runs the test suite
#   docker run --rm chemoqbit python scripts/fetch_structures.py
#
# tblite ships wheels with its Fortran libraries bundled, so GFN2-xTB needs no
# conda and no compiler here. It is the dependency most likely to break an
# install from source, which is why it is pinned exactly.

FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/quantumproteinsai/chemoqbit"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.description="Reaction energetics on qubits, from molecular integrals to an FPGA instruction stream"

# PySCF needs a BLAS at runtime; rdkit needs the usual X-less image libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libopenblas0 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# versions as published, from pip freeze on the machine that produced the results
RUN pip install --no-cache-dir \
        numpy==2.5.4 \
        scipy==1.18.1 \
        pyscf==2.14.0 \
        tblite==0.7.0 \
        geometric==1.1.1 \
        rdkit==2026.9.1 \
        fastapi==0.143.3 \
        uvicorn==0.54.0 \
        httpx==0.28.1 \
        pytest==9.1.1

COPY . /app
RUN pip install --no-cache-dir --no-deps -e .

# The crystal structures are NOT baked in: they are third-party data and belong to
# RCSB, so fetch them at run time instead of redistributing them in an image.
#   docker run --rm -v "$PWD/data:/app/data" chemoqbit python scripts/fetch_structures.py

# Two checks, because they test different things: pytest covers the code, and
# test_checks.py covers the physical bounds against cases whose verdict is known.
CMD ["sh", "-c", "pytest -q && python scripts/test_checks.py"]
