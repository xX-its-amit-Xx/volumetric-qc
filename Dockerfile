FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for scientific Python wheels and reading compressed NIfTI.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install package + extras for reports and dev tooling.
COPY pyproject.toml README.md ./
COPY LICENSE ./
COPY src ./src

RUN pip install --upgrade pip && \
    pip install .[dev]

# Bake in the demo data so `docker run` produces an immediately useful artifact.
COPY scripts ./scripts
COPY examples ./examples
COPY tests ./tests
COPY docs ./docs
COPY assets ./assets

VOLUME ["/data", "/output"]

ENTRYPOINT ["volumetric-qc"]
CMD ["--help"]
