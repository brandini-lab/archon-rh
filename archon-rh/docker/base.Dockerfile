FROM python:3.11-slim@sha256:1dd1d8bd6b69e5258fa40a7c4fa2f2b5756fa0fe58f3d82405463e99e36fcff8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md /workspace/
RUN pip install --upgrade pip && pip install --no-cache-dir .[dev]
