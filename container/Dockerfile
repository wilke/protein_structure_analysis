# Dockerfile for protein_compare — structure analysis and reporting
#
# Provides the protein_compare CLI for characterizing and comparing
# predicted protein structures. Designed to consume normalized output
# from predict-structure (model_1.pdb, confidence.json, etc.).
#
# Build:
#   docker build -t bvbrc/protein-compare:latest -f container/Dockerfile .
#
# Run:
#   docker run -v $(pwd)/output:/data bvbrc/protein-compare:latest \
#     characterize /data/model_1.pdb -o /data/report
#
# Batch comparison:
#   docker run -v $(pwd)/results:/data bvbrc/protein-compare:latest \
#     batch /data/*/model_1.pdb -o /data/comparison.csv

FROM python:3.12-slim

LABEL maintainer="BV-BRC Team"
LABEL description="Protein structure analysis and comparison tool"
LABEL org.opencontainers.image.source="https://github.com/BV-BRC/protein_structure_analysis"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies for matplotlib rendering and DSSP
RUN apt-get update && apt-get install -y --no-install-recommends \
        dssp \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Install protein_compare
WORKDIR /app
COPY setup.py README.md ./
COPY protein_compare/ protein_compare/

RUN pip install --no-cache-dir -e "." && \
    pip cache purge

# Verify installation
RUN protein_compare --help

WORKDIR /data

CMD ["python", "-m", "protein_compare", "--help"]
