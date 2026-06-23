# protein_compare

Batch comparison tool for predicted protein structures from AlphaFold, ESMFold, Chai, and Boltz with pLDDT confidence score integration.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Compare two structures
python -m protein_compare compare struct1.pdb struct2.pdb

# Batch comparison (all pairwise)
python -m protein_compare batch *.pdb -o results.csv

# Batch comparison against reference
python -m protein_compare batch *.pdb --reference ref.pdb -o results.csv

# Generate visualization
python -m protein_compare visualize struct1.pdb struct2.pdb -o aligned.pml

# View structure info
python -m protein_compare info structure.pdb

# Generate contact map
python -m protein_compare contacts structure.pdb -o contacts.png
```

## Requirements

- Python >= 3.12
- biopython, tmtools, numpy, scipy, pandas, matplotlib, click, joblib
- DSSP binary (mkdssp) for secondary structure analysis
