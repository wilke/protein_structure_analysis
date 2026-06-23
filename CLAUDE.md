# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based protein structure comparison pipeline for comparing predicted structures from AlphaFold, ESMFold, Chai, and Boltz, with pLDDT confidence score integration and PAE (Predicted Aligned Error) visualization.

## Runtime environment

For testing we use the python venv in `/Users/olson/structure-venv`

## Commands

### Install
```bash
pip install -e .
# or
pip install -r requirements.txt
```

### Run CLI
```bash
# Compare two structures
python -m protein_compare compare struct1.pdb struct2.pdb

# Batch comparison (all pairwise)
python -m protein_compare batch *.pdb -o results.csv

# Batch comparison against reference
python -m protein_compare batch *.pdb --reference ref.pdb -o results.csv

# Generate visualization (creates pre-aligned PDB for reliable PyMOL superposition)
python -m protein_compare visualize struct1.pdb struct2.pdb -o aligned.pml

# View structure info (auto-detects predicted vs experimental structures)
python -m protein_compare info structure.pdb

# Generate contact map
python -m protein_compare contacts structure.pdb -o contacts.png

# Generate comprehensive structure characterization report
python -m protein_compare characterize structure.pdb -o report

# With PAE data from AlphaFold (requires monomer_ptm or multimer model)
python -m protein_compare characterize alphafold.pdb --pae scores.json -o report
```

### Testing
```bash
pytest
pytest tests/test_specific.py::test_function  # single test
```

## Architecture

### Core Data Flow
1. **StructureLoader** (`io/parser.py`) parses PDB files into **ProteinStructure** dataclass containing Cα/Cβ coordinates, pLDDT scores (from B-factor column), and sequence
2. **StructuralAligner** (`core/alignment.py`) uses tmtools for TM-align, producing **AlignmentResult** with TM-scores, RMSD, transformation matrix, and residue mapping
3. **BatchComparator** (`core/batch.py`) orchestrates comparisons, optionally using joblib for parallel execution, and produces **PairwiseResult** objects
4. **ComparisonReporter** (`io/reporter.py`) generates CSV/JSON/HTML output

### Key Classes
- `ProteinStructure`: Immutable container for parsed structure with `ca_coords`, `plddt`, `sequence`
- `AlignmentResult`: Contains `tm_score`, `rmsd`, `rotation_matrix`, `translation_vector`, `residue_mapping`, `per_residue_distance`
- `PairwiseResult`: Full comparison result including `weighted_rmsd`, `ss_agreement`, `contact_jaccard`, `gdt_ts/gdt_ha`
- `PAEData`: Container for Predicted Aligned Error matrix with `pae_matrix`, `ptm`, `iptm`, and domain identification methods
- `PAEAnalysis`: Analysis results including detected domains, mean/median PAE, inter/intra-domain PAE

### Analysis Modules
- `core/metrics.py`: RMSD calculations, confidence-weighted RMSD, GDT-TS/GDT-HA scores
- `core/secondary.py`: DSSP-based secondary structure assignment and comparison
- `core/contacts.py`: Contact map generation (Cα distance < 8Å) and Jaccard similarity
- `core/confidence.py`: pLDDT-based confidence analysis

### Visualization
- `visualization/alignment_viz.py`: PyMOL script generation with pre-aligned PDB output, divergence plots
- `visualization/contact_maps.py`: Contact map heatmaps
- `visualization/divergence.py`: Divergence region highlighting
- `visualization/structure_report.py`: Comprehensive HTML/PDF reports with 3D viewer, PAE visualization

## Key Concepts

- **pLDDT scores**: Stored in PDB B-factor column (0-100 scale); high confidence ≥70, low confidence <50
- **B-factors**: For experimental structures, B-factor indicates atomic displacement/flexibility (lower = more ordered)
- **TM-score**: >0.5 indicates same fold, >0.4 similar fold
- **Confidence-weighted RMSD**: Weights each residue by min(pLDDT₁, pLDDT₂)/100
- **Divergence threshold**: Default 3Å for counting divergent residues
- **PAE (Predicted Aligned Error)**: AlphaFold's estimate of position error between residue pairs; low PAE (<5Å) indicates confident relative positioning
- **pTM (predicted TM-score)**: AlphaFold's confidence in overall fold prediction (0-1); >0.5 indicates confident prediction
- **ipTM (interface pTM)**: For multimers, confidence in protein-protein interface prediction

## AlphaFold Output Files

When using AlphaFold with `--model_preset=monomer_ptm` or `multimer`:
- `*_ranked_0.pdb`: Best ranked structure (pLDDT in B-factor column)
- `*_pae_model_*_ptm_pred_0.json`: PAE matrix for each model
- `*_confidence_model_*.json`: Per-residue pLDDT scores (also in PDB)
- `*_ranking_debug.json`: Model ranking information

Note: The default `monomer` preset does NOT generate PAE data. Use `monomer_ptm` for PAE output.

## Dependencies

Requires Python ≥3.12 (the codebase uses PEP 701 nested f-strings). Key packages: biopython, tmtools, numpy, scipy, pandas, matplotlib, click, joblib. External: DSSP binary (mkdssp) for secondary structure analysis.
