"""Protein Structure Comparison Pipeline.

A Python-based batch comparison tool for protein structures from
computational prediction tools (AlphaFold, ESMFold, Chai, Boltz),
with pLDDT confidence score integration.
"""

__version__ = "0.2.0"
__author__ = "Protein Compare"

from protein_compare.io.parser import StructureLoader, ProteinStructure
from protein_compare.core.alignment import StructuralAligner, AlignmentResult
from protein_compare.core.metrics import MetricsCalculator
from protein_compare.core.secondary import SecondaryStructureAnalyzer
from protein_compare.core.contacts import ContactMapAnalyzer
from protein_compare.core.batch import BatchComparator

__all__ = [
    "StructureLoader",
    "ProteinStructure",
    "StructuralAligner",
    "AlignmentResult",
    "MetricsCalculator",
    "SecondaryStructureAnalyzer",
    "ContactMapAnalyzer",
    "BatchComparator",
]
