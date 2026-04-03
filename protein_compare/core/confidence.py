"""Confidence score handling for predicted protein structures.

Handles pLDDT scores from AlphaFold, ESMFold, Chai, and Boltz,
providing utilities for confidence-based analysis and filtering.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from protein_compare.io.parser import ProteinStructure


# pLDDT confidence categories (AlphaFold conventions)
PLDDT_CATEGORIES = {
    "very_high": (90, 100),  # Very high confidence
    "confident": (70, 90),   # Confident
    "low": (50, 70),         # Low confidence
    "very_low": (0, 50),     # Very low confidence (often disordered)
}


@dataclass
class ConfidenceStats:
    """Statistics about structure confidence scores."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    n_very_high: int  # pLDDT >= 90
    n_confident: int  # 70 <= pLDDT < 90
    n_low: int  # 50 <= pLDDT < 70
    n_very_low: int  # pLDDT < 50
    frac_confident: float  # Fraction with pLDDT >= 70
    n_residues: int

    @property
    def is_reliable(self) -> bool:
        """Check if structure is generally reliable (>70% confident residues)."""
        return self.frac_confident >= 0.7

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dictionary."""
        return {
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
            "std": round(self.std, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "n_very_high": self.n_very_high,
            "n_confident": self.n_confident,
            "n_low": self.n_low,
            "n_very_low": self.n_very_low,
            "frac_confident": round(self.frac_confident, 4),
            "n_residues": self.n_residues,
            "is_reliable": self.is_reliable,
        }


@dataclass
class ConfidenceComparison:
    """Comparison of confidence between two structures."""

    stats1: ConfidenceStats
    stats2: ConfidenceStats
    correlation: float  # Correlation of pLDDT between aligned residues
    mean_diff: float  # Mean difference in pLDDT
    divergent_low_confidence: list[int]  # Residues divergent AND low confidence


class ConfidenceAnalyzer:
    """Analyze and utilize confidence scores for structure comparison."""

    def __init__(
        self,
        high_threshold: float = 70.0,
        low_threshold: float = 50.0,
    ):
        """Initialize the analyzer.

        Args:
            high_threshold: Threshold for high-confidence residues.
            low_threshold: Threshold for very low-confidence residues.
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def compute_stats(self, plddt: np.ndarray) -> ConfidenceStats:
        """Compute confidence statistics.

        Args:
            plddt: Array of pLDDT scores.

        Returns:
            ConfidenceStats object.
        """
        n = len(plddt)

        n_very_high = int(np.sum(plddt >= 90))
        n_confident = int(np.sum((plddt >= 70) & (plddt < 90)))
        n_low = int(np.sum((plddt >= 50) & (plddt < 70)))
        n_very_low = int(np.sum(plddt < 50))

        return ConfidenceStats(
            mean=float(np.mean(plddt)),
            median=float(np.median(plddt)),
            std=float(np.std(plddt)),
            min=float(np.min(plddt)),
            max=float(np.max(plddt)),
            n_very_high=n_very_high,
            n_confident=n_confident,
            n_low=n_low,
            n_very_low=n_very_low,
            frac_confident=(n_very_high + n_confident) / n if n > 0 else 0.0,
            n_residues=n,
        )

    def get_confidence_weights(
        self,
        plddt1: np.ndarray,
        plddt2: np.ndarray,
        method: Literal["min", "mean", "product"] = "min",
    ) -> np.ndarray:
        """Calculate confidence weights for paired residues.

        Args:
            plddt1: pLDDT scores for structure 1.
            plddt2: pLDDT scores for structure 2.
            method: How to combine scores:
                - "min": Use minimum of the two (conservative)
                - "mean": Use average
                - "product": Use product (penalizes low confidence more)

        Returns:
            Weight array (0-1 per residue).
        """
        p1 = plddt1 / 100.0
        p2 = plddt2 / 100.0

        if method == "min":
            return np.minimum(p1, p2)
        elif method == "mean":
            return (p1 + p2) / 2.0
        elif method == "product":
            return p1 * p2
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_high_confidence_mask(
        self,
        plddt: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Get mask for high-confidence residues.

        Args:
            plddt: Array of pLDDT scores.
            threshold: Optional threshold override.

        Returns:
            Boolean mask array.
        """
        thresh = threshold or self.high_threshold
        return plddt >= thresh

    def get_low_confidence_mask(
        self,
        plddt: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Get mask for low-confidence residues.

        Args:
            plddt: Array of pLDDT scores.
            threshold: Optional threshold override.

        Returns:
            Boolean mask array.
        """
        thresh = threshold or self.low_threshold
        return plddt < thresh

    def filter_by_confidence(
        self,
        coords: np.ndarray,
        plddt: np.ndarray,
        min_confidence: float = 70.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter coordinates to high-confidence residues only.

        Args:
            coords: Coordinate array, shape (n, 3).
            plddt: pLDDT scores, shape (n,).
            min_confidence: Minimum confidence threshold.

        Returns:
            Tuple of (filtered_coords, filtered_plddt, original_indices).
        """
        mask = plddt >= min_confidence
        indices = np.where(mask)[0]

        return coords[mask], plddt[mask], indices

    def compare_confidence(
        self,
        struct1: ProteinStructure,
        struct2: ProteinStructure,
        residue_mapping: Optional[list[tuple[int, int]]] = None,
        divergence_threshold: float = 3.0,
    ) -> ConfidenceComparison:
        """Compare confidence profiles between structures.

        Args:
            struct1: First protein structure.
            struct2: Second protein structure.
            residue_mapping: Optional residue correspondence.
            divergence_threshold: RMSD threshold for divergent residues.

        Returns:
            ConfidenceComparison object.
        """
        stats1 = self.compute_stats(struct1.plddt)
        stats2 = self.compute_stats(struct2.plddt)

        if residue_mapping:
            plddt1_aligned = struct1.plddt[[m[0] for m in residue_mapping]]
            plddt2_aligned = struct2.plddt[[m[1] for m in residue_mapping]]

            # Calculate per-residue distances
            coords1 = struct1.ca_coords[[m[0] for m in residue_mapping]]
            coords2 = struct2.ca_coords[[m[1] for m in residue_mapping]]
            distances = np.linalg.norm(coords1 - coords2, axis=1)
        else:
            min_len = min(len(struct1.plddt), len(struct2.plddt))
            plddt1_aligned = struct1.plddt[:min_len]
            plddt2_aligned = struct2.plddt[:min_len]
            distances = np.linalg.norm(
                struct1.ca_coords[:min_len] - struct2.ca_coords[:min_len],
                axis=1
            )

        # Correlation of confidence scores
        if len(plddt1_aligned) > 1:
            correlation = float(np.corrcoef(plddt1_aligned, plddt2_aligned)[0, 1])
        else:
            correlation = 0.0

        # Find divergent AND low-confidence residues
        divergent = distances > divergence_threshold
        low_conf = (plddt1_aligned < self.high_threshold) | (plddt2_aligned < self.high_threshold)
        divergent_low = list(np.where(divergent & low_conf)[0])

        return ConfidenceComparison(
            stats1=stats1,
            stats2=stats2,
            correlation=correlation,
            mean_diff=float(np.mean(np.abs(plddt1_aligned - plddt2_aligned))),
            divergent_low_confidence=divergent_low,
        )

    def categorize_residue(self, plddt_score: float) -> str:
        """Categorize a residue by its pLDDT score.

        Args:
            plddt_score: Single pLDDT value.

        Returns:
            Category name.
        """
        if plddt_score >= 90:
            return "very_high"
        elif plddt_score >= 70:
            return "confident"
        elif plddt_score >= 50:
            return "low"
        else:
            return "very_low"

    def get_disorder_prediction(
        self,
        plddt: np.ndarray,
        threshold: float = 50.0,
        min_length: int = 5,
    ) -> list[tuple[int, int]]:
        """Predict disordered regions based on low pLDDT.

        Low pLDDT regions often correspond to intrinsically
        disordered regions (IDRs).

        Args:
            plddt: Array of pLDDT scores.
            threshold: pLDDT threshold for disorder.
            min_length: Minimum length of disordered region.

        Returns:
            List of (start, end) tuples for disordered regions.
        """
        low_conf = plddt < threshold
        regions = []

        start = None
        for i, is_low in enumerate(low_conf):
            if is_low and start is None:
                start = i
            elif not is_low and start is not None:
                if i - start >= min_length:
                    regions.append((start, i))
                start = None

        # Handle region at end
        if start is not None and len(plddt) - start >= min_length:
            regions.append((start, len(plddt)))

        return regions

    def adjust_metrics_by_confidence(
        self,
        rmsd: float,
        tm_score: float,
        mean_plddt: float,
    ) -> dict:
        """Provide confidence-adjusted interpretation of metrics.

        Args:
            rmsd: Calculated RMSD.
            tm_score: Calculated TM-score.
            mean_plddt: Mean pLDDT of compared regions.

        Returns:
            Dict with adjusted metrics and reliability assessment.
        """
        # Confidence factor (0-1)
        conf_factor = min(mean_plddt / 100.0, 1.0)

        # Expected RMSD increase for low confidence
        # Low confidence structures have higher expected RMSD
        expected_noise = 2.0 * (1.0 - conf_factor)  # Up to 2Å noise

        # Adjusted RMSD (accounting for expected noise)
        adjusted_rmsd = max(0, rmsd - expected_noise)

        # Reliability of comparison
        if mean_plddt >= 70:
            reliability = "high"
        elif mean_plddt >= 50:
            reliability = "moderate"
        else:
            reliability = "low"

        return {
            "original_rmsd": rmsd,
            "adjusted_rmsd": adjusted_rmsd,
            "expected_noise": expected_noise,
            "tm_score": tm_score,
            "confidence_factor": conf_factor,
            "reliability": reliability,
            "interpretation": self._interpret_comparison(
                adjusted_rmsd, tm_score, reliability
            ),
        }

    @staticmethod
    def _interpret_comparison(
        rmsd: float,
        tm_score: float,
        reliability: str,
    ) -> str:
        """Generate human-readable interpretation.

        Args:
            rmsd: RMSD value.
            tm_score: TM-score value.
            reliability: Reliability level.

        Returns:
            Interpretation string.
        """
        if tm_score >= 0.5:
            fold_sim = "same fold"
        elif tm_score >= 0.4:
            fold_sim = "similar fold"
        else:
            fold_sim = "different folds"

        if rmsd < 1.0:
            struct_sim = "nearly identical"
        elif rmsd < 2.0:
            struct_sim = "very similar"
        elif rmsd < 4.0:
            struct_sim = "similar"
        else:
            struct_sim = "different"

        reliability_note = (
            "" if reliability == "high"
            else f" (comparison {reliability} reliability due to low confidence)"
        )

        return f"Structures are {struct_sim} ({fold_sim}){reliability_note}"
