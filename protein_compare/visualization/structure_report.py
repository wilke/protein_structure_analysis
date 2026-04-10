"""Structure characterization and report generation.

Provides comprehensive analysis and visualization of a single protein structure,
generating HTML and PDF reports with embedded figures.
"""

import base64
import json as _json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from protein_compare.io.parser import ProteinStructure, PAEData, PAELoader, ChaiScores, ChaiScoresLoader, MSADepth, MSADepthLoader
from protein_compare.core.confidence import ConfidenceAnalyzer, ConfidenceStats
from protein_compare.core.contacts import ContactMapAnalyzer
from protein_compare.core.secondary import SecondaryStructureAnalyzer


# Amino acid properties for sequence analysis
AA_PROPERTIES = {
    "A": {"name": "Alanine", "mw": 89.1, "type": "hydrophobic"},
    "C": {"name": "Cysteine", "mw": 121.2, "type": "polar"},
    "D": {"name": "Aspartic acid", "mw": 133.1, "type": "negative"},
    "E": {"name": "Glutamic acid", "mw": 147.1, "type": "negative"},
    "F": {"name": "Phenylalanine", "mw": 165.2, "type": "hydrophobic"},
    "G": {"name": "Glycine", "mw": 75.1, "type": "special"},
    "H": {"name": "Histidine", "mw": 155.2, "type": "positive"},
    "I": {"name": "Isoleucine", "mw": 131.2, "type": "hydrophobic"},
    "K": {"name": "Lysine", "mw": 146.2, "type": "positive"},
    "L": {"name": "Leucine", "mw": 131.2, "type": "hydrophobic"},
    "M": {"name": "Methionine", "mw": 149.2, "type": "hydrophobic"},
    "N": {"name": "Asparagine", "mw": 132.1, "type": "polar"},
    "P": {"name": "Proline", "mw": 115.1, "type": "special"},
    "Q": {"name": "Glutamine", "mw": 146.2, "type": "polar"},
    "R": {"name": "Arginine", "mw": 174.2, "type": "positive"},
    "S": {"name": "Serine", "mw": 105.1, "type": "polar"},
    "T": {"name": "Threonine", "mw": 119.1, "type": "polar"},
    "V": {"name": "Valine", "mw": 117.1, "type": "hydrophobic"},
    "W": {"name": "Tryptophan", "mw": 204.2, "type": "hydrophobic"},
    "Y": {"name": "Tyrosine", "mw": 181.2, "type": "polar"},
    "X": {"name": "Unknown", "mw": 110.0, "type": "special"},
}

AA_TYPE_COLORS = {
    "hydrophobic": "#FFA500",
    "polar": "#32CD32",
    "positive": "#4169E1",
    "negative": "#DC143C",
    "special": "#9370DB",
}

PLDDT_COLORS = {
    "very_high": "#0053D6",
    "confident": "#65CBF3",
    "low": "#FFDB13",
    "very_low": "#FF7D45",
}

# B-factor colors for experimental structures (lower = more ordered)
BFACTOR_COLORS = {
    "very_low": "#0053D6",    # Very ordered (B < 20)
    "low": "#65CBF3",          # Ordered (20-40)
    "medium": "#FFDB13",       # Moderate (40-60)
    "high": "#FF7D45",         # Flexible (> 60)
}

SS_COLORS = {"H": "#E41A1C", "E": "#377EB8", "C": "#999999"}

# Glossary of protein structure terms
GLOSSARY = {
    "pLDDT": {
        "term": "pLDDT (predicted Local Distance Difference Test)",
        "definition": "A per-residue confidence score (0-100) from structure prediction tools like AlphaFold and ESMFold. Indicates how confident the model is about the predicted position of each residue. Scores ≥90 are very high confidence, 70-90 are confident, 50-70 are low confidence, and <50 are very low confidence.",
    },
    "B-factor": {
        "term": "B-factor (Temperature Factor)",
        "definition": "In experimental structures, represents atomic displacement/flexibility. In predicted structures (AlphaFold, ESMFold), the B-factor column stores pLDDT confidence scores instead.",
    },
    "Contact Map": {
        "term": "Contact Map",
        "definition": "A 2D matrix showing which residue pairs are in spatial proximity (typically Cα-Cα distance < 8Å). Contacts near the diagonal are sequential neighbors; off-diagonal contacts indicate 3D folding bringing distant residues together.",
    },
    "Contact Order": {
        "term": "Contact Order",
        "definition": "The sequence separation between contacting residues. Short-range (<6 residues apart) contacts are local; long-range (>12 residues) contacts indicate complex folding topology.",
    },
    "Contact Density": {
        "term": "Contact Density",
        "definition": "The fraction of all possible residue pairs that are in contact. Higher density indicates a more compact, well-folded structure.",
    },
    "Secondary Structure": {
        "term": "Secondary Structure",
        "definition": "Local 3D arrangements of the protein backbone. The three main types are α-helix (H), β-sheet/strand (E), and coil/loop (C).",
    },
    "Alpha Helix": {
        "term": "α-Helix (H)",
        "definition": "A right-handed spiral structure stabilized by hydrogen bonds between backbone atoms (i to i+4). Common in many proteins, providing structural stability.",
    },
    "Beta Sheet": {
        "term": "β-Sheet/Strand (E)",
        "definition": "Extended chain conformations that form sheets through hydrogen bonds between adjacent strands. Can be parallel or antiparallel.",
    },
    "Coil": {
        "term": "Coil/Loop (C)",
        "definition": "Regions without regular secondary structure. Often flexible and found connecting helices and sheets. Includes turns, bends, and disordered regions.",
    },
    "DSSP": {
        "term": "DSSP (Define Secondary Structure of Proteins)",
        "definition": "Standard algorithm for assigning secondary structure from 3D coordinates based on hydrogen bonding patterns. Uses 8 states that are often simplified to 3 (H, E, C).",
    },
    "Cα (C-alpha)": {
        "term": "Cα (Alpha Carbon)",
        "definition": "The central carbon atom in each amino acid, bonded to the amino group, carboxyl group, hydrogen, and side chain. Cα positions define the protein backbone trace.",
    },
    "Cβ (C-beta)": {
        "term": "Cβ (Beta Carbon)",
        "definition": "The first carbon of the amino acid side chain, attached to Cα. All amino acids except glycine have a Cβ atom. Used in some structural analyses.",
    },
    "Residue": {
        "term": "Residue",
        "definition": "A single amino acid unit within a protein chain. Each residue has a backbone (N-Cα-C) and a side chain (R group) that determines its properties.",
    },
    "Molecular Weight": {
        "term": "Molecular Weight (MW)",
        "definition": "The total mass of the protein in Daltons (Da) or kiloDaltons (kDa). Calculated from the sum of amino acid masses minus water lost in peptide bond formation.",
    },
    "Hydrophobic": {
        "term": "Hydrophobic Residues",
        "definition": "Amino acids with non-polar side chains (A, V, L, I, M, F, W) that avoid water. Typically found in the protein core, driving protein folding.",
    },
    "Polar": {
        "term": "Polar Residues",
        "definition": "Amino acids with uncharged but polar side chains (S, T, N, Q, Y, C) that can form hydrogen bonds. Often found on protein surfaces.",
    },
    "Charged": {
        "term": "Charged Residues",
        "definition": "Amino acids with ionizable side chains. Positive: K, R, H (basic). Negative: D, E (acidic). Important for protein solubility and interactions.",
    },
    "TM-score": {
        "term": "TM-score",
        "definition": "Template Modeling score (0-1) measuring structural similarity between proteins, normalized by protein length. TM-score >0.5 indicates the same fold; >0.17 is typically significant.",
    },
    "RMSD": {
        "term": "RMSD (Root Mean Square Deviation)",
        "definition": "A measure of average distance between aligned atoms (usually Cα) in Ångströms. Lower RMSD means more similar structures. Depends on alignment length.",
    },
    "PAE": {
        "term": "PAE (Predicted Aligned Error)",
        "definition": "AlphaFold's estimate of position error (in Å) between residue pairs. Low PAE (<5Å) indicates high confidence in relative positions. High inter-domain PAE suggests flexible linkers or uncertain domain orientation.",
    },
    "pTM": {
        "term": "pTM (Predicted TM-score)",
        "definition": "AlphaFold's predicted TM-score (0-1) for the model. Values >0.5 indicate confident fold prediction. Higher is better.",
    },
    "ipTM": {
        "term": "ipTM (Interface pTM)",
        "definition": "For multimer predictions, measures confidence in protein-protein interfaces. Values >0.8 suggest reliable interface prediction.",
    },
    "Domain": {
        "term": "Structural Domain",
        "definition": "A compact, semi-independent folding unit within a protein. Domains often have low internal PAE but high PAE to other domains, indicating confident internal structure but uncertain relative orientation.",
    },
    "Ångström": {
        "term": "Ångström (Å)",
        "definition": "Unit of length equal to 10⁻¹⁰ meters (0.1 nanometers). Standard unit for atomic distances. A typical C-C bond is ~1.5Å; contact distance cutoff is typically 8Å.",
    },
}


@dataclass
class SequenceComposition:
    """Sequence composition analysis results."""
    length: int
    aa_counts: dict
    aa_fractions: dict
    molecular_weight: float
    type_counts: dict
    type_fractions: dict
    chains: list

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dictionary."""
        return {
            "length": self.length,
            "aa_counts": self.aa_counts,
            "aa_fractions": {k: round(v, 4) for k, v in self.aa_fractions.items()},
            "molecular_weight": round(self.molecular_weight, 1),
            "type_counts": self.type_counts,
            "type_fractions": {k: round(v, 4) for k, v in self.type_fractions.items()},
            "chains": self.chains,
        }


@dataclass
class ContactAnalysis:
    """Contact map analysis results."""
    contact_map: np.ndarray
    n_contacts: int
    contact_density: float
    n_short_range: int
    n_medium_range: int
    n_long_range: int
    n_very_long_range: int
    contacts_per_residue: np.ndarray

    def to_dict(self, include_matrices: bool = False) -> dict:
        """Serialize to JSON-safe dictionary.

        Args:
            include_matrices: If True, include the full contact map and
                per-residue arrays. These can be large for big structures.
        """
        d = {
            "n_contacts": self.n_contacts,
            "contact_density": round(self.contact_density, 4),
            "n_short_range": self.n_short_range,
            "n_medium_range": self.n_medium_range,
            "n_long_range": self.n_long_range,
            "n_very_long_range": self.n_very_long_range,
        }
        if include_matrices:
            d["contacts_per_residue"] = self.contacts_per_residue.tolist()
        return d


@dataclass
class SSAnalysis:
    """Secondary structure analysis results."""
    ss_sequence: list
    helix_fraction: float
    sheet_fraction: float
    coil_fraction: float
    helix_count: int
    sheet_count: int
    coil_count: int

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dictionary."""
        return {
            "ss_sequence": "".join(self.ss_sequence),
            "helix_fraction": round(self.helix_fraction, 4),
            "sheet_fraction": round(self.sheet_fraction, 4),
            "coil_fraction": round(self.coil_fraction, 4),
            "helix_count": self.helix_count,
            "sheet_count": self.sheet_count,
            "coil_count": self.coil_count,
        }


@dataclass
class PAEAnalysis:
    """PAE analysis results."""
    pae_data: PAEData
    mean_pae: float
    median_pae: float
    domains: list[list[int]]
    n_domains: int
    inter_domain_pae: Optional[float]  # Mean PAE between domains
    intra_domain_pae: float  # Mean PAE within domains

    def to_dict(self, include_matrix: bool = False) -> dict:
        """Serialize to JSON-safe dictionary.

        Args:
            include_matrix: If True, include the full NxN PAE matrix.
        """
        d: dict = {
            "mean_pae": round(self.mean_pae, 2),
            "median_pae": round(self.median_pae, 2),
            "n_domains": self.n_domains,
            "domains": self.domains,
            "intra_domain_pae": round(self.intra_domain_pae, 2),
        }
        if self.inter_domain_pae is not None:
            d["inter_domain_pae"] = round(self.inter_domain_pae, 2)
        if self.pae_data.ptm is not None:
            d["ptm"] = round(self.pae_data.ptm, 4)
        if self.pae_data.iptm is not None:
            d["iptm"] = round(self.pae_data.iptm, 4)
        if include_matrix:
            d["pae_matrix"] = self.pae_data.pae_matrix.tolist()
        return d


class StructureCharacterizer:
    """Generate comprehensive characterization of a single protein structure."""

    def __init__(
        self,
        structure: ProteinStructure,
        contact_cutoff: float = 8.0,
        dpi: int = 150,
        structure_type: Optional[str] = None,
        pae_data: Optional[PAEData] = None,
        pae_path: Optional[str] = None,
        chai_scores: Optional[ChaiScores] = None,
        chai_scores_path: Optional[str] = None,
        msa_depth: Optional[MSADepth] = None,
        msa_path: Optional[str] = None,
    ):
        """Initialize the characterizer.

        Args:
            structure: ProteinStructure to characterize.
            contact_cutoff: Distance cutoff for contacts in Ångströms.
            dpi: Resolution for generated figures.
            structure_type: "predicted" or "experimental". If None, auto-detects.
            pae_data: Optional PAEData object for AlphaFold PAE visualization.
            pae_path: Optional path to PAE JSON file. If provided, will load PAE data.
            chai_scores: Optional ChaiScores object for Chai confidence metrics.
            chai_scores_path: Optional path to Chai scores NPZ file.
            msa_depth: Optional MSADepth object for MSA depth visualization.
            msa_path: Optional path to MSA parquet file.
        """
        self.structure = structure
        self.contact_cutoff = contact_cutoff
        self.dpi = dpi

        # Auto-detect or use provided structure type
        if structure_type is None:
            from protein_compare.io.parser import StructureLoader
            self.structure_type = StructureLoader.detect_structure_type(structure)
        else:
            self.structure_type = structure_type

        self.is_predicted = self.structure_type == "predicted"

        # Handle PAE data
        if pae_data is not None:
            self.pae_data = pae_data
        elif pae_path is not None:
            self.pae_data = PAELoader.load(pae_path)
        elif self.is_predicted and structure.source_path:
            # Try to auto-find PAE file
            pae_file = PAELoader.find_pae_file(structure.source_path)
            if pae_file:
                self.pae_data = PAELoader.load(pae_file)
            else:
                self.pae_data = None
        else:
            self.pae_data = None

        # Handle Chai scores
        if chai_scores is not None:
            self.chai_scores = chai_scores
        elif chai_scores_path is not None:
            self.chai_scores = ChaiScoresLoader().load(chai_scores_path)
        elif self.is_predicted and structure.source_path:
            # Try to auto-find Chai scores file
            chai_scores_file = self._find_chai_scores_file(structure.source_path)
            if chai_scores_file:
                self.chai_scores = ChaiScoresLoader().load(chai_scores_file)
            else:
                self.chai_scores = None
        else:
            self.chai_scores = None

        # Handle MSA depth data (optional, requires pyarrow)
        if msa_depth is not None:
            self.msa_depth = msa_depth
        elif msa_path is not None and MSADepthLoader.is_available():
            try:
                self.msa_depth = MSADepthLoader().load(msa_path)
            except Exception:
                self.msa_depth = None
        elif self.is_predicted and structure.source_path and MSADepthLoader.is_available():
            # Try to auto-find MSA file
            msa_file = MSADepthLoader().find_msa_file(structure.source_path)
            if msa_file:
                try:
                    self.msa_depth = MSADepthLoader().load(msa_file)
                except Exception:
                    self.msa_depth = None
            else:
                self.msa_depth = None
        else:
            self.msa_depth = None

        self.confidence_analyzer = ConfidenceAnalyzer()
        self.contact_analyzer = ContactMapAnalyzer(cutoff=contact_cutoff)
        self.ss_analyzer = SecondaryStructureAnalyzer()
        self._seq_comp: Optional[SequenceComposition] = None
        self._conf_stats: Optional[ConfidenceStats] = None
        self._contact_analysis: Optional[ContactAnalysis] = None
        self._ss_analysis: Optional[SSAnalysis] = None
        self._pae_analysis: Optional[PAEAnalysis] = None

    def analyze_sequence_composition(self) -> SequenceComposition:
        """Analyze sequence composition (amino acids or nucleotides)."""
        if self._seq_comp is not None:
            return self._seq_comp
        seq = self.structure.sequence
        length = len(seq)
        aa_counts = dict(Counter(seq))
        aa_fractions = {aa: c / length for aa, c in aa_counts.items()}
        chains = list(set(rid[0] for rid in self.structure.residue_ids))

        if self.structure.is_nucleic_acid:
            # Nucleic acid — skip molecular weight and residue type classification
            mw = 0.0
            type_counts = {"purine": 0, "pyrimidine": 0}
            purines = set("AG")
            for nt in seq:
                if nt in purines:
                    type_counts["purine"] += 1
                else:
                    type_counts["pyrimidine"] += 1
            type_fractions = {t: c / length for t, c in type_counts.items()}
        else:
            mw = sum(AA_PROPERTIES.get(aa, AA_PROPERTIES["X"])["mw"] for aa in seq) - 18.015 * (length - 1)
            type_counts = {"hydrophobic": 0, "polar": 0, "positive": 0, "negative": 0, "special": 0}
            for aa in seq:
                t = AA_PROPERTIES.get(aa, AA_PROPERTIES["X"])["type"]
                type_counts[t] = type_counts.get(t, 0) + 1
            type_fractions = {t: c / length for t, c in type_counts.items()}

        self._seq_comp = SequenceComposition(length, aa_counts, aa_fractions, mw, type_counts, type_fractions, chains)
        return self._seq_comp

    def analyze_confidence(self) -> ConfidenceStats:
        """Analyze pLDDT confidence scores."""
        if self._conf_stats is not None:
            return self._conf_stats
        self._conf_stats = self.confidence_analyzer.compute_stats(self.structure.plddt)
        return self._conf_stats

    def analyze_contacts(self) -> ContactAnalysis:
        """Analyze contact map."""
        if self._contact_analysis is not None:
            return self._contact_analysis
        cmap = self.contact_analyzer.compute_contact_map(self.structure)
        n = len(cmap)
        n_contacts = int(np.sum(np.triu(cmap, k=1)))
        max_c = n * (n - 1) // 2
        density = n_contacts / max_c if max_c > 0 else 0
        n_short = n_medium = n_long = n_vlong = 0
        for i in range(n):
            for j in range(i + 1, n):
                if cmap[i, j]:
                    sep = j - i
                    if sep < 6: n_short += 1
                    elif sep < 12: n_medium += 1
                    elif sep < 24: n_long += 1
                    else: n_vlong += 1
        cpr = np.sum(cmap, axis=1)
        self._contact_analysis = ContactAnalysis(cmap, n_contacts, density, n_short, n_medium, n_long, n_vlong, cpr)
        return self._contact_analysis

    def analyze_secondary_structure(self) -> SSAnalysis:
        """Analyze secondary structure (protein only; NA returns all coil)."""
        if self._ss_analysis is not None:
            return self._ss_analysis
        n = self.structure.n_residues
        if self.structure.is_nucleic_acid:
            # DSSP does not assign secondary structure for nucleic acids
            ss_seq = ["C"] * n
        else:
            try:
                ss_seq = self.ss_analyzer.assign_ss(self.structure, simplify=True)
            except Exception:
                ss_seq = ["C"] * n
        h, e, c = ss_seq.count("H"), ss_seq.count("E"), ss_seq.count("C")
        self._ss_analysis = SSAnalysis(ss_seq, h/n if n else 0, e/n if n else 0, c/n if n else 0, h, e, c)
        return self._ss_analysis

    def analyze_pae(self) -> Optional[PAEAnalysis]:
        """Analyze PAE data if available.

        Returns:
            PAEAnalysis object or None if no PAE data.
        """
        if self._pae_analysis is not None:
            return self._pae_analysis
        if self.pae_data is None:
            return None

        pae = self.pae_data
        domains = pae.identify_domains(pae_cutoff=5.0, min_domain_size=15)

        # Calculate intra-domain PAE (mean PAE within domains)
        intra_pae_values = []
        for domain in domains:
            if len(domain) > 1:
                for i, idx1 in enumerate(domain):
                    for idx2 in domain[i+1:]:
                        intra_pae_values.append(pae.pae_matrix[idx1, idx2])
        intra_domain_pae = np.mean(intra_pae_values) if intra_pae_values else 0.0

        # Calculate inter-domain PAE (mean PAE between domains)
        inter_domain_pae = None
        if len(domains) > 1:
            inter_pae_values = []
            for i, dom1 in enumerate(domains):
                for dom2 in domains[i+1:]:
                    inter_pae_values.append(pae.get_domain_pae(dom1, dom2))
            inter_domain_pae = np.mean(inter_pae_values) if inter_pae_values else None

        self._pae_analysis = PAEAnalysis(
            pae_data=pae,
            mean_pae=pae.mean_pae,
            median_pae=pae.median_pae,
            domains=domains,
            n_domains=len(domains),
            inter_domain_pae=inter_domain_pae,
            intra_domain_pae=intra_domain_pae,
        )
        return self._pae_analysis

    @property
    def has_pae(self) -> bool:
        """Check if PAE data is available."""
        return self.pae_data is not None

    @property
    def has_chai_scores(self) -> bool:
        """Check if Chai scores are available."""
        return self.chai_scores is not None

    @property
    def has_msa_depth(self) -> bool:
        """Check if MSA depth data is available."""
        return self.msa_depth is not None

    def _find_chai_scores_file(self, structure_path: Path) -> Optional[Path]:
        """Try to find Chai scores file in same directory as structure.

        Looks for scores.model_idx_*.npz files matching the structure name.
        """
        parent = structure_path.parent
        stem = structure_path.stem

        # Extract model index from structure name (e.g., pred.model_idx_0)
        if "model_idx_" in stem:
            # Try exact match first
            scores_file = parent / f"scores.{stem.split('pred.')[-1]}.npz"
            if scores_file.exists():
                return scores_file

            # Try pattern match
            import re
            match = re.search(r'model_idx_(\d+)', stem)
            if match:
                idx = match.group(1)
                scores_file = parent / f"scores.model_idx_{idx}.npz"
                if scores_file.exists():
                    return scores_file

        # Look for any scores file in directory
        scores_files = list(parent.glob("scores.model_idx_*.npz"))
        if len(scores_files) == 1:
            return scores_files[0]

        return None

    def _get_bfactor_color(self, val: float) -> str:
        """Get color for B-factor value (experimental structures)."""
        if val < 20: return BFACTOR_COLORS["very_low"]
        if val < 40: return BFACTOR_COLORS["low"]
        if val < 60: return BFACTOR_COLORS["medium"]
        return BFACTOR_COLORS["high"]

    def _get_plddt_color(self, val: float) -> str:
        """Get color for pLDDT value (predicted structures)."""
        if val >= 90: return PLDDT_COLORS["very_high"]
        if val >= 70: return PLDDT_COLORS["confident"]
        if val >= 50: return PLDDT_COLORS["low"]
        return PLDDT_COLORS["very_low"]

    def _get_value_color(self, val: float) -> str:
        """Get color based on structure type."""
        if self.is_predicted:
            return self._get_plddt_color(val)
        else:
            return self._get_bfactor_color(val)

    def plot_plddt_distribution(self) -> Figure:
        """Plot pLDDT/B-factor score distribution histogram."""
        fig, ax = plt.subplots(figsize=(8, 5))
        values = self.structure.plddt
        stats = self.analyze_confidence()

        if self.is_predicted:
            # pLDDT mode (predicted structures)
            bins = np.arange(0, 105, 5)
            n, _, patches = ax.hist(values, bins=bins, edgecolor="white", linewidth=0.5)
            for i, patch in enumerate(patches):
                bc = (bins[i] + bins[i + 1]) / 2
                patch.set_facecolor(self._get_plddt_color(bc))
            for thresh in [50, 70, 90]:
                ax.axvline(thresh, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            txt = f"Mean: {stats.mean:.1f}\nMedian: {stats.median:.1f}\nHigh conf: {stats.frac_confident:.1%}"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=10, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.set_xlabel("pLDDT Score")
            ax.set_title("pLDDT Score Distribution")
            ax.set_xlim(0, 100)
            legend_patches = [
                mpatches.Patch(color=PLDDT_COLORS["very_high"], label="Very High (≥90)"),
                mpatches.Patch(color=PLDDT_COLORS["confident"], label="Confident (70-90)"),
                mpatches.Patch(color=PLDDT_COLORS["low"], label="Low (50-70)"),
                mpatches.Patch(color=PLDDT_COLORS["very_low"], label="Very Low (<50)"),
            ]
        else:
            # B-factor mode (experimental structures)
            max_val = max(100, np.max(values) * 1.1)
            bins = np.arange(0, max_val + 5, 5)
            n, _, patches = ax.hist(values, bins=bins, edgecolor="white", linewidth=0.5)
            for i, patch in enumerate(patches):
                bc = (bins[i] + bins[i + 1]) / 2
                patch.set_facecolor(self._get_bfactor_color(bc))
            for thresh in [20, 40, 60]:
                ax.axvline(thresh, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            low_mobility = np.sum(values < 30) / len(values)
            txt = f"Mean: {stats.mean:.1f} Ų\nMedian: {stats.median:.1f} Ų\nOrdered (<30): {low_mobility:.1%}"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=10, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.set_xlabel("B-factor (Ų)")
            ax.set_title("B-factor Distribution")
            ax.set_xlim(0, max_val)
            legend_patches = [
                mpatches.Patch(color=BFACTOR_COLORS["very_low"], label="Very Ordered (<20)"),
                mpatches.Patch(color=BFACTOR_COLORS["low"], label="Ordered (20-40)"),
                mpatches.Patch(color=BFACTOR_COLORS["medium"], label="Moderate (40-60)"),
                mpatches.Patch(color=BFACTOR_COLORS["high"], label="Flexible (>60)"),
            ]

        ax.set_ylabel("Number of Residues")
        ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
        fig.tight_layout()
        return fig

    def plot_plddt_profile(self) -> Figure:
        """Plot per-residue pLDDT/B-factor profile."""
        fig, ax = plt.subplots(figsize=(12, 4))
        values = self.structure.plddt
        res = np.arange(1, len(values) + 1)

        if self.is_predicted:
            # pLDDT mode
            ax.axhspan(0, 50, alpha=0.1, color=PLDDT_COLORS["very_low"])
            ax.axhspan(50, 70, alpha=0.1, color=PLDDT_COLORS["low"])
            ax.axhspan(70, 90, alpha=0.1, color=PLDDT_COLORS["confident"])
            ax.axhspan(90, 100, alpha=0.1, color=PLDDT_COLORS["very_high"])
            colors = [self._get_plddt_color(p) for p in values]
            ax.scatter(res, values, c=colors, s=10, zorder=3)
            ax.plot(res, values, color="gray", linewidth=0.5, alpha=0.5)
            for thresh in [50, 70, 90]:
                ax.axhline(thresh, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_ylabel("pLDDT Score")
            ax.set_title("Per-Residue pLDDT Profile")
            ax.set_ylim(0, 100)
        else:
            # B-factor mode
            max_val = max(80, np.max(values) * 1.1)
            ax.axhspan(0, 20, alpha=0.1, color=BFACTOR_COLORS["very_low"])
            ax.axhspan(20, 40, alpha=0.1, color=BFACTOR_COLORS["low"])
            ax.axhspan(40, 60, alpha=0.1, color=BFACTOR_COLORS["medium"])
            ax.axhspan(60, max_val, alpha=0.1, color=BFACTOR_COLORS["high"])
            colors = [self._get_bfactor_color(b) for b in values]
            ax.scatter(res, values, c=colors, s=10, zorder=3)
            ax.plot(res, values, color="gray", linewidth=0.5, alpha=0.5)
            for thresh in [20, 40, 60]:
                ax.axhline(thresh, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_ylabel("B-factor (Ų)")
            ax.set_title("Per-Residue B-factor Profile")
            ax.set_ylim(0, max_val)

        ax.set_xlabel("Residue Number")
        ax.set_xlim(1, len(values))
        fig.tight_layout()
        return fig

    def plot_contact_map(self) -> Figure:
        """Plot contact map heatmap."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ca = self.analyze_contacts()
        im = ax.imshow(ca.contact_map, cmap="Blues", origin="lower", aspect="equal")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Contact")
        n = len(ca.contact_map)
        ax.plot([0, n-1], [0, n-1], "k--", linewidth=0.5, alpha=0.5)
        txt = f"Contacts: {ca.n_contacts}\nDensity: {ca.contact_density:.3f}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("Residue Number"); ax.set_ylabel("Residue Number")
        ax.set_title(f"Contact Map (cutoff: {self.contact_cutoff} Å)")
        fig.tight_layout()
        return fig

    def plot_contact_order(self) -> Figure:
        """Plot contact order distribution."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ca = self.analyze_contacts()
        cats = ["Short\n(<6)", "Medium\n(6-12)", "Long\n(12-24)", "Very Long\n(>24)"]
        counts = [ca.n_short_range, ca.n_medium_range, ca.n_long_range, ca.n_very_long_range]
        colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
        bars = ax.bar(cats, counts, color=colors, edgecolor="white")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(cnt), ha="center", fontsize=11)
        ax.set_xlabel("Sequence Separation"); ax.set_ylabel("Number of Contacts")
        ax.set_title("Contact Order Distribution")
        ax.text(0.98, 0.98, f"Total: {ca.n_contacts}", transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        fig.tight_layout()
        return fig

    def plot_residue_contacts(self) -> Figure:
        """Plot contacts per residue."""
        fig, ax = plt.subplots(figsize=(12, 4))
        ca = self.analyze_contacts()
        res = np.arange(1, len(ca.contacts_per_residue) + 1)
        ax.bar(res, ca.contacts_per_residue, color="#377eb8", width=1.0)
        mean_c = np.mean(ca.contacts_per_residue)
        ax.axhline(mean_c, color="red", linestyle="--", label=f"Mean: {mean_c:.1f}")
        ax.set_xlabel("Residue Number"); ax.set_ylabel("Number of Contacts")
        ax.set_title("Contacts Per Residue"); ax.set_xlim(1, len(ca.contacts_per_residue))
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig

    def plot_aa_composition(self) -> Figure:
        """Plot sequence composition (amino acids or nucleotides)."""
        fig, ax = plt.subplots(figsize=(12, 5))
        comp = self.analyze_sequence_composition()

        if self.structure.is_nucleic_acid:
            nt_order = ["A", "C", "G", "T", "U"]
            nt_colors = {"A": "#e41a1c", "C": "#377eb8", "G": "#4daf4a", "T": "#984ea3", "U": "#ff7f00"}
            present = [nt for nt in nt_order if comp.aa_fractions.get(nt, 0) > 0]
            fracs = [comp.aa_fractions.get(nt, 0) * 100 for nt in present]
            colors = [nt_colors.get(nt, "#999999") for nt in present]
            ax.bar(present, fracs, color=colors, edgecolor="white", linewidth=0.5)
            expected = 100 / len(present) if present else 25
            ax.axhline(expected, color="gray", linestyle="--", alpha=0.7, label=f"Expected ({expected:.0f}%)")
            ax.set_xlabel("Nucleotide"); ax.set_ylabel("Frequency (%)")
            ax.set_title("Nucleotide Composition")
        else:
            aa_order = ["A","V","L","I","M","F","W","S","T","N","Q","Y","C","K","R","H","D","E","G","P"]
            fracs = [comp.aa_fractions.get(aa, 0) * 100 for aa in aa_order]
            colors = [AA_TYPE_COLORS[AA_PROPERTIES[aa]["type"]] for aa in aa_order]
            ax.bar(aa_order, fracs, color=colors, edgecolor="white", linewidth=0.5)
            ax.axhline(5, color="gray", linestyle="--", alpha=0.7, label="Expected (5%)")
            ax.set_xlabel("Amino Acid"); ax.set_ylabel("Frequency (%)")
            ax.set_title("Amino Acid Composition")
            legend_patches = [mpatches.Patch(color=c, label=t.title()) for t, c in AA_TYPE_COLORS.items()]
            ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        return fig

    def plot_ss_composition(self) -> Figure:
        """Plot secondary structure composition."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ss = self.analyze_secondary_structure()
        sizes = [ss.helix_fraction, ss.sheet_fraction, ss.coil_fraction]
        labels, colors = ["Helix", "Sheet", "Coil"], [SS_COLORS["H"], SS_COLORS["E"], SS_COLORS["C"]]
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if non_zero:
            sz, lb, cl = zip(*non_zero)
            ax1.pie(sz, labels=lb, colors=cl, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Secondary Structure Content")
        counts = [ss.helix_count, ss.sheet_count, ss.coil_count]
        bars = ax2.bar(labels, counts, color=[SS_COLORS["H"], SS_COLORS["E"], SS_COLORS["C"]])
        for bar, cnt in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(cnt), ha="center")
        ax2.set_xlabel("Secondary Structure"); ax2.set_ylabel("Residues")
        ax2.set_title("Residue Counts by SS Type")
        fig.tight_layout()
        return fig

    def plot_ss_profile(self) -> Figure:
        """Plot secondary structure along sequence."""
        fig, ax = plt.subplots(figsize=(12, 2))
        ss = self.analyze_secondary_structure()
        for i, s in enumerate(ss.ss_sequence):
            ax.bar(i + 1, 1, color=SS_COLORS.get(s, SS_COLORS["C"]), width=1.0, edgecolor="none")
        ax.set_xlim(0.5, len(ss.ss_sequence) + 0.5); ax.set_ylim(0, 1)
        ax.set_xlabel("Residue Number"); ax.set_yticks([])
        ax.set_title("Secondary Structure Profile")
        legend_patches = [mpatches.Patch(color=SS_COLORS[k], label=l) for k, l in [("H","Helix"),("E","Sheet"),("C","Coil")]]
        ax.legend(handles=legend_patches, loc="upper right", ncol=3)
        fig.tight_layout()
        return fig

    def plot_pae_heatmap(self) -> Optional[Figure]:
        """Plot PAE heatmap.

        Returns:
            Figure or None if no PAE data available.
        """
        if not self.has_pae:
            return None

        pae_analysis = self.analyze_pae()
        pae = pae_analysis.pae_data

        fig, ax = plt.subplots(figsize=(8, 8))

        # Use green-white color scheme (low PAE = green, high PAE = white)
        # This matches AlphaFold's standard visualization
        cmap = plt.cm.Greens_r

        im = ax.imshow(pae.pae_matrix, cmap=cmap, origin="lower", aspect="equal",
                       vmin=0, vmax=pae.max_pae)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Expected Position Error (Å)")

        # Add domain boundaries if multiple domains detected
        if pae_analysis.n_domains > 1:
            for domain in pae_analysis.domains:
                if len(domain) > 0:
                    start, end = min(domain), max(domain)
                    rect = plt.Rectangle((start - 0.5, start - 0.5), end - start + 1, end - start + 1,
                                          fill=False, edgecolor="red", linewidth=2, linestyle="--")
                    ax.add_patch(rect)

        # Add diagonal line
        n = len(pae.pae_matrix)
        ax.plot([0, n-1], [0, n-1], "k--", linewidth=0.5, alpha=0.3)

        # Stats text
        txt_lines = [f"Mean PAE: {pae_analysis.mean_pae:.1f} Å"]
        if pae.ptm is not None:
            txt_lines.append(f"pTM: {pae.ptm:.3f}")
        if pae.iptm is not None:
            txt_lines.append(f"ipTM: {pae.iptm:.3f}")
        if pae_analysis.n_domains > 1:
            txt_lines.append(f"Domains: {pae_analysis.n_domains}")
        ax.text(0.02, 0.98, "\n".join(txt_lines), transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel("Residue Number")
        ax.set_ylabel("Residue Number")
        ax.set_title("Predicted Aligned Error (PAE)")
        fig.tight_layout()
        return fig

    def plot_pae_domains(self) -> Optional[Figure]:
        """Plot PAE with domain analysis.

        Shows both the PAE heatmap and a domain segmentation bar.

        Returns:
            Figure or None if no PAE data available.
        """
        if not self.has_pae:
            return None

        pae_analysis = self.analyze_pae()
        pae = pae_analysis.pae_data
        n = pae.n_residues

        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 15], width_ratios=[15, 1],
                              hspace=0.02, wspace=0.02)

        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        # Main PAE heatmap
        cmap = plt.cm.Greens_r
        im = ax_main.imshow(pae.pae_matrix, cmap=cmap, origin="lower", aspect="equal",
                            vmin=0, vmax=pae.max_pae)

        # Domain coloring
        domain_colors = plt.cm.tab10.colors
        domain_assignment = np.zeros(n)
        for i, domain in enumerate(pae_analysis.domains):
            for idx in domain:
                domain_assignment[idx] = i + 1

        # Top bar showing domain assignment
        for i in range(n):
            color = domain_colors[int(domain_assignment[i]) % len(domain_colors)] if domain_assignment[i] > 0 else "#CCCCCC"
            ax_top.bar(i, 1, color=color, width=1.0, edgecolor="none")
        ax_top.set_xlim(-0.5, n - 0.5)
        ax_top.set_ylim(0, 1)
        ax_top.axis("off")
        ax_top.set_title(f"Predicted Aligned Error (PAE) - {pae_analysis.n_domains} Domain(s) Detected")

        # Right bar showing pLDDT if available
        if self.is_predicted:
            plddt = self.structure.plddt
            for i in range(len(plddt)):
                color = self._get_plddt_color(plddt[i])
                ax_right.barh(i, 1, color=color, height=1.0, edgecolor="none")
            ax_right.set_ylim(-0.5, len(plddt) - 0.5)
            ax_right.set_xlim(0, 1)
        ax_right.axis("off")

        # Add domain boundaries to main plot
        for domain in pae_analysis.domains:
            if len(domain) > 0:
                start, end = min(domain), max(domain)
                rect = plt.Rectangle((start - 0.5, start - 0.5), end - start + 1, end - start + 1,
                                      fill=False, edgecolor="red", linewidth=2, linestyle="--")
                ax_main.add_patch(rect)

        ax_main.set_xlabel("Residue Number")
        ax_main.set_ylabel("Residue Number")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax_right, shrink=0.8, label="Expected Position Error (Å)",
                            location="right", pad=0.3)

        # Stats
        txt_lines = [f"Mean PAE: {pae_analysis.mean_pae:.1f} Å",
                     f"Intra-domain: {pae_analysis.intra_domain_pae:.1f} Å"]
        if pae_analysis.inter_domain_pae is not None:
            txt_lines.append(f"Inter-domain: {pae_analysis.inter_domain_pae:.1f} Å")
        if pae.ptm is not None:
            txt_lines.append(f"pTM: {pae.ptm:.3f}")
        if pae.iptm is not None:
            txt_lines.append(f"ipTM: {pae.iptm:.3f}")
        ax_main.text(0.02, 0.98, "\n".join(txt_lines), transform=ax_main.transAxes, fontsize=10, va="top",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_pae_per_residue(self) -> Optional[Figure]:
        """Plot per-residue mean PAE.

        Shows the average PAE for each residue (how confident we are about its
        position relative to all other residues).

        Returns:
            Figure or None if no PAE data available.
        """
        if not self.has_pae:
            return None

        pae = self.pae_data
        pae_analysis = self.analyze_pae()

        fig, ax = plt.subplots(figsize=(12, 4))

        # Mean PAE per residue (average of row and column to get symmetric measure)
        mean_pae_per_res = (np.mean(pae.pae_matrix, axis=0) + np.mean(pae.pae_matrix, axis=1)) / 2
        residues = np.arange(1, len(mean_pae_per_res) + 1)

        # Color by domain if multiple domains
        if pae_analysis.n_domains > 1:
            domain_colors = plt.cm.tab10.colors
            domain_assignment = np.zeros(len(mean_pae_per_res), dtype=int)
            for i, domain in enumerate(pae_analysis.domains):
                for idx in domain:
                    domain_assignment[idx] = i
            colors = [domain_colors[d % len(domain_colors)] for d in domain_assignment]
            ax.bar(residues, mean_pae_per_res, color=colors, width=1.0)
        else:
            # Color by PAE value
            colors = ['#2ca02c' if v < 5 else '#ff7f0e' if v < 10 else '#d62728' for v in mean_pae_per_res]
            ax.bar(residues, mean_pae_per_res, color=colors, width=1.0)

        # Threshold lines
        ax.axhline(y=5, color="green", linestyle="--", alpha=0.7, label="5Å (high confidence)")
        ax.axhline(y=10, color="orange", linestyle="--", alpha=0.7, label="10Å (medium confidence)")

        ax.set_xlabel("Residue Number")
        ax.set_ylabel("Mean PAE (Å)")
        ax.set_title("Per-Residue Predicted Aligned Error")
        ax.set_xlim(1, len(mean_pae_per_res))
        ax.set_ylim(0, min(pae.max_pae, np.max(mean_pae_per_res) * 1.1))
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig

    def plot_msa_depth(self) -> Optional[Figure]:
        """Plot MSA depth profile.

        Shows the number of homologous sequences aligned at each position,
        which correlates with prediction confidence.

        Returns:
            Figure or None if no MSA depth data available.
        """
        if not self.has_msa_depth:
            return None

        msa = self.msa_depth
        fig, ax = plt.subplots(figsize=(12, 4))

        positions = np.arange(1, msa.n_residues + 1)
        depths = msa.depths

        # Color bars by depth (darker = deeper MSA)
        max_depth = msa.max_depth
        colors = plt.cm.Blues(depths / max_depth * 0.8 + 0.2)  # Scale to avoid too light
        ax.bar(positions, depths, color=colors, width=1.0, edgecolor='none')

        # Add mean line
        ax.axhline(y=msa.mean_depth, color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {msa.mean_depth:.0f}")

        # Add threshold lines for interpretation
        if max_depth > 100:
            ax.axhline(y=100, color="green", linestyle=":", alpha=0.7, label="100 (good coverage)")
        if max_depth > 1000:
            ax.axhline(y=1000, color="blue", linestyle=":", alpha=0.7, label="1000 (excellent coverage)")

        # Stats text box
        txt = f"Mean: {msa.mean_depth:.0f}\nMedian: {msa.median_depth:.0f}\nMax: {msa.max_depth}\nMin: {msa.min_depth}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel("Residue Position")
        ax.set_ylabel("MSA Depth")
        ax.set_title("Multiple Sequence Alignment Depth")
        ax.set_xlim(1, msa.n_residues)
        ax.set_ylim(0, max_depth * 1.1)
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig

    def _fig_to_base64(self, fig: Figure) -> str:
        """Convert figure to base64 PNG string."""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def generate_all_figures(self) -> dict:
        """Generate all characterization figures."""
        figures = {
            "plddt_distribution": self.plot_plddt_distribution(),
            "plddt_profile": self.plot_plddt_profile(),
            "contact_map": self.plot_contact_map(),
            "contact_order": self.plot_contact_order(),
            "residue_contacts": self.plot_residue_contacts(),
            "aa_composition": self.plot_aa_composition(),
        }
        # Secondary structure analysis only for proteins (DSSP doesn't work on NA)
        if not self.structure.is_nucleic_acid:
            figures["ss_composition"] = self.plot_ss_composition()
            figures["ss_profile"] = self.plot_ss_profile()

        # Add PAE figures if available
        if self.has_pae:
            pae_heatmap = self.plot_pae_heatmap()
            if pae_heatmap:
                figures["pae_heatmap"] = pae_heatmap
            pae_domains = self.plot_pae_domains()
            if pae_domains:
                figures["pae_domains"] = pae_domains
            pae_per_residue = self.plot_pae_per_residue()
            if pae_per_residue:
                figures["pae_per_residue"] = pae_per_residue

        # Add MSA depth figure if available
        if self.has_msa_depth:
            msa_depth_fig = self.plot_msa_depth()
            if msa_depth_fig:
                figures["msa_depth"] = msa_depth_fig

        return figures

    def generate_html_report(self, output_path: str) -> None:
        """Generate self-contained HTML report with embedded images."""
        seq_comp = self.analyze_sequence_composition()
        conf_stats = self.analyze_confidence()
        contact_analysis = self.analyze_contacts()
        ss_analysis = self.analyze_secondary_structure()
        pae_analysis = self.analyze_pae()  # May be None
        figures = self.generate_all_figures()
        images_b64 = {name: self._fig_to_base64(fig) for name, fig in figures.items()}
        for fig in figures.values():
            plt.close(fig)
        # Read PDB content for 3D viewer
        structure_content = ""
        structure_format = "pdb"
        if self.structure.source_path and self.structure.source_path.exists():
            structure_content = self.structure.source_path.read_text()
            suffix = self.structure.source_path.suffix.lower()
            if suffix in (".cif", ".mmcif"):
                structure_format = "cif"
        html = self._build_html(seq_comp, conf_stats, contact_analysis, ss_analysis, images_b64, structure_content, pae_analysis, structure_format)
        Path(output_path).write_text(html)

    def generate_pdf_report(self, output_path: str) -> None:
        """Generate PDF report with all figures."""
        seq_comp = self.analyze_sequence_composition()
        conf_stats = self.analyze_confidence()
        contact_analysis = self.analyze_contacts()
        ss_analysis = self.analyze_secondary_structure()
        pae_analysis = self.analyze_pae()  # May be None
        with PdfPages(output_path) as pdf:
            fig = self._create_summary_page(seq_comp, conf_stats, contact_analysis, ss_analysis, pae_analysis)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            plot_fns = [self.plot_aa_composition, self.plot_plddt_distribution, self.plot_plddt_profile,
                        self.plot_contact_map, self.plot_contact_order, self.plot_residue_contacts]
            if not self.structure.is_nucleic_acid:
                plot_fns.extend([self.plot_ss_composition, self.plot_ss_profile])
            for plot_fn in plot_fns:
                fig = plot_fn()
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            # Add PAE plots if available
            if self.has_pae:
                for plot_fn in [self.plot_pae_heatmap, self.plot_pae_domains, self.plot_pae_per_residue]:
                    fig = plot_fn()
                    if fig:
                        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            # Add MSA depth plot if available
            if self.has_msa_depth:
                fig = self.plot_msa_depth()
                if fig:
                    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            # Add glossary pages
            glossary_figs = self._create_glossary_pages()
            for fig in glossary_figs:
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    def generate_json_report(self, output_path: str, include_per_residue: bool = True) -> dict:
        """Generate JSON analysis report with all computed data.

        Args:
            output_path: Path to write JSON file.
            include_per_residue: Include per-residue arrays (pLDDT, contacts,
                secondary structure). Set False for compact output.

        Returns:
            The analysis dictionary that was written to file.
        """
        seq_comp = self.analyze_sequence_composition()
        conf_stats = self.analyze_confidence()
        contact_analysis = self.analyze_contacts()
        ss_analysis = self.analyze_secondary_structure()
        pae_analysis = self.analyze_pae()  # May be None

        analysis: dict = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "tool": "protein_compare",
                "command": "characterize",
                "structure_name": self.structure.name,
                "structure_type": self.structure_type,
                "source_path": str(self.structure.source_path) if self.structure.source_path else None,
            },
            "structure": {
                "name": self.structure.name,
                "molecule_type": self.structure.molecule_type,
                "n_residues": self.structure.n_residues,
                "sequence": self.structure.sequence,
                "chains": seq_comp.chains,
                "is_predicted": self.is_predicted,
            },
            "sequence_composition": seq_comp.to_dict(),
            "confidence": conf_stats.to_dict(),
            "contacts": contact_analysis.to_dict(include_matrices=include_per_residue),
            "secondary_structure": ss_analysis.to_dict(),
        }

        # Per-residue arrays (for charts)
        if include_per_residue:
            analysis["per_residue"] = {
                "plddt": [round(float(v), 2) for v in self.structure.plddt],
                "residue_ids": [
                    {"chain": rid[0], "resnum": rid[1]}
                    for rid in self.structure.residue_ids
                ],
            }

        # PAE analysis (optional)
        if pae_analysis is not None:
            analysis["pae"] = pae_analysis.to_dict(include_matrix=include_per_residue)

        # Chai scores (optional)
        if self.has_chai_scores:
            scores = self.chai_scores
            analysis["chai_scores"] = {
                "aggregate_score": round(float(scores.aggregate_score), 4),
                "ptm": round(float(scores.ptm), 4),
                "iptm": round(float(scores.iptm), 4),
                "has_inter_chain_clashes": bool(scores.has_inter_chain_clashes),
                "n_chains": scores.n_chains,
                "is_multimer": scores.is_multimer,
            }

        # MSA depth (optional)
        if self.has_msa_depth:
            msa = self.msa_depth
            analysis["msa_depth"] = {
                "mean_depth": round(float(msa.mean_depth), 1),
                "median_depth": round(float(msa.median_depth), 1),
                "max_depth": int(msa.max_depth),
                "min_depth": int(msa.min_depth),
                "n_residues": msa.n_residues,
            }
            if include_per_residue:
                analysis["msa_depth"]["per_residue_depth"] = msa.depths.tolist()

        with open(output_path, "w") as f:
            _json.dump(analysis, f, indent=2)

        return analysis

    def _create_summary_page(self, seq_comp, conf_stats, contact_analysis, ss_analysis, pae_analysis=None) -> Figure:
        """Create summary page for PDF."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.95, "Structure Characterization Report", fontsize=20, ha="center", fontweight="bold")
        ax.text(0.5, 0.90, self.structure.name, fontsize=16, ha="center", style="italic")

        # Structure-type-specific labels
        if self.is_predicted:
            score_name = "pLDDT"
            section_name = "CONFIDENCE ANALYSIS"
            high_label = "Very High (≥90)"
            low_label = "Very Low (<50)"
            high_value = conf_stats.n_very_high
            low_value = conf_stats.n_very_low
        else:
            score_name = "B-factor"
            section_name = "B-FACTOR ANALYSIS"
            high_label = "Flexible (>60)"
            low_label = "Ordered (<20)"
            high_value = int(np.sum(self.structure.plddt > 60))
            low_value = int(np.sum(self.structure.plddt < 20))

        lines = [
            "", "=" * 50, "BASIC INFORMATION", "=" * 50,
            f"Structure Name:     {self.structure.name}",
            f"Number of Residues: {seq_comp.length}",
            f"Chain(s):           {', '.join(seq_comp.chains)}",
            f"Molecular Weight:   {seq_comp.molecular_weight/1000:.1f} kDa",
            f"Structure Type:     {self.structure_type.title()}",
            "", "=" * 50, section_name, "=" * 50,
            f"Mean {score_name}:         {conf_stats.mean:.1f}",
            f"Median {score_name}:       {conf_stats.median:.1f}",
            f"High Confidence:    {conf_stats.frac_confident:.1%}" if self.is_predicted else f"Ordered (<30):      {np.sum(self.structure.plddt < 30) / len(self.structure.plddt):.1%}",
            f"{high_label}:    {high_value} residues",
            f"{low_label}:     {low_value} residues",
        ]

        # Add PAE section if available
        if pae_analysis is not None:
            lines.extend([
                "", "=" * 50, "PREDICTED ALIGNED ERROR (PAE)", "=" * 50,
                f"Mean PAE:           {pae_analysis.mean_pae:.1f} Å",
                f"Median PAE:         {pae_analysis.median_pae:.1f} Å",
                f"Domains Detected:   {pae_analysis.n_domains}",
                f"Intra-domain PAE:   {pae_analysis.intra_domain_pae:.1f} Å",
            ])
            if pae_analysis.inter_domain_pae is not None:
                lines.append(f"Inter-domain PAE:   {pae_analysis.inter_domain_pae:.1f} Å")
            if pae_analysis.pae_data.ptm is not None:
                lines.append(f"pTM Score:          {pae_analysis.pae_data.ptm:.3f}")
            if pae_analysis.pae_data.iptm is not None:
                lines.append(f"ipTM Score:         {pae_analysis.pae_data.iptm:.3f}")

        # Add Chai scores section if available
        if self.has_chai_scores:
            scores = self.chai_scores
            quality = "High" if scores.ptm >= 0.8 else ("Moderate" if scores.ptm >= 0.5 else "Low")
            lines.extend([
                "", "=" * 50, "CHAI PREDICTION SCORES", "=" * 50,
                f"pTM Score:          {scores.ptm:.3f}",
                f"ipTM Score:         {scores.iptm:.3f}",
                f"Aggregate Score:    {scores.aggregate_score:.3f}",
                f"Quality:            {quality} confidence",
            ])
            if scores.is_multimer:
                lines.append(f"Chains:             {scores.n_chains}")
            if scores.has_inter_chain_clashes:
                lines.append(f"Inter-chain Clashes: Yes")

        # Add MSA depth section if available
        if self.has_msa_depth:
            msa = self.msa_depth
            quality = "Excellent" if msa.mean_depth >= 1000 else ("Good" if msa.mean_depth >= 100 else "Limited")
            lines.extend([
                "", "=" * 50, "MSA DEPTH ANALYSIS", "=" * 50,
                f"Mean Depth:         {msa.mean_depth:.0f}",
                f"Median Depth:       {msa.median_depth:.0f}",
                f"Max Depth:          {msa.max_depth}",
                f"Min Depth:          {msa.min_depth}",
                f"Coverage Quality:   {quality}",
            ])

        lines.extend([
            "", "=" * 50, "CONTACT ANALYSIS", "=" * 50,
            f"Total Contacts:     {contact_analysis.n_contacts}",
            f"Contact Density:    {contact_analysis.contact_density:.3f}",
            f"Long-range (>12):   {contact_analysis.n_long_range + contact_analysis.n_very_long_range}",
            "", "=" * 50, "SECONDARY STRUCTURE", "=" * 50,
            f"Helix:              {ss_analysis.helix_fraction:.1%} ({ss_analysis.helix_count} residues)",
            f"Sheet:              {ss_analysis.sheet_fraction:.1%} ({ss_analysis.sheet_count} residues)",
            f"Coil:               {ss_analysis.coil_fraction:.1%} ({ss_analysis.coil_count} residues)",
            "", "=" * 50, "SEQUENCE COMPOSITION", "=" * 50,
            f"Hydrophobic:        {seq_comp.type_fractions.get('hydrophobic', 0):.1%}",
            f"Polar:              {seq_comp.type_fractions.get('polar', 0):.1%}",
            f"Charged (+):        {seq_comp.type_fractions.get('positive', 0):.1%}",
            f"Charged (-):        {seq_comp.type_fractions.get('negative', 0):.1%}",
        ])
        ax.text(0.1, 0.85, "\n".join(lines), fontsize=10, ha="left", va="top", family="monospace", transform=ax.transAxes)
        return fig

    def _build_html(self, seq_comp, conf_stats, contact_analysis, ss_analysis, images_b64, structure_content="", pae_analysis=None, structure_format="pdb") -> str:
        """Build HTML report string."""
        # Escape structure content for JavaScript
        structure_escaped = structure_content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$") if structure_content else ""

        # Structure-type-specific labels and thresholds
        if self.is_predicted:
            score_name = "pLDDT"
            score_unit = ""
            section_title = "Confidence Analysis (pLDDT)"
            mean_label = "Mean pLDDT"
            median_label = "Median pLDDT"
            high_label = "Very High (≥90)"
            low_label = "Very Low (<50)"
            high_value = conf_stats.n_very_high
            low_value = conf_stats.n_very_low
            highlight_class = "highlight" if conf_stats.mean >= 70 else ("warning" if conf_stats.mean >= 50 else "")
            summary_quality_label = "High Confidence"
            summary_quality_value = f"{conf_stats.frac_confident:.0%}"
            dist_caption = "Distribution of pLDDT confidence scores"
            profile_caption = "Per-residue pLDDT profile"
            color_btn_label = "Color by pLDDT"
        else:
            score_name = "B-factor"
            score_unit = " Ų"
            section_title = "B-factor Analysis (Flexibility)"
            mean_label = "Mean B-factor"
            median_label = "Median B-factor"
            high_label = "Flexible (>60)"
            low_label = "Ordered (<20)"
            # For B-factors, "high" means flexible (high B), "low" means ordered (low B)
            high_value = int(np.sum(self.structure.plddt > 60))
            low_value = int(np.sum(self.structure.plddt < 20))
            # For B-factors, lower is better (more ordered)
            highlight_class = "highlight" if conf_stats.mean < 30 else ("warning" if conf_stats.mean < 50 else "")
            summary_quality_label = "Ordered (<30)"
            ordered_frac = np.sum(self.structure.plddt < 30) / len(self.structure.plddt)
            summary_quality_value = f"{ordered_frac:.0%}"
            dist_caption = "Distribution of B-factor (atomic displacement) values"
            profile_caption = "Per-residue B-factor profile"
            color_btn_label = "Color by B-factor"

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Structure Characterization: {self.structure.name}</title>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 15px 0; }}
        .metric-box {{ background: #ecf0f1; padding: 15px; border-radius: 6px; text-align: center; }}
        .metric-box.highlight {{ background: #d5f4e6; }}
        .metric-box.warning {{ background: #ffeaa7; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        .figure {{ text-align: center; margin: 20px 0; }}
        .figure img {{ max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .figure-caption {{ font-style: italic; color: #666; margin-top: 8px; }}
        .sequence {{ font-family: monospace; font-size: 12px; word-break: break-all; background: #f8f9fa; padding: 10px; border-radius: 4px; max-height: 150px; overflow-y: auto; }}
        .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; }}
        .glossary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 15px; }}
        .glossary-item {{ background: #f8f9fa; padding: 12px 15px; border-radius: 6px; border-left: 4px solid #3498db; }}
        .glossary-term {{ font-weight: bold; color: #2c3e50; margin-bottom: 5px; }}
        .glossary-def {{ font-size: 13px; color: #555; line-height: 1.5; }}
        #viewer-container {{ width: 100%; height: 500px; position: relative; border-radius: 8px; overflow: hidden; }}
        #viewer {{ width: 100%; height: 100%; }}
        .viewer-controls {{ display: flex; gap: 10px; margin-top: 10px; flex-wrap: wrap; justify-content: center; }}
        .viewer-controls button {{ padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; transition: background 0.2s; }}
        .viewer-controls button {{ background: #3498db; color: white; }}
        .viewer-controls button:hover {{ background: #2980b9; }}
        .viewer-controls button.active {{ background: #2c3e50; }}
    </style>
</head>
<body>
    <h1>Structure Characterization Report</h1>
    <p style="font-size: 18px; color: #666;"><strong>{self.structure.name}</strong></p>

    <div class="section" id="structure-viewer">
        <h2>3D Structure</h2>
        <div id="viewer-container">
            <div id="viewer"></div>
        </div>
        <div class="viewer-controls">
            <button onclick="setStyle('cartoon')" id="btn-cartoon" class="active">Cartoon</button>
            <button onclick="setStyle('stick')" id="btn-stick">Sticks</button>
            <button onclick="setStyle('sphere')" id="btn-sphere">Spheres</button>
            <button onclick="setStyle('line')" id="btn-line">Lines</button>
            <button onclick="colorBy('ss')" id="btn-ss">Color by SS</button>
            <button onclick="colorBy('bfactor')" id="btn-bfactor">{color_btn_label}</button>
            <button onclick="colorBy('chain')" id="btn-chain">Color by Chain</button>
            <button onclick="viewer.spin(spinning = !spinning)" id="btn-spin">Spin</button>
            <button onclick="viewer.zoomTo(); viewer.render();">Reset View</button>
        </div>
        <div class="figure-caption">Interactive 3D viewer. Drag to rotate, scroll to zoom, right-click drag to translate.</div>
    </div>

    <div class="section" id="summary"><h2>Summary</h2>
        <div class="metrics-grid">
            <div class="metric-box"><div class="metric-value">{seq_comp.length}</div><div class="metric-label">{"Nucleotides" if self.structure.is_nucleic_acid else "Residues"}</div></div>
            {"" if self.structure.is_nucleic_acid else f'<div class="metric-box"><div class="metric-value">{seq_comp.molecular_weight/1000:.1f} kDa</div><div class="metric-label">Molecular Weight</div></div>'}
            <div class="metric-box {highlight_class}"><div class="metric-value">{conf_stats.mean:.1f}{score_unit}</div><div class="metric-label">{mean_label}</div></div>
            <div class="metric-box highlight"><div class="metric-value">{summary_quality_value}</div><div class="metric-label">{summary_quality_label}</div></div>
            <div class="metric-box"><div class="metric-value">{contact_analysis.n_contacts}</div><div class="metric-label">Contacts</div></div>
            {"" if self.structure.is_nucleic_acid else f'<div class="metric-box"><div class="metric-value">{ss_analysis.helix_fraction:.0%}/{ss_analysis.sheet_fraction:.0%}</div><div class="metric-label">Helix/Sheet</div></div>'}
        </div>
    </div>
    <div class="section" id="sequence"><h2>{"Nucleotide" if self.structure.is_nucleic_acid else "Sequence"} Analysis</h2>
        <div class="metrics-grid">
            {f"""<div class="metric-box"><div class="metric-value">{seq_comp.type_fractions.get("purine", 0):.0%}</div><div class="metric-label">Purine (A/G)</div></div>
            <div class="metric-box"><div class="metric-value">{seq_comp.type_fractions.get("pyrimidine", 0):.0%}</div><div class="metric-label">Pyrimidine (C/T/U)</div></div>""" if self.structure.is_nucleic_acid else f"""<div class="metric-box"><div class="metric-value">{seq_comp.type_fractions.get("hydrophobic", 0):.0%}</div><div class="metric-label">Hydrophobic</div></div>
            <div class="metric-box"><div class="metric-value">{seq_comp.type_fractions.get("polar", 0):.0%}</div><div class="metric-label">Polar</div></div>
            <div class="metric-box"><div class="metric-value">{seq_comp.type_fractions.get("positive", 0):.0%}</div><div class="metric-label">Positive (+)</div></div>
            <div class="metric-box"><div class="metric-value">{seq_comp.type_fractions.get("negative", 0):.0%}</div><div class="metric-label">Negative (-)</div></div>"""}
        </div>
        <div class="figure"><img src="data:image/png;base64,{images_b64["aa_composition"]}" alt="AA Composition"><div class="figure-caption">Amino acid composition by residue type</div></div>
        <h3>Sequence</h3><div class="sequence">{self.structure.sequence}</div>
    </div>
    <div class="section" id="confidence"><h2>{section_title}</h2>
        <div class="metrics-grid">
            <div class="metric-box"><div class="metric-value">{conf_stats.mean:.1f}{score_unit}</div><div class="metric-label">{mean_label}</div></div>
            <div class="metric-box"><div class="metric-value">{conf_stats.median:.1f}{score_unit}</div><div class="metric-label">{median_label}</div></div>
            <div class="metric-box"><div class="metric-value">{high_value}</div><div class="metric-label">{high_label}</div></div>
            <div class="metric-box"><div class="metric-value">{low_value}</div><div class="metric-label">{low_label}</div></div>
        </div>
        <div class="figure"><img src="data:image/png;base64,{images_b64["plddt_distribution"]}" alt="{score_name} Distribution"><div class="figure-caption">{dist_caption}</div></div>
        <div class="figure"><img src="data:image/png;base64,{images_b64["plddt_profile"]}" alt="{score_name} Profile"><div class="figure-caption">{profile_caption}</div></div>
    </div>
    <div class="section" id="contacts"><h2>Contact Analysis</h2>
        <div class="metrics-grid">
            <div class="metric-box"><div class="metric-value">{contact_analysis.n_contacts}</div><div class="metric-label">Total Contacts</div></div>
            <div class="metric-box"><div class="metric-value">{contact_analysis.contact_density:.3f}</div><div class="metric-label">Contact Density</div></div>
            <div class="metric-box"><div class="metric-value">{contact_analysis.n_long_range + contact_analysis.n_very_long_range}</div><div class="metric-label">Long-Range (&gt;12)</div></div>
            <div class="metric-box"><div class="metric-value">{self.contact_cutoff} Å</div><div class="metric-label">Cutoff Distance</div></div>
        </div>
        <div class="figure"><img src="data:image/png;base64,{images_b64["contact_map"]}" alt="Contact Map"><div class="figure-caption">Contact map (Cα-Cα distance &lt; {self.contact_cutoff} Å)</div></div>
        <div class="figure"><img src="data:image/png;base64,{images_b64["contact_order"]}" alt="Contact Order"><div class="figure-caption">Contact order distribution</div></div>
        <div class="figure"><img src="data:image/png;base64,{images_b64["residue_contacts"]}" alt="Residue Contacts"><div class="figure-caption">Contacts per residue</div></div>
    </div>
    {"" if self.structure.is_nucleic_acid else f"""<div class="section" id="secondary"><h2>Secondary Structure</h2>
        <div class="metrics-grid">
            <div class="metric-box"><div class="metric-value">{ss_analysis.helix_fraction:.1%}</div><div class="metric-label">Helix ({ss_analysis.helix_count} res)</div></div>
            <div class="metric-box"><div class="metric-value">{ss_analysis.sheet_fraction:.1%}</div><div class="metric-label">Sheet ({ss_analysis.sheet_count} res)</div></div>
            <div class="metric-box"><div class="metric-value">{ss_analysis.coil_fraction:.1%}</div><div class="metric-label">Coil ({ss_analysis.coil_count} res)</div></div>
        </div>
        <div class="figure"><img src="data:image/png;base64,{images_b64['ss_composition']}" alt="SS Composition"><div class="figure-caption">Secondary structure composition</div></div>
        <div class="figure"><img src="data:image/png;base64,{images_b64['ss_profile']}" alt="SS Profile"><div class="figure-caption">Secondary structure profile</div></div>
    </div>"""}

    {self._build_pae_html_section(pae_analysis, images_b64)}

    {self._build_chai_scores_html_section()}

    {self._build_msa_html_section(images_b64)}

    <div class="section" id="glossary">
        <h2>Glossary of Terms</h2>
        <div class="glossary-grid">
            {self._build_glossary_html()}
        </div>
    </div>
    <div class="footer">Generated by protein_compare v0.1.0</div>

    <script>
        let viewer = null;
        let spinning = false;
        let currentStyle = 'cartoon';
        let currentColor = 'ss';

        const structureData = `{structure_escaped}`;
        const structureFormat = '{structure_format}';

        document.addEventListener('DOMContentLoaded', function() {{
            if (structureData.trim()) {{
                let element = document.getElementById('viewer');
                let config = {{ backgroundColor: 'white' }};
                viewer = $3Dmol.createViewer(element, config);
                viewer.addModel(structureData, structureFormat);
                applyStyle();
                viewer.zoomTo();
                viewer.render();
            }} else {{
                document.getElementById('viewer-container').innerHTML = '<p style="text-align:center;padding:50px;color:#999;">Structure data not available</p>';
            }}
        }});

        function setStyle(style) {{
            currentStyle = style;
            document.querySelectorAll('.viewer-controls button').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-' + style).classList.add('active');
            applyStyle();
        }}

        function colorBy(scheme) {{
            currentColor = scheme;
            applyStyle();
        }}

        function applyStyle() {{
            if (!viewer) return;
            viewer.setStyle({{}}, {{}});

            let styleSpec = {{}};
            if (currentStyle === 'cartoon') {{
                styleSpec = {{ cartoon: {{ color: 'spectrum' }} }};
            }} else if (currentStyle === 'stick') {{
                styleSpec = {{ stick: {{ radius: 0.15 }} }};
            }} else if (currentStyle === 'sphere') {{
                styleSpec = {{ sphere: {{ scale: 0.3 }} }};
            }} else if (currentStyle === 'line') {{
                styleSpec = {{ line: {{}} }};
            }}

            if (currentColor === 'ss') {{
                if (currentStyle === 'cartoon') {{
                    styleSpec.cartoon.color = 'ss';
                }} else {{
                    styleSpec[currentStyle].colorscheme = 'ssJmol';
                }}
            }} else if (currentColor === 'bfactor') {{
                // Color by B-factor (pLDDT) - blue high, red low
                if (currentStyle === 'cartoon') {{
                    styleSpec.cartoon.color = 'b';
                    styleSpec.cartoon.colorscheme = {{ prop: 'b', gradient: 'roygb', min: 0, max: 100 }};
                }} else {{
                    styleSpec[currentStyle].colorscheme = {{ prop: 'b', gradient: 'roygb', min: 0, max: 100 }};
                }}
            }} else if (currentColor === 'chain') {{
                if (currentStyle === 'cartoon') {{
                    styleSpec.cartoon.color = 'chain';
                }} else {{
                    styleSpec[currentStyle].colorscheme = 'chain';
                }}
            }}

            viewer.setStyle({{}}, styleSpec);
            viewer.render();
        }}
    </script>
</body>
</html>
'''

    def _build_glossary_html(self) -> str:
        """Build HTML for glossary section."""
        items = []
        for key in GLOSSARY:
            entry = GLOSSARY[key]
            items.append(
                f'<div class="glossary-item">'
                f'<div class="glossary-term">{entry["term"]}</div>'
                f'<div class="glossary-def">{entry["definition"]}</div>'
                f'</div>'
            )
        return "\n            ".join(items)

    def _build_pae_html_section(self, pae_analysis: Optional[PAEAnalysis], images_b64: dict) -> str:
        """Build HTML for PAE section if PAE data is available."""
        if pae_analysis is None or "pae_heatmap" not in images_b64:
            return ""  # No PAE section if no data

        # Build metrics for PAE
        ptm_html = ""
        if pae_analysis.pae_data.ptm is not None:
            ptm_html = f'''<div class="metric-box highlight"><div class="metric-value">{pae_analysis.pae_data.ptm:.3f}</div><div class="metric-label">pTM Score</div></div>'''

        iptm_html = ""
        if pae_analysis.pae_data.iptm is not None:
            iptm_html = f'''<div class="metric-box"><div class="metric-value">{pae_analysis.pae_data.iptm:.3f}</div><div class="metric-label">ipTM Score</div></div>'''

        inter_domain_html = ""
        if pae_analysis.inter_domain_pae is not None:
            inter_domain_html = f'''<div class="metric-box"><div class="metric-value">{pae_analysis.inter_domain_pae:.1f} Å</div><div class="metric-label">Inter-domain PAE</div></div>'''

        # PAE domains figure (if available)
        pae_domains_html = ""
        if "pae_domains" in images_b64:
            pae_domains_html = f'''<div class="figure"><img src="data:image/png;base64,{images_b64["pae_domains"]}" alt="PAE Domains"><div class="figure-caption">PAE heatmap with domain segmentation</div></div>'''

        # Per-residue PAE figure (if available)
        pae_per_residue_html = ""
        if "pae_per_residue" in images_b64:
            pae_per_residue_html = f'''<div class="figure"><img src="data:image/png;base64,{images_b64["pae_per_residue"]}" alt="PAE Per Residue"><div class="figure-caption">Per-residue mean PAE</div></div>'''

        return f'''
    <div class="section" id="pae"><h2>Predicted Aligned Error (PAE)</h2>
        <p style="color: #666; margin-bottom: 15px;">PAE measures the expected position error (in Ångströms) between residue pairs. Low PAE indicates high confidence in relative positioning. Off-diagonal blocks with high PAE may indicate domain boundaries or flexible regions.</p>
        <div class="metrics-grid">
            <div class="metric-box"><div class="metric-value">{pae_analysis.mean_pae:.1f} Å</div><div class="metric-label">Mean PAE</div></div>
            <div class="metric-box"><div class="metric-value">{pae_analysis.median_pae:.1f} Å</div><div class="metric-label">Median PAE</div></div>
            <div class="metric-box"><div class="metric-value">{pae_analysis.n_domains}</div><div class="metric-label">Domains Detected</div></div>
            <div class="metric-box"><div class="metric-value">{pae_analysis.intra_domain_pae:.1f} Å</div><div class="metric-label">Intra-domain PAE</div></div>
            {inter_domain_html}
            {ptm_html}
            {iptm_html}
        </div>
        <div class="figure"><img src="data:image/png;base64,{images_b64["pae_heatmap"]}" alt="PAE Heatmap"><div class="figure-caption">Predicted Aligned Error matrix (green = low error, white = high error)</div></div>
        {pae_domains_html}
        {pae_per_residue_html}
    </div>
'''

    def _build_chai_scores_html_section(self) -> str:
        """Build HTML for Chai scores section if Chai scores are available."""
        if not self.has_chai_scores:
            return ""  # No Chai scores section if no data

        scores = self.chai_scores

        # Determine quality assessment
        if scores.ptm >= 0.8:
            ptm_class = "highlight"
            quality = "High confidence prediction"
        elif scores.ptm >= 0.5:
            ptm_class = ""
            quality = "Moderate confidence prediction"
        else:
            ptm_class = "warning"
            quality = "Low confidence prediction"

        # ipTM metric (relevant for multimers)
        iptm_html = ""
        if scores.is_multimer or scores.iptm > 0:
            iptm_html = f'''<div class="metric-box"><div class="metric-value">{scores.iptm:.3f}</div><div class="metric-label">ipTM Score</div></div>'''

        # Clash detection
        clash_html = ""
        if scores.has_inter_chain_clashes:
            clash_html = '''<div class="metric-box warning"><div class="metric-value">Yes</div><div class="metric-label">Inter-chain Clashes</div></div>'''

        # Chains info for multimers
        chains_html = ""
        if scores.is_multimer:
            chains_html = f'''<div class="metric-box"><div class="metric-value">{scores.n_chains}</div><div class="metric-label">Chains</div></div>'''

        return f'''
    <div class="section" id="chai-scores"><h2>Chai Prediction Scores</h2>
        <p style="color: #666; margin-bottom: 15px;">Chai confidence metrics assess the quality of the structure prediction. pTM (predicted TM-score) measures overall fold confidence, while ipTM assesses interface quality in multimers.</p>
        <div class="metrics-grid">
            <div class="metric-box {ptm_class}"><div class="metric-value">{scores.ptm:.3f}</div><div class="metric-label">pTM Score</div></div>
            {iptm_html}
            <div class="metric-box"><div class="metric-value">{scores.aggregate_score:.3f}</div><div class="metric-label">Aggregate Score</div></div>
            {chains_html}
            {clash_html}
        </div>
        <p style="color: #555; font-style: italic; margin-top: 15px;">Quality assessment: {quality}</p>
    </div>
'''

    def _build_msa_html_section(self, images_b64: dict) -> str:
        """Build HTML for MSA depth section if MSA data is available."""
        if not self.has_msa_depth or "msa_depth" not in images_b64:
            return ""  # No MSA section if no data

        msa = self.msa_depth

        # Assess MSA quality
        if msa.mean_depth >= 1000:
            depth_class = "highlight"
            quality = "Excellent MSA coverage"
        elif msa.mean_depth >= 100:
            depth_class = ""
            quality = "Good MSA coverage"
        else:
            depth_class = "warning"
            quality = "Limited MSA coverage - predictions may be less reliable"

        return f'''
    <div class="section" id="msa-depth"><h2>MSA Depth Analysis</h2>
        <p style="color: #666; margin-bottom: 15px;">Multiple Sequence Alignment (MSA) depth indicates the number of homologous sequences aligned at each position. Higher depth generally correlates with better prediction quality.</p>
        <div class="metrics-grid">
            <div class="metric-box {depth_class}"><div class="metric-value">{msa.mean_depth:.0f}</div><div class="metric-label">Mean Depth</div></div>
            <div class="metric-box"><div class="metric-value">{msa.median_depth:.0f}</div><div class="metric-label">Median Depth</div></div>
            <div class="metric-box"><div class="metric-value">{msa.max_depth}</div><div class="metric-label">Max Depth</div></div>
            <div class="metric-box"><div class="metric-value">{msa.min_depth}</div><div class="metric-label">Min Depth</div></div>
        </div>
        <div class="figure"><img src="data:image/png;base64,{images_b64["msa_depth"]}" alt="MSA Depth"><div class="figure-caption">MSA depth per residue position</div></div>
        <p style="color: #555; font-style: italic; margin-top: 15px;">{quality}</p>
    </div>
'''

    def _create_glossary_pages(self) -> list:
        """Create glossary pages for PDF report."""
        figures = []
        entries = list(GLOSSARY.values())
        entries_per_page = 8

        for page_start in range(0, len(entries), entries_per_page):
            page_entries = entries[page_start:page_start + entries_per_page]
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")

            # Title
            if page_start == 0:
                ax.text(0.5, 0.96, "Glossary of Terms", fontsize=18, ha="center", fontweight="bold")
                start_y = 0.90
            else:
                ax.text(0.5, 0.96, "Glossary of Terms (continued)", fontsize=18, ha="center", fontweight="bold")
                start_y = 0.90

            y = start_y
            for entry in page_entries:
                # Term in bold
                ax.text(0.05, y, entry["term"], fontsize=11, fontweight="bold",
                        transform=ax.transAxes, verticalalignment="top")
                y -= 0.03

                # Definition with word wrap
                definition = entry["definition"]
                # Simple word wrapping for PDF
                words = definition.split()
                lines = []
                current_line = []
                for word in words:
                    current_line.append(word)
                    if len(" ".join(current_line)) > 85:
                        if len(current_line) > 1:
                            current_line.pop()
                            lines.append(" ".join(current_line))
                            current_line = [word]
                        else:
                            lines.append(" ".join(current_line))
                            current_line = []
                if current_line:
                    lines.append(" ".join(current_line))

                for line in lines:
                    ax.text(0.07, y, line, fontsize=9, color="#444",
                            transform=ax.transAxes, verticalalignment="top")
                    y -= 0.025

                y -= 0.02  # Extra space between entries

            figures.append(fig)

        return figures
