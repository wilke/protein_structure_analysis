"""PDB and mmCIF file parsing with confidence score extraction.

Handles PDB and mmCIF files from AlphaFold, ESMFold, Chai, and Boltz,
extracting pLDDT confidence scores from the B-factor column.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue


@dataclass
class PAEData:
    """Container for Predicted Aligned Error (PAE) data from AlphaFold.

    PAE represents the expected position error (in Ångströms) of residue j
    when the structure is aligned on residue i. Low PAE values indicate
    high confidence in the relative positions of residue pairs.
    """

    pae_matrix: np.ndarray  # N×N PAE matrix in Ångströms
    max_pae: float  # Maximum PAE value (typically 31.75 for AF2)
    ptm: Optional[float] = None  # Predicted TM-score
    iptm: Optional[float] = None  # Interface pTM (for multimers)

    @property
    def n_residues(self) -> int:
        """Number of residues."""
        return len(self.pae_matrix)

    @property
    def mean_pae(self) -> float:
        """Mean PAE across all residue pairs."""
        return float(np.mean(self.pae_matrix))

    @property
    def median_pae(self) -> float:
        """Median PAE across all residue pairs."""
        return float(np.median(self.pae_matrix))

    def get_domain_pae(self, domain1_indices: list[int], domain2_indices: list[int]) -> float:
        """Get mean PAE between two sets of residues (potential domains).

        Args:
            domain1_indices: Residue indices for first domain.
            domain2_indices: Residue indices for second domain.

        Returns:
            Mean PAE between the two domains.
        """
        submatrix = self.pae_matrix[np.ix_(domain1_indices, domain2_indices)]
        return float(np.mean(submatrix))

    def identify_domains(self, pae_cutoff: float = 5.0, min_domain_size: int = 20) -> list[list[int]]:
        """Identify potential domains based on PAE clustering.

        Residues within a domain have low PAE to each other but high PAE
        to residues in other domains.

        Args:
            pae_cutoff: PAE threshold for considering residues in same domain.
            min_domain_size: Minimum number of residues to form a domain.

        Returns:
            List of domains, each domain is a list of residue indices.
        """
        n = self.n_residues
        # Create connectivity matrix based on PAE cutoff
        connected = self.pae_matrix < pae_cutoff
        # Make symmetric (use AND to be conservative)
        connected = connected & connected.T

        # Simple clustering: group consecutive residues with mutual low PAE
        domains = []
        visited = set()

        for i in range(n):
            if i in visited:
                continue

            # Start new domain
            domain = [i]
            visited.add(i)

            # Expand domain by adding connected residues
            for j in range(i + 1, n):
                if j in visited:
                    continue
                # Check if j is connected to most residues in current domain
                connections = sum(connected[j, k] for k in domain)
                if connections >= len(domain) * 0.5:  # 50% connectivity threshold
                    domain.append(j)
                    visited.add(j)

            if len(domain) >= min_domain_size:
                domains.append(domain)

        # Merge small gaps between domains
        if not domains:
            return [[i for i in range(n)]]  # Single domain

        return domains


@dataclass
class ChaiScores:
    """Container for Chai prediction confidence scores.

    Chai outputs scores in NPZ format containing pTM, ipTM, and
    per-chain metrics for structure quality assessment.
    """

    aggregate_score: float  # Combined ranking score
    ptm: float  # Predicted TM-score (0-1)
    iptm: float  # Interface pTM (0-1, relevant for multimers)
    per_chain_ptm: np.ndarray  # Per-chain pTM values, shape (n_chains,)
    per_chain_pair_iptm: np.ndarray  # Pairwise chain ipTM, shape (n_chains, n_chains)
    has_inter_chain_clashes: bool  # Whether inter-chain clashes detected
    chain_chain_clashes: np.ndarray  # Clash counts between chain pairs
    source_path: Optional[Path] = None

    @property
    def n_chains(self) -> int:
        """Number of chains in the prediction."""
        if self.per_chain_ptm.ndim == 0:
            return 1
        return len(self.per_chain_ptm)

    @property
    def is_multimer(self) -> bool:
        """Whether this is a multimer prediction."""
        return self.n_chains > 1


class ChaiScoresLoader:
    """Load Chai prediction scores from NPZ files."""

    def load(self, path: str | Path) -> ChaiScores:
        """Load Chai scores from NPZ file.

        Args:
            path: Path to scores.model_idx_*.npz file.

        Returns:
            ChaiScores object with confidence metrics.

        Raises:
            FileNotFoundError: If NPZ file doesn't exist.
            ValueError: If file format is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Chai scores file not found: {path}")

        try:
            data = np.load(path)
        except Exception as e:
            raise ValueError(f"Failed to load NPZ file {path}: {e}")

        required_keys = {"aggregate_score", "ptm", "iptm", "per_chain_ptm",
                         "per_chain_pair_iptm", "has_inter_chain_clashes",
                         "chain_chain_clashes"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"Missing keys in Chai scores file: {missing}")

        return ChaiScores(
            aggregate_score=float(data["aggregate_score"].flat[0]),
            ptm=float(data["ptm"].flat[0]),
            iptm=float(data["iptm"].flat[0]),
            per_chain_ptm=data["per_chain_ptm"].squeeze(),
            per_chain_pair_iptm=data["per_chain_pair_iptm"].squeeze(),
            has_inter_chain_clashes=bool(data["has_inter_chain_clashes"].flat[0]),
            chain_chain_clashes=data["chain_chain_clashes"].squeeze(),
            source_path=path,
        )

    def load_all_models(self, output_dir: str | Path) -> list[ChaiScores]:
        """Load scores for all models in a Chai output directory.

        Args:
            output_dir: Path to Chai output directory containing
                        scores.model_idx_*.npz files.

        Returns:
            List of ChaiScores, one per model, sorted by model index.
        """
        output_dir = Path(output_dir)
        score_files = sorted(output_dir.glob("scores.model_idx_*.npz"))
        return [self.load(f) for f in score_files]


@dataclass
class MSADepth:
    """Container for MSA (Multiple Sequence Alignment) depth data from Chai.

    MSA depth indicates the number of homologous sequences aligned at each
    position, which correlates with prediction confidence.
    """

    positions: np.ndarray  # Residue positions, shape (n_residues,)
    depths: np.ndarray  # MSA depth per position, shape (n_residues,)
    source_path: Optional[Path] = None

    @property
    def n_residues(self) -> int:
        """Number of residues."""
        return len(self.positions)

    @property
    def mean_depth(self) -> float:
        """Mean MSA depth across all positions."""
        return float(np.mean(self.depths))

    @property
    def median_depth(self) -> float:
        """Median MSA depth."""
        return float(np.median(self.depths))

    @property
    def max_depth(self) -> int:
        """Maximum MSA depth."""
        return int(np.max(self.depths))

    @property
    def min_depth(self) -> int:
        """Minimum MSA depth."""
        return int(np.min(self.depths))


class MSADepthLoader:
    """Load MSA depth data from Chai parquet files.

    Requires pyarrow or fastparquet to be installed.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if parquet reading is available (pyarrow installed)."""
        try:
            import pandas as pd
            # Try to import a parquet engine
            pd.read_parquet  # Just check the method exists
            # Actually test if an engine is available
            try:
                import pyarrow
                return True
            except ImportError:
                pass
            try:
                import fastparquet
                return True
            except ImportError:
                pass
            return False
        except ImportError:
            return False

    def load(self, path: str | Path) -> MSADepth:
        """Load MSA depth from parquet file.

        Args:
            path: Path to .aligned.pqt file from Chai MSA output.

        Returns:
            MSADepth object with per-position depth data.

        Raises:
            ImportError: If pyarrow/fastparquet is not installed.
            FileNotFoundError: If parquet file doesn't exist.
            ValueError: If file format is invalid.
        """
        if not self.is_available():
            raise ImportError(
                "MSA depth loading requires pyarrow or fastparquet. "
                "Install with: pip install pyarrow"
            )

        import pandas as pd

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MSA parquet file not found: {path}")

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise ValueError(f"Failed to load parquet file {path}: {e}")

        # Extract MSA depth per position
        # The parquet contains aligned sequences; depth = number of non-gap chars per column
        if "sequence" in df.columns:
            # Each row is a sequence in the alignment
            sequences = df["sequence"].values
            if len(sequences) > 0:
                # Use maximum sequence length to handle variable-length sequences
                seq_len = max(len(s) for s in sequences)
                depths = np.zeros(seq_len, dtype=int)
                for seq in sequences:
                    for i, char in enumerate(seq):
                        if char != "-" and char != ".":
                            depths[i] += 1
                positions = np.arange(seq_len)
            else:
                positions = np.array([])
                depths = np.array([])
        else:
            # Fallback: assume columns represent positions
            # This handles different parquet formats
            positions = np.arange(len(df))
            depths = np.ones(len(df), dtype=int) * len(df.columns)

        return MSADepth(
            positions=positions,
            depths=depths,
            source_path=path,
        )

    def find_msa_file(self, structure_path: Path) -> Optional[Path]:
        """Try to find MSA parquet file in Chai output directory.

        Args:
            structure_path: Path to structure file (e.g., pred.model_idx_0.cif).

        Returns:
            Path to MSA parquet file if found, None otherwise.
        """
        parent = structure_path.parent
        msas_dir = parent / "msas"

        if msas_dir.exists():
            pqt_files = list(msas_dir.glob("*.aligned.pqt"))
            if len(pqt_files) == 1:
                return pqt_files[0]
            elif len(pqt_files) > 1:
                # Return the first one (could improve matching logic)
                return pqt_files[0]

        return None


@dataclass
class ProteinStructure:
    """Container for protein/nucleic acid structure data with confidence scores."""

    name: str
    ca_coords: np.ndarray  # Backbone coordinates: Cα (protein) or C3' (NA), shape (n_residues, 3)
    cb_coords: np.ndarray  # Sidechain proxy: Cβ/Cα (protein) or C1' (NA), shape (n_residues, 3)
    plddt: np.ndarray  # pLDDT scores per residue, shape (n_residues,)
    residue_ids: list[tuple[str, int]]  # (chain_id, residue_number)
    sequence: str  # One-letter sequence (amino acids or nucleotides)
    source_path: Optional[Path] = None  # Original PDB file path
    biopython_structure: Optional[Structure] = field(default=None, repr=False)
    molecule_type: str = "protein"  # "protein", "dna", "rna", or "mixed"

    @property
    def is_nucleic_acid(self) -> bool:
        """True if this structure contains DNA or RNA."""
        return self.molecule_type in ("dna", "rna")

    @property
    def n_residues(self) -> int:
        """Number of residues in the structure."""
        return len(self.ca_coords)

    @property
    def mean_plddt(self) -> float:
        """Mean pLDDT score across all residues."""
        return float(np.mean(self.plddt))

    @property
    def high_confidence_mask(self) -> np.ndarray:
        """Boolean mask for high-confidence residues (pLDDT >= 70)."""
        return self.plddt >= 70.0

    @property
    def low_confidence_mask(self) -> np.ndarray:
        """Boolean mask for low-confidence residues (pLDDT < 50)."""
        return self.plddt < 50.0


class StructureLoader:
    """Load and parse PDB and mmCIF files with pLDDT extraction."""

    # Standard amino acid 3-letter to 1-letter mapping
    AA_MAP = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        # Non-standard
        "MSE": "M",  # Selenomethionine
        "UNK": "X",  # Unknown
    }

    # Nucleic acid residue name to 1-letter mapping
    NA_MAP = {
        # DNA
        "DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U",
        # RNA
        "A": "A", "C": "C", "G": "G", "U": "U",
        # Modified
        "PSU": "U",  # Pseudouridine
    }

    def __init__(self, quiet: bool = True):
        """Initialize the structure loader.

        Args:
            quiet: Suppress BioPython parser warnings.
        """
        self.pdb_parser = PDBParser(QUIET=quiet)
        self.cif_parser = MMCIFParser(QUIET=quiet)

    def _get_parser(self, path: Path):
        """Get appropriate parser based on file extension.

        Args:
            path: Path to structure file.

        Returns:
            PDBParser or MMCIFParser instance.
        """
        suffix = path.suffix.lower()
        if suffix in (".cif", ".mmcif"):
            return self.cif_parser
        else:
            return self.pdb_parser

    def load(self, path: str | Path) -> ProteinStructure:
        """Load a PDB or mmCIF file and extract structure data.

        Args:
            path: Path to PDB or mmCIF file.

        Returns:
            ProteinStructure with coordinates and confidence scores.

        Raises:
            FileNotFoundError: If structure file doesn't exist.
            ValueError: If no valid residues found.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Structure file not found: {path}")

        parser = self._get_parser(path)
        structure = parser.get_structure(path.stem, str(path))

        ca_coords = []
        cb_coords = []
        plddt_scores = []
        residue_ids = []
        sequence = []
        has_protein = False
        has_na = False
        na_type = None  # "dna" or "rna"

        for model in structure:
            for chain in model:
                for residue in chain:
                    # Skip hetero residues and water
                    if residue.get_id()[0] != " ":
                        continue

                    resname = residue.get_resname().strip()

                    # Try protein first
                    if resname in self.AA_MAP:
                        if "CA" not in residue:
                            continue
                        has_protein = True
                        ca_atom = residue["CA"]
                        ca_coords.append(ca_atom.get_coord())
                        cb_coords.append(
                            residue["CB"].get_coord() if "CB" in residue
                            else ca_atom.get_coord()
                        )
                        plddt_scores.append(ca_atom.get_bfactor())
                        residue_ids.append((chain.get_id(), residue.get_id()[1]))
                        sequence.append(self.AA_MAP[resname])

                    # Try nucleic acid
                    elif resname in self.NA_MAP:
                        if "C3'" not in residue:
                            continue
                        has_na = True
                        if na_type is None:
                            na_type = "dna" if resname.startswith("D") else "rna"
                        backbone = residue["C3'"]
                        ca_coords.append(backbone.get_coord())
                        # Use C1' (glycosidic carbon) as sidechain proxy
                        cb_coords.append(
                            residue["C1'"].get_coord() if "C1'" in residue
                            else backbone.get_coord()
                        )
                        plddt_scores.append(backbone.get_bfactor())
                        residue_ids.append((chain.get_id(), residue.get_id()[1]))
                        sequence.append(self.NA_MAP[resname])

            # Only process first model
            break

        if not ca_coords:
            raise ValueError(f"No valid residues found in {path}")

        # Determine molecule type
        if has_protein and has_na:
            mol_type = "mixed"
        elif has_na:
            mol_type = na_type or "dna"
        else:
            mol_type = "protein"

        return ProteinStructure(
            name=path.stem,
            ca_coords=np.array(ca_coords),
            cb_coords=np.array(cb_coords),
            plddt=np.array(plddt_scores),
            residue_ids=residue_ids,
            sequence="".join(sequence),
            source_path=path,
            biopython_structure=structure,
            molecule_type=mol_type,
        )

    def load_multiple(self, paths: list[str | Path]) -> list[ProteinStructure]:
        """Load multiple PDB files.

        Args:
            paths: List of paths to PDB files.

        Returns:
            List of ProteinStructure objects.
        """
        return [self.load(p) for p in paths]

    @staticmethod
    def extract_plddt(structure: ProteinStructure) -> np.ndarray:
        """Extract pLDDT scores from a structure.

        Args:
            structure: ProteinStructure object.

        Returns:
            Array of pLDDT scores per residue.
        """
        return structure.plddt.copy()

    @staticmethod
    def get_ca_coords(structure: ProteinStructure) -> np.ndarray:
        """Get Cα coordinates from a structure.

        Args:
            structure: ProteinStructure object.

        Returns:
            Array of Cα coordinates, shape (n_residues, 3).
        """
        return structure.ca_coords.copy()

    @staticmethod
    def get_cb_coords(structure: ProteinStructure) -> np.ndarray:
        """Get Cβ coordinates from a structure.

        Args:
            structure: ProteinStructure object.

        Returns:
            Array of Cβ coordinates, shape (n_residues, 3).
        """
        return structure.cb_coords.copy()

    @staticmethod
    def detect_prediction_source(structure: ProteinStructure) -> str:
        """Attempt to detect the source of the predicted structure.

        Heuristics based on pLDDT distribution and structure metadata.

        Args:
            structure: ProteinStructure object.

        Returns:
            One of: "alphafold", "esmfold", "chai", "boltz", "unknown"
        """
        plddt = structure.plddt

        # AlphaFold typically has pLDDT values with specific characteristics
        # ESMFold tends to have slightly different distribution
        # This is a simplified heuristic

        if np.all((plddt >= 0) & (plddt <= 100)):
            # Check for characteristic distributions
            if np.mean(plddt) > 80 and np.std(plddt) < 15:
                return "alphafold"  # High confidence, low variance typical of AF2
            elif np.mean(plddt) > 60:
                return "esmfold"
            else:
                return "unknown"

        return "unknown"

    @staticmethod
    def detect_structure_type(structure: ProteinStructure) -> str:
        """Detect whether structure is predicted or experimental.

        Uses heuristics based on B-factor/pLDDT distribution:
        - Predicted structures (AlphaFold, ESMFold): B-factor contains pLDDT (0-100)
          with values typically clustered in specific ranges
        - Experimental structures: B-factor contains temperature factors,
          often with different distribution patterns

        Args:
            structure: ProteinStructure object.

        Returns:
            "predicted" or "experimental"
        """
        bfactors = structure.plddt  # stored in plddt field regardless of meaning

        # Heuristics to distinguish predicted vs experimental:

        # 1. Check if values are strictly within 0-100 (pLDDT range)
        if np.any(bfactors < 0) or np.any(bfactors > 100):
            return "experimental"  # B-factors can exceed 100

        # 2. Predicted structures typically have pLDDT values with:
        #    - Most values > 50 (confident regions)
        #    - Values often clustered near 70-95
        #    - Relatively discrete-looking distribution
        mean_val = np.mean(bfactors)
        std_val = np.std(bfactors)
        min_val = np.min(bfactors)
        max_val = np.max(bfactors)

        # 3. Experimental B-factors typically:
        #    - Have lower mean (often 15-40)
        #    - Can have very low values near 0
        #    - Often have values < 20 for well-ordered regions

        # If mean is high (>50) and range is reasonable, likely predicted
        if mean_val > 50 and max_val <= 100:
            # Additional check: predicted structures rarely have very low values
            low_value_fraction = np.sum(bfactors < 20) / len(bfactors)
            if low_value_fraction < 0.1:  # Less than 10% below 20
                return "predicted"

        # If mean is low or many low values, likely experimental
        if mean_val < 40:
            return "experimental"

        # Edge cases: check for characteristic predicted structure patterns
        # pLDDT often has values clustered in 70-95 range
        high_confidence_fraction = np.sum((bfactors >= 70) & (bfactors <= 100)) / len(bfactors)
        if high_confidence_fraction > 0.5:
            return "predicted"

        # Default to experimental if uncertain
        return "experimental"


class PAELoader:
    """Load Predicted Aligned Error (PAE) data from AlphaFold output files."""

    @staticmethod
    def load(path: str | Path) -> PAEData:
        """Load PAE data from an AlphaFold JSON file.

        Supports multiple AlphaFold output formats:
        - *_predicted_aligned_error.json (older format)
        - *_scores.json (newer format, includes pTM/ipTM)
        - Full model output JSON

        Args:
            path: Path to PAE JSON file.

        Returns:
            PAEData object with PAE matrix and scores.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is not recognized.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PAE file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        return PAELoader._parse_pae_data(data)

    @staticmethod
    def _parse_pae_data(data: dict | list) -> PAEData:
        """Parse PAE data from various AlphaFold JSON formats.

        Args:
            data: Parsed JSON data.

        Returns:
            PAEData object.
        """
        pae_matrix = None
        max_pae = 31.75  # Default AlphaFold max PAE
        ptm = None
        iptm = None

        # Handle list format (older AlphaFold output)
        if isinstance(data, list):
            # Format: [{"predicted_aligned_error": [[...]], "max_predicted_aligned_error": ...}]
            if len(data) > 0 and "predicted_aligned_error" in data[0]:
                pae_matrix = np.array(data[0]["predicted_aligned_error"])
                max_pae = data[0].get("max_predicted_aligned_error", max_pae)
            else:
                raise ValueError("Unrecognized PAE list format")

        # Handle dict format (newer formats)
        elif isinstance(data, dict):
            # Format 1: {"pae": [[...]], "plddt": [...], "ptm": ..., "iptm": ...}
            if "pae" in data:
                pae_matrix = np.array(data["pae"])
                ptm = data.get("ptm")
                iptm = data.get("iptm")
                max_pae = data.get("max_pae", max_pae)

            # Format 2: {"predicted_aligned_error": [[...]], ...}
            elif "predicted_aligned_error" in data:
                pae_matrix = np.array(data["predicted_aligned_error"])
                max_pae = data.get("max_predicted_aligned_error", max_pae)
                ptm = data.get("ptm")
                iptm = data.get("iptm")

            # Format 3: Nested under model key
            elif any(key.startswith("model_") for key in data.keys()):
                # Take the first model
                for key in data:
                    if key.startswith("model_") and "pae" in data[key]:
                        pae_matrix = np.array(data[key]["pae"])
                        ptm = data[key].get("ptm")
                        iptm = data[key].get("iptm")
                        break

            # Format 4: distance_matrix style (residue1, residue2, distance format)
            elif "residue1" in data and "residue2" in data and "distance" in data:
                # Sparse format used by some tools
                res1 = np.array(data["residue1"])
                res2 = np.array(data["residue2"])
                dist = np.array(data["distance"])
                n = max(max(res1), max(res2))
                pae_matrix = np.zeros((n, n))
                for i, j, d in zip(res1, res2, dist):
                    pae_matrix[i-1, j-1] = d  # Convert to 0-indexed

            else:
                raise ValueError(f"Unrecognized PAE dict format. Keys: {list(data.keys())}")

        else:
            raise ValueError(f"Unexpected PAE data type: {type(data)}")

        if pae_matrix is None:
            raise ValueError("Could not extract PAE matrix from data")

        return PAEData(
            pae_matrix=pae_matrix,
            max_pae=max_pae,
            ptm=ptm,
            iptm=iptm,
        )

    @staticmethod
    def find_pae_file(structure_path: str | Path) -> Optional[Path]:
        """Try to find a PAE file associated with a structure file.

        Searches for common AlphaFold PAE file naming patterns.

        Args:
            structure_path: Path to PDB/CIF structure file.

        Returns:
            Path to PAE file if found, None otherwise.
        """
        structure_path = Path(structure_path)
        base_dir = structure_path.parent
        stem = structure_path.stem

        # Common PAE file patterns
        patterns = [
            f"{stem}_predicted_aligned_error.json",
            f"{stem}_scores.json",
            f"{stem}_pae.json",
            f"{stem}.pae.json",
            # AlphaFold DB patterns
            f"AF-{stem}-F1-predicted_aligned_error_v4.json",
            # ColabFold patterns
            f"{stem}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000_scores.json",
        ]

        # Also try removing common suffixes
        for suffix in ["_relaxed", "_unrelaxed", "_model_1", "_rank_001"]:
            if stem.endswith(suffix):
                base_stem = stem[:-len(suffix)]
                patterns.append(f"{base_stem}_scores.json")
                patterns.append(f"{base_stem}_predicted_aligned_error.json")

        for pattern in patterns:
            pae_path = base_dir / pattern
            if pae_path.exists():
                return pae_path

        # Search for any JSON file with "pae" or "scores" in the name
        for json_file in base_dir.glob("*.json"):
            name_lower = json_file.name.lower()
            if "pae" in name_lower or "scores" in name_lower or "aligned_error" in name_lower:
                # Verify it contains PAE data
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    # Quick check for PAE-like structure
                    if isinstance(data, list) and len(data) > 0:
                        if "predicted_aligned_error" in data[0]:
                            return json_file
                    elif isinstance(data, dict):
                        if "pae" in data or "predicted_aligned_error" in data:
                            return json_file
                except (json.JSONDecodeError, KeyError):
                    continue

        return None
