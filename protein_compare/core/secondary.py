"""Secondary structure assignment and comparison using DSSP.

Provides DSSP integration for secondary structure assignment
and comparison between protein structures.
"""

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Optional
import warnings

import numpy as np

try:
    from Bio.PDB.DSSP import DSSP, dssp_dict_from_pdb_file
    HAS_DSSP = True
except ImportError:
    HAS_DSSP = False

from protein_compare.io.parser import ProteinStructure


# DSSP secondary structure codes
DSSP_CODES = {
    "H": "alpha-helix",
    "G": "3-10 helix",
    "I": "pi-helix",
    "E": "beta-strand",
    "B": "beta-bridge",
    "T": "turn",
    "S": "bend",
    "-": "coil",
    " ": "coil",
    "C": "coil",
}

# 3-state simplification mapping
THREE_STATE_MAP = {
    "H": "H",  # Helix
    "G": "H",  # 3-10 helix -> Helix
    "I": "H",  # Pi helix -> Helix
    "E": "E",  # Strand
    "B": "E",  # Bridge -> Strand
    "T": "C",  # Turn -> Coil
    "S": "C",  # Bend -> Coil
    "-": "C",  # Coil
    " ": "C",  # Unknown -> Coil
    "C": "C",  # Coil
}


@dataclass
class SSComparisonResult:
    """Result of secondary structure comparison."""

    agreement_score: float  # Overall agreement (0-1)
    helix_agreement: float  # Agreement in helical regions
    sheet_agreement: float  # Agreement in sheet regions
    coil_agreement: float  # Agreement in coil regions
    transitions: list[int]  # Residue positions where SS differs
    ss1: list[str]  # SS assignment for structure 1
    ss2: list[str]  # SS assignment for structure 2
    ss1_3state: list[str]  # 3-state SS for structure 1
    ss2_3state: list[str]  # 3-state SS for structure 2
    confusion_matrix: dict  # Confusion matrix of SS types

    @property
    def n_transitions(self) -> int:
        """Number of residues with different SS assignments."""
        return len(self.transitions)

    @property
    def helix_fraction_1(self) -> float:
        """Fraction of helical residues in structure 1."""
        return self.ss1_3state.count("H") / len(self.ss1_3state) if self.ss1_3state else 0

    @property
    def helix_fraction_2(self) -> float:
        """Fraction of helical residues in structure 2."""
        return self.ss2_3state.count("H") / len(self.ss2_3state) if self.ss2_3state else 0

    @property
    def sheet_fraction_1(self) -> float:
        """Fraction of sheet residues in structure 1."""
        return self.ss1_3state.count("E") / len(self.ss1_3state) if self.ss1_3state else 0

    @property
    def sheet_fraction_2(self) -> float:
        """Fraction of sheet residues in structure 2."""
        return self.ss2_3state.count("E") / len(self.ss2_3state) if self.ss2_3state else 0


class SecondaryStructureAnalyzer:
    """Analyze and compare secondary structures using DSSP."""

    def __init__(self, dssp_path: Optional[str] = None):
        """Initialize the analyzer.

        Args:
            dssp_path: Path to DSSP executable. If None, uses system default.
        """
        self.dssp_path = dssp_path or "mkdssp"

    def assign_ss(
        self,
        structure: ProteinStructure,
        simplify: bool = True,
        ss_file: Optional[str | Path] = None,
    ) -> list[str]:
        """Assign secondary structure using DSSP.

        Args:
            structure: ProteinStructure with BioPython structure.
            simplify: If True, return 3-state (H, E, C) instead of 8-state.
            ss_file: Optional precomputed assignment: a 3-state H/E/C string
                file or a classic DSSP file. Used instead of running DSSP when
                its residue count matches the structure; otherwise a warning
                is issued, the file is ignored and secondary structure is
                computed from the structure (DSSP or the Cα fallback) as
                usual. A 3-state file carries no
                8-state detail, so ``simplify=False`` returns its H/E/C codes.

        Returns:
            List of SS codes per residue.

        Raises:
            RuntimeError: If DSSP fails.
        """
        if ss_file is not None:
            supplied = self._read_ss_file(ss_file, structure.n_residues)
            if supplied is not None:
                if simplify:
                    supplied = [THREE_STATE_MAP.get(s, "C") for s in supplied]
                return supplied

        if not HAS_DSSP:
            raise ImportError("BioPython DSSP module not available")

        if structure.biopython_structure is None:
            raise ValueError("Structure must have BioPython structure attached")

        # Run DSSP
        try:
            model = structure.biopython_structure[0]
            # DSSP requires a file path, not a Structure object
            if structure.source_path is not None:
                dssp = DSSP(model, str(structure.source_path), dssp=self.dssp_path, file_type="PDB")
            else:
                raise ValueError("Structure must have source_path for DSSP")
        except FileNotFoundError:
            warnings.warn(f"DSSP binary not found ({self.dssp_path}). Using fallback assignment.")
            return self._fallback_ss_assignment(structure, simplify)
        except Exception as e:
            warnings.warn(f"DSSP failed: {e}. Using fallback assignment.")
            return self._fallback_ss_assignment(structure, simplify)

        # Extract SS per residue
        ss_list = []
        residue_set = {(rid[0], rid[1]) for rid in structure.residue_ids}

        for key in dssp.keys():
            chain_id = key[0]
            res_id = key[1][1]

            if (chain_id, res_id) in residue_set:
                ss = dssp[key][2]
                if simplify:
                    ss = THREE_STATE_MAP.get(ss, "C")
                ss_list.append(ss)

        # If DSSP returned fewer residues, pad with coil
        while len(ss_list) < structure.n_residues:
            ss_list.append("C" if simplify else "-")

        return ss_list[:structure.n_residues]

    @staticmethod
    def _read_ss_file(path: str | Path, n_residues: int) -> Optional[list[str]]:
        """Read a supplied SS file; return None (with a warning) if unusable."""
        try:
            text = Path(path).read_text()
        except (OSError, UnicodeDecodeError) as e:
            warnings.warn(
                f"Cannot read SS file {path}: {e}. "
                "Ignoring --ss-file and computing secondary structure from the structure instead."
            )
            return None

        if "  #  RESIDUE" in text:
            ss: list[str] = []
            in_table = False
            for line in text.splitlines():
                if line.startswith("  #  RESIDUE"):
                    in_table = True
                elif in_table and len(line) > 16 and line[13] != "!":  # "!" = chain break
                    ss.append(line[16])
        else:
            ss = list("".join(text.split()))
            if set(ss) - set("HEC"):
                warnings.warn(
                    f"SS file {path} is not a 3-state H/E/C string. "
                    "Ignoring --ss-file and computing secondary structure from the structure instead."
                )
                return None

        if len(ss) != n_residues:
            warnings.warn(
                f"SS file {path} has {len(ss)} residues but the structure has "
                f"{n_residues}. "
                "Ignoring --ss-file and computing secondary structure from the structure instead."
            )
            return None
        return ss

    def _fallback_ss_assignment(
        self,
        structure: ProteinStructure,
        simplify: bool = True,
    ) -> list[str]:
        """Fallback SS assignment based on Cα geometry.

        Simple heuristic when DSSP is unavailable:
        - Helix: local curvature and regular spacing
        - Sheet: extended conformation
        - Coil: everything else

        Args:
            structure: ProteinStructure object.
            simplify: Return 3-state.

        Returns:
            List of SS codes per residue.
        """
        coords = structure.ca_coords
        n = len(coords)
        ss = ["C"] * n

        if n < 5:
            return ss

        # Calculate local geometry
        for i in range(2, n - 2):
            # Virtual bond angles
            v1 = coords[i] - coords[i - 2]
            v2 = coords[i + 2] - coords[i]

            # Distance between i-2 and i+2
            dist = np.linalg.norm(coords[i + 2] - coords[i - 2])

            # Angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

            # Helix: ~5.4Å rise per turn, specific angle
            if 5.0 < dist < 6.5 and cos_angle > 0.5:
                ss[i] = "H"
            # Sheet: extended, larger distance
            elif dist > 10.0 and cos_angle < -0.3:
                ss[i] = "E"

        return ss

    def assign_ss_from_pdb(
        self,
        pdb_path: str | Path,
        simplify: bool = True,
    ) -> list[str]:
        """Assign secondary structure directly from PDB file.

        Args:
            pdb_path: Path to PDB file.
            simplify: Return 3-state.

        Returns:
            List of SS codes per residue.
        """
        if not HAS_DSSP:
            raise ImportError("BioPython DSSP module not available")

        try:
            dssp_dict, keys = dssp_dict_from_pdb_file(str(pdb_path))
        except Exception as e:
            raise RuntimeError(f"DSSP failed on {pdb_path}: {e}")

        ss_list = []
        for key in keys:
            ss = dssp_dict[key][1]
            if simplify:
                ss = THREE_STATE_MAP.get(ss, "C")
            ss_list.append(ss)

        return ss_list

    def compare_ss(
        self,
        ss1: list[str],
        ss2: list[str],
        residue_mapping: Optional[list[tuple[int, int]]] = None,
    ) -> SSComparisonResult:
        """Compare secondary structure assignments.

        Args:
            ss1: SS assignment for structure 1.
            ss2: SS assignment for structure 2.
            residue_mapping: Optional mapping between residues.
                If None, assumes 1:1 correspondence.

        Returns:
            SSComparisonResult with comparison metrics.
        """
        if residue_mapping:
            # Use mapping to align SS
            aligned_ss1 = [ss1[i] for i, _ in residue_mapping]
            aligned_ss2 = [ss2[j] for _, j in residue_mapping]
        else:
            # Assume 1:1 correspondence
            min_len = min(len(ss1), len(ss2))
            aligned_ss1 = ss1[:min_len]
            aligned_ss2 = ss2[:min_len]

        # Convert to 3-state for comparison
        ss1_3state = [THREE_STATE_MAP.get(s, "C") for s in aligned_ss1]
        ss2_3state = [THREE_STATE_MAP.get(s, "C") for s in aligned_ss2]

        # Find transitions (differences)
        transitions = [
            i for i, (s1, s2) in enumerate(zip(ss1_3state, ss2_3state))
            if s1 != s2
        ]

        # Overall agreement
        n_residues = len(ss1_3state)
        agreement = 1.0 - len(transitions) / n_residues if n_residues > 0 else 0.0

        # Per-type agreement
        helix_agreement = self._type_agreement(ss1_3state, ss2_3state, "H")
        sheet_agreement = self._type_agreement(ss1_3state, ss2_3state, "E")
        coil_agreement = self._type_agreement(ss1_3state, ss2_3state, "C")

        # Confusion matrix
        confusion = self._build_confusion_matrix(ss1_3state, ss2_3state)

        return SSComparisonResult(
            agreement_score=agreement,
            helix_agreement=helix_agreement,
            sheet_agreement=sheet_agreement,
            coil_agreement=coil_agreement,
            transitions=transitions,
            ss1=aligned_ss1,
            ss2=aligned_ss2,
            ss1_3state=ss1_3state,
            ss2_3state=ss2_3state,
            confusion_matrix=confusion,
        )

    @staticmethod
    def _type_agreement(
        ss1: list[str],
        ss2: list[str],
        ss_type: str,
    ) -> float:
        """Calculate agreement for specific SS type.

        Args:
            ss1: First SS list.
            ss2: Second SS list.
            ss_type: SS type to check (H, E, or C).

        Returns:
            Agreement fraction (0-1) for this type.
        """
        # Find positions where either structure has this type
        union_count = sum(1 for s1, s2 in zip(ss1, ss2) if s1 == ss_type or s2 == ss_type)

        if union_count == 0:
            return 1.0  # No residues of this type in either structure

        # Count matches
        match_count = sum(1 for s1, s2 in zip(ss1, ss2) if s1 == ss_type and s2 == ss_type)

        return match_count / union_count

    @staticmethod
    def _build_confusion_matrix(
        ss1: list[str],
        ss2: list[str],
    ) -> dict:
        """Build confusion matrix for SS types.

        Args:
            ss1: First SS list (reference).
            ss2: Second SS list (prediction).

        Returns:
            Dict with confusion matrix.
        """
        types = ["H", "E", "C"]
        matrix = {t1: {t2: 0 for t2 in types} for t1 in types}

        for s1, s2 in zip(ss1, ss2):
            if s1 in types and s2 in types:
                matrix[s1][s2] += 1

        return matrix

    def ss_agreement_score(
        self,
        ss1: list[str],
        ss2: list[str],
    ) -> float:
        """Calculate simple SS agreement score.

        Args:
            ss1: First SS list.
            ss2: Second SS list.

        Returns:
            Fraction of matching SS assignments.
        """
        result = self.compare_ss(ss1, ss2)
        return result.agreement_score

    @staticmethod
    def ss_content(ss: list[str]) -> dict[str, float]:
        """Calculate SS content fractions.

        Args:
            ss: List of SS assignments.

        Returns:
            Dict with fractions for H, E, C.
        """
        n = len(ss)
        if n == 0:
            return {"H": 0.0, "E": 0.0, "C": 0.0}

        # Convert to 3-state
        ss_3state = [THREE_STATE_MAP.get(s, "C") for s in ss]

        return {
            "H": ss_3state.count("H") / n,
            "E": ss_3state.count("E") / n,
            "C": ss_3state.count("C") / n,
        }
