"""Tests for supplying a precomputed secondary structure (--ss-file, #76).

The tests skip cleanly when the scientific dependencies are not installed
locally; run them inside the folding container in that case.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

sec = pytest.importorskip(
    "protein_compare.core.secondary",
    reason="protein_compare dependencies not installed",
)

# Regex-safe: no metacharacters ("-" is literal outside a character class).
_SS_FILE_IGNORED = (
    "Ignoring --ss-file and computing secondary structure from the structure instead"
)


def _structure(n, source_path=None):
    return SimpleNamespace(
        n_residues=n,
        biopython_structure=[object()],
        source_path=source_path,
        residue_ids=[("A", i) for i in range(1, n + 1)],
    )


def _dssp_text(codes):
    lines = ["  #  RESIDUE AA STRUCTURE BP1 BP2  ACC\n"]
    for i, ss in enumerate(codes, 1):
        if ss is None:  # chain-break row: "!" in the AA column
            lines.append(f"{i:5d}" + " " * 8 + "!" + " " * 20 + "\n")
        else:
            lines.append(f"{i:5d}{i:5d} A A  {ss}    0   0\n")
    return "".join(lines)


@pytest.fixture
def dssp_mock(monkeypatch):
    """Replace the DSSP runner with a mock that reports a missing binary."""
    m = mock.Mock(side_effect=FileNotFoundError("mkdssp"))
    monkeypatch.setattr(sec, "HAS_DSSP", True)
    monkeypatch.setattr(sec, "DSSP", m, raising=False)
    return m


class TestAssignSsFile:
    def test_three_state_file_is_used_without_running_dssp(self, tmp_path, dssp_mock):
        f = tmp_path / "model_1.ss3.txt"
        f.write_text("HHHEECCC\n")
        analyzer = sec.SecondaryStructureAnalyzer()

        ss = analyzer.assign_ss(_structure(8, tmp_path / "m.pdb"), ss_file=f)

        assert ss == list("HHHEECCC")
        dssp_mock.assert_not_called()

    def test_dssp_file_is_used_and_simplified(self, tmp_path, dssp_mock):
        f = tmp_path / "model_1.dssp"
        f.write_text(_dssp_text(["H", "G", "E", None, "B", "T", "S", " "]))
        analyzer = sec.SecondaryStructureAnalyzer()
        structure = _structure(7, tmp_path / "m.pdb")

        assert analyzer.assign_ss(structure, ss_file=f) == list("HHEECCC")
        assert analyzer.assign_ss(structure, simplify=False, ss_file=f) == list("HGEBTS ")
        dssp_mock.assert_not_called()

    def test_length_mismatch_warns_and_runs_dssp(self, tmp_path, dssp_mock, monkeypatch):
        f = tmp_path / "model_1.ss3.txt"
        f.write_text("HHH\n")
        analyzer = sec.SecondaryStructureAnalyzer()
        monkeypatch.setattr(
            analyzer, "_fallback_ss_assignment", lambda s, simplify=True: ["F"] * s.n_residues
        )

        with pytest.warns(UserWarning, match=_SS_FILE_IGNORED):
            ss = analyzer.assign_ss(_structure(8, tmp_path / "m.pdb"), ss_file=f)

        dssp_mock.assert_called_once()
        assert ss == ["F"] * 8

    def test_invalid_file_warns_and_runs_dssp(self, tmp_path, dssp_mock, monkeypatch):
        f = tmp_path / "bad.txt"
        f.write_text("not a secondary structure")
        analyzer = sec.SecondaryStructureAnalyzer()
        monkeypatch.setattr(
            analyzer, "_fallback_ss_assignment", lambda s, simplify=True: ["F"] * s.n_residues
        )

        with pytest.warns(UserWarning, match=_SS_FILE_IGNORED):
            analyzer.assign_ss(_structure(8, tmp_path / "m.pdb"), ss_file=f)

        dssp_mock.assert_called_once()

    def test_unreadable_file_warns_and_runs_dssp(self, tmp_path, dssp_mock, monkeypatch):
        analyzer = sec.SecondaryStructureAnalyzer()
        monkeypatch.setattr(
            analyzer, "_fallback_ss_assignment", lambda s, simplify=True: ["F"] * s.n_residues
        )

        with pytest.warns(UserWarning, match=_SS_FILE_IGNORED):
            analyzer.assign_ss(
                _structure(8, tmp_path / "m.pdb"), ss_file=tmp_path / "missing.txt"
            )

        dssp_mock.assert_called_once()


def test_characterizer_forwards_ss_path_to_assign_ss(tmp_path):
    report_mod = pytest.importorskip(
        "protein_compare.visualization.structure_report",
        reason="protein_compare dependencies not installed",
    )
    ss_file = tmp_path / "model_1.ss3.txt"
    ss_file.write_text("HHEC\n")
    structure = SimpleNamespace(n_residues=4, is_nucleic_acid=False, source_path=None)
    # "experimental" skips the PAE/Chai/MSA/metadata auto-discovery.
    characterizer = report_mod.StructureCharacterizer(
        structure, structure_type="experimental", ss_path=str(ss_file)
    )
    characterizer.ss_analyzer = mock.Mock()
    characterizer.ss_analyzer.assign_ss.return_value = list("HHEC")

    characterizer.analyze_secondary_structure()

    characterizer.ss_analyzer.assign_ss.assert_called_once_with(
        structure, simplify=True, ss_file=str(ss_file)
    )


def test_characterize_cli_accepts_ss_file():
    cli_mod = pytest.importorskip(
        "protein_compare.cli", reason="protein_compare dependencies not installed"
    )
    from click.testing import CliRunner

    result = CliRunner().invoke(cli_mod.cli, ["characterize", "--help"])

    assert result.exit_code == 0
    assert "--ss-file" in result.output


def test_characterize_cli_passes_ss_file_to_characterizer(tmp_path):
    cli_mod = pytest.importorskip(
        "protein_compare.cli", reason="protein_compare dependencies not installed"
    )
    from click.testing import CliRunner

    class _Stop(Exception):
        pass

    seen = {}

    def _characterizer(**kwargs):
        seen.update(kwargs)
        raise _Stop

    pdb = tmp_path / "model_1.pdb"
    pdb.write_text("END\n")
    ss_file = tmp_path / "model_1.ss3.txt"
    ss_file.write_text("HHHEECCC\n")

    with mock.patch.object(
        cli_mod, "StructureLoader",
        return_value=SimpleNamespace(load=lambda path: SimpleNamespace()),
    ), mock.patch.object(cli_mod, "StructureCharacterizer", side_effect=_characterizer):
        result = CliRunner().invoke(
            cli_mod.cli, ["characterize", str(pdb), "--ss-file", str(ss_file)]
        )

    assert isinstance(result.exception, _Stop), result.output
    assert seen["ss_path"] == str(ss_file)
