"""String-level tests for the HTML report table of contents (#80).

Checks that:
- the report carries a TOC whose links all resolve to existing id= anchors,
- conditional sections (PAE, Chai scores, MSA depth, provenance) only get a
  TOC entry when the section itself is rendered,
- the per-residue pLDDT/B-factor profile figure precedes the distribution
  figure in the confidence section.

The tests skip cleanly when the scientific dependencies (numpy, matplotlib,
Biopython, ...) are not installed locally; run them inside the folding
container in that case.
"""

import re
from types import SimpleNamespace

import pytest

sr = pytest.importorskip(
    "protein_compare.visualization.structure_report",
    reason="protein_compare dependencies not installed",
)


def _make_characterizer(has_metadata=False):
    """Build a StructureCharacterizer without running its heavy __init__."""
    c = sr.StructureCharacterizer.__new__(sr.StructureCharacterizer)
    c.structure = SimpleNamespace(
        name="test_model",
        sequence="ACDEFGHIKLMNPQRSTVWY",
        is_nucleic_acid=False,
        plddt=None,
    )
    c.is_predicted = True
    c.contact_cutoff = 8.0
    # has_chai_scores / has_msa_depth / has_metadata are properties derived
    # from these attributes.
    c.chai_scores = None
    c.msa_depth = None
    c.metadata = {"tool": "boltz", "status": "success"} if has_metadata else None
    return c


def _images(**extra):
    imgs = {
        k: "AAAA"
        for k in (
            "aa_composition",
            "plddt_distribution",
            "plddt_profile",
            "contact_map",
            "contact_order",
            "residue_contacts",
            "ss_composition",
            "ss_profile",
        )
    }
    imgs.update(extra)
    return imgs


def _build_html(characterizer, pae_analysis=None, images_b64=None):
    seq_comp = SimpleNamespace(
        length=20,
        molecular_weight=2200.0,
        type_fractions={"hydrophobic": 0.4, "polar": 0.3, "positive": 0.15, "negative": 0.15},
    )
    conf_stats = SimpleNamespace(
        mean=85.0, median=88.0, n_very_high=10, n_very_low=1, frac_confident=0.9
    )
    contact_analysis = SimpleNamespace(
        n_contacts=30, contact_density=0.1, n_long_range=5, n_very_long_range=2
    )
    ss_analysis = SimpleNamespace(
        helix_fraction=0.4, sheet_fraction=0.2, coil_fraction=0.4,
        helix_count=8, sheet_count=4, coil_count=8,
    )
    return characterizer._build_html(
        seq_comp,
        conf_stats,
        contact_analysis,
        ss_analysis,
        images_b64 or _images(),
        structure_content="",
        pae_analysis=pae_analysis,
    )


def _make_pae_analysis():
    return SimpleNamespace(
        mean_pae=2.0,
        median_pae=1.5,
        n_domains=1,
        intra_domain_pae=1.8,
        inter_domain_pae=None,
        pae_data=SimpleNamespace(ptm=0.9, iptm=None),
    )


def _toc_hrefs(html):
    toc = re.search(r'<nav class="toc".*?</nav>', html, re.S)
    assert toc is not None, "report has no TOC <nav>"
    return re.findall(r'href="#([^"]+)"', toc.group(0))


def test_toc_present_and_anchors_resolve():
    html = _build_html(_make_characterizer())
    hrefs = _toc_hrefs(html)
    ids = set(re.findall(r'id="([^"]+)"', html))
    assert hrefs, "TOC has no entries"
    missing = [h for h in hrefs if h not in ids]
    assert not missing, f"TOC links point to missing anchors: {missing}"
    # Core sections always present
    for anchor in ("structure-viewer", "summary", "sequence", "confidence",
                   "contacts", "secondary", "glossary"):
        assert anchor in hrefs


def test_toc_omits_absent_conditional_sections():
    html = _build_html(_make_characterizer())
    hrefs = _toc_hrefs(html)
    for anchor in ("pae", "chai-scores", "msa-depth", "provenance"):
        assert anchor not in hrefs, f"TOC lists absent section '{anchor}'"
        assert f'id="{anchor}"' not in html


def test_toc_includes_pae_when_present():
    html = _build_html(
        _make_characterizer(),
        pae_analysis=_make_pae_analysis(),
        images_b64=_images(pae_heatmap="AAAA"),
    )
    hrefs = _toc_hrefs(html)
    assert "pae" in hrefs
    assert 'id="pae"' in html


def test_toc_includes_provenance_when_present():
    html = _build_html(_make_characterizer(has_metadata=True))
    hrefs = _toc_hrefs(html)
    assert "provenance" in hrefs
    assert 'id="provenance"' in html


def test_profile_figure_precedes_distribution():
    html = _build_html(_make_characterizer())
    confidence = re.search(
        r'id="confidence".*?<div class="section" id="contacts"', html, re.S
    )
    assert confidence is not None
    section = confidence.group(0)
    profile = section.find("Per-residue pLDDT profile")
    distribution = section.find("Distribution of pLDDT confidence scores")
    assert profile != -1 and distribution != -1
    assert profile < distribution, (
        "per-residue profile figure must come before the distribution figure"
    )
