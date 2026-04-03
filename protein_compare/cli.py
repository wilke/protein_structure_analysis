"""Command-line interface for protein structure comparison.

Provides CLI commands for batch comparison, visualization,
and report generation.
"""

import sys
from pathlib import Path
from typing import Optional

import click

from protein_compare.io.parser import StructureLoader
from protein_compare.io.reporter import ComparisonReporter
from protein_compare.core.batch import BatchComparator
from protein_compare.core.alignment import StructuralAligner
from protein_compare.core.contacts import ContactMapAnalyzer
from protein_compare.core.secondary import SecondaryStructureAnalyzer
from protein_compare.visualization.alignment_viz import AlignmentVisualizer
from protein_compare.visualization.contact_maps import ContactMapVisualizer
from protein_compare.visualization.divergence import DivergenceAnalyzer, DivergenceVisualizer
from protein_compare.visualization.structure_report import StructureCharacterizer


@click.group()
@click.version_option(version="0.1.0", prog_name="protein_compare")
def cli():
    """Protein structure comparison pipeline.

    Compare protein structures from AlphaFold, ESMFold, Chai, Boltz,
    and other prediction tools with pLDDT confidence integration.
    """
    pass


@cli.command()
@click.argument("structure1", type=click.Path(exists=True))
@click.argument("structure2", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file for results (JSON)")
@click.option("--pymol", "-p", type=click.Path(), help="Generate PyMOL visualization script")
@click.option("--plot", type=click.Path(), help="Save divergence plot to file")
@click.option("--confidence-weighted/--no-confidence-weighted", default=True,
              help="Use pLDDT-weighted RMSD")
@click.option("--contact-cutoff", default=8.0, help="Contact map distance cutoff (Å)")
def compare(structure1, structure2, output, pymol, plot, confidence_weighted, contact_cutoff):
    """Compare two protein structures.

    Calculates TM-score, RMSD, secondary structure agreement,
    and contact map similarity.
    """
    click.echo(f"Loading structures...")

    loader = StructureLoader()
    try:
        struct1 = loader.load(structure1)
        struct2 = loader.load(structure2)
    except Exception as e:
        click.echo(f"Error loading structures: {e}", err=True)
        sys.exit(1)

    click.echo(f"  {struct1.name}: {struct1.n_residues} residues, mean pLDDT: {struct1.mean_plddt:.1f}")
    click.echo(f"  {struct2.name}: {struct2.n_residues} residues, mean pLDDT: {struct2.mean_plddt:.1f}")

    # Perform comparison
    click.echo("\nAligning structures...")
    comparator = BatchComparator(
        contact_cutoff=contact_cutoff,
        confidence_weighted=confidence_weighted,
    )

    try:
        result = comparator.compare_pair(struct1, struct2, store_alignment=True)
    except Exception as e:
        click.echo(f"Error during comparison: {e}", err=True)
        sys.exit(1)

    # Display results
    click.echo("\n" + "=" * 50)
    click.echo("COMPARISON RESULTS")
    click.echo("=" * 50)
    click.echo(f"\nStructural Alignment:")
    click.echo(f"  TM-score:       {result.tm_score:.4f}")
    click.echo(f"  RMSD:           {result.rmsd:.2f} Å")
    if confidence_weighted:
        click.echo(f"  Weighted RMSD:  {result.weighted_rmsd:.2f} Å")
    click.echo(f"  Aligned length: {result.aligned_length} residues")
    click.echo(f"  Seq identity:   {result.seq_identity:.1%}")

    click.echo(f"\nGlobal Distance Test:")
    click.echo(f"  GDT-TS: {result.gdt_ts:.3f}")
    click.echo(f"  GDT-HA: {result.gdt_ha:.3f}")

    click.echo(f"\nSecondary Structure:")
    click.echo(f"  SS Agreement: {result.ss_agreement:.1%}")

    click.echo(f"\nContact Maps:")
    click.echo(f"  Contact Jaccard: {result.contact_jaccard:.3f}")

    click.echo(f"\nDivergence:")
    click.echo(f"  Divergent residues (>3Å): {result.n_divergent_residues}")

    # Fold classification
    if result.tm_score >= 0.5:
        fold_class = "SAME FOLD"
    elif result.tm_score >= 0.4:
        fold_class = "SIMILAR FOLD"
    else:
        fold_class = "DIFFERENT FOLD"
    click.echo(f"\n  Classification: {fold_class}")
    click.echo("=" * 50)

    # Save results
    if output:
        import json
        with open(output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        click.echo(f"\nResults saved to: {output}")

    # Generate PyMOL script
    if pymol and result.alignment:
        viz = AlignmentVisualizer()

        # Generate aligned PDB path (same directory as output, with _aligned suffix)
        pymol_path = Path(pymol)
        aligned_pdb_path = pymol_path.parent / f"{struct2.name}_aligned.pdb"

        viz.generate_pymol_script(
            struct1, struct2, result.alignment,
            color_by="rmsd",
            output_path=pymol,
            aligned_pdb_path=aligned_pdb_path,
        )
        click.echo(f"PyMOL script saved to: {pymol}")
        click.echo(f"Aligned structure saved to: {aligned_pdb_path}")

    # Generate plot
    if plot and result.alignment:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        viz = AlignmentVisualizer()
        fig = viz.divergence_plot(
            result.alignment,
            struct1.plddt,
            struct2.plddt,
        )
        fig.savefig(plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        click.echo(f"Divergence plot saved to: {plot}")


@cli.command()
@click.argument("structures", nargs=-1, type=click.Path(exists=True))
@click.option("--reference", "-r", type=click.Path(exists=True),
              help="Reference structure for all-vs-reference comparison")
@click.option("--output", "-o", default="results.csv",
              help="Output CSV file for results")
@click.option("--json", "json_output", type=click.Path(),
              help="Also save results as JSON")
@click.option("--html", "html_output", type=click.Path(),
              help="Generate HTML report")
@click.option("--contact-cutoff", default=8.0, help="Contact map distance cutoff (Å)")
@click.option("--confidence-weighted/--no-confidence-weighted", default=True,
              help="Use pLDDT-weighted RMSD")
@click.option("--no-secondary", is_flag=True, help="Skip secondary structure comparison")
@click.option("--no-contacts", is_flag=True, help="Skip contact map comparison")
@click.option("--jobs", "-j", default=-1, help="Number of parallel jobs (-1 for all CPUs)")
def batch(structures, reference, output, json_output, html_output,
          contact_cutoff, confidence_weighted, no_secondary, no_contacts, jobs):
    """Compare multiple structures in batch mode.

    Without --reference: performs all pairwise comparisons.
    With --reference: compares all structures to the reference.

    Examples:

        protein_compare batch *.pdb -o results.csv

        protein_compare batch *.pdb --reference ref.pdb -o results.csv
    """
    if len(structures) < 2 and reference is None:
        click.echo("Error: Need at least 2 structures for comparison", err=True)
        sys.exit(1)

    if len(structures) < 1:
        click.echo("Error: No structures provided", err=True)
        sys.exit(1)

    click.echo(f"Loading {len(structures)} structures...")

    loader = StructureLoader()
    loaded_structures = []

    with click.progressbar(structures, label="Loading") as bar:
        for path in bar:
            try:
                struct = loader.load(path)
                loaded_structures.append(struct)
            except Exception as e:
                click.echo(f"\nWarning: Failed to load {path}: {e}", err=True)

    if len(loaded_structures) < 1:
        click.echo("Error: No structures loaded successfully", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(loaded_structures)} structures successfully")

    # Load reference if provided
    ref_struct = None
    if reference:
        try:
            ref_struct = loader.load(reference)
            click.echo(f"Reference: {ref_struct.name} ({ref_struct.n_residues} residues)")
        except Exception as e:
            click.echo(f"Error loading reference: {e}", err=True)
            sys.exit(1)

    # Initialize comparator
    comparator = BatchComparator(
        structures=loaded_structures,
        reference=ref_struct,
        contact_cutoff=contact_cutoff,
        confidence_weighted=confidence_weighted,
        compute_ss=not no_secondary,
        compute_contacts=not no_contacts,
    )

    # Perform comparisons
    if ref_struct:
        n_comparisons = len(loaded_structures)
        click.echo(f"\nComparing {n_comparisons} structures to reference...")
        results = comparator.compare_to_reference(n_jobs=jobs)
    else:
        n_comparisons = len(loaded_structures) * (len(loaded_structures) - 1) // 2
        click.echo(f"\nPerforming {n_comparisons} pairwise comparisons...")
        results = comparator.compare_all_pairs(n_jobs=jobs)

    # Generate reports
    reporter = ComparisonReporter(results)

    # Save CSV
    reporter.to_csv(output)
    click.echo(f"\nResults saved to: {output}")

    # Save JSON if requested
    if json_output:
        reporter.to_json(json_output)
        click.echo(f"JSON saved to: {json_output}")

    # Generate HTML if requested
    if html_output:
        reporter.generate_html_report(html_output)
        click.echo(f"HTML report saved to: {html_output}")

    # Print summary
    click.echo("\n" + reporter.summary_report())


@cli.command()
@click.argument("structure1", type=click.Path(exists=True))
@click.argument("structure2", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output file (PyMOL .pml, or image .png/.pdf)")
@click.option("--color-by", type=click.Choice(["rmsd", "plddt", "chain"]),
              default="rmsd", help="Coloring scheme")
@click.option("--format", "fmt", type=click.Choice(["pymol", "plot"]),
              default="pymol", help="Output format")
def visualize(structure1, structure2, output, color_by, fmt):
    """Generate visualization of aligned structures.

    Creates PyMOL scripts or matplotlib plots showing
    structural alignment and divergence.
    """
    loader = StructureLoader()
    struct1 = loader.load(structure1)
    struct2 = loader.load(structure2)

    aligner = StructuralAligner()
    alignment = aligner.align(struct1, struct2)

    if fmt == "pymol":
        viz = AlignmentVisualizer()

        # Generate aligned PDB path (same directory as output, with _aligned suffix)
        output_path = Path(output)
        aligned_pdb_path = output_path.parent / f"{struct2.name}_aligned.pdb"

        script = viz.generate_pymol_script(
            struct1, struct2, alignment,
            color_by=color_by,
            output_path=output,
            aligned_pdb_path=aligned_pdb_path,
        )
        click.echo(f"PyMOL script saved to: {output}")
        click.echo(f"Aligned structure saved to: {aligned_pdb_path}")
        click.echo("\nTo visualize, run:")
        click.echo(f"  pymol {output}")

    else:  # plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        viz = AlignmentVisualizer()
        fig = viz.divergence_plot(
            alignment,
            struct1.plddt,
            struct2.plddt,
        )
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        click.echo(f"Plot saved to: {output}")


@cli.command()
@click.argument("structure", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file for contact map image")
@click.option("--cutoff", default=8.0, help="Contact distance cutoff (Å)")
@click.option("--show", is_flag=True, help="Display plot interactively")
def contacts(structure, output, cutoff, show):
    """Generate and visualize contact map for a structure."""
    loader = StructureLoader()
    struct = loader.load(structure)

    analyzer = ContactMapAnalyzer(cutoff=cutoff)
    contact_map = analyzer.compute_contact_map(struct)

    n_contacts = analyzer.long_range_contacts(contact_map, min_sep=12)
    density = analyzer.contact_density(contact_map)

    click.echo(f"Structure: {struct.name}")
    click.echo(f"Residues: {struct.n_residues}")
    click.echo(f"Long-range contacts (>12 residues): {n_contacts}")
    click.echo(f"Contact density: {density:.3f}")

    if output or show:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        viz = ContactMapVisualizer()
        fig = viz.plot_single_map(contact_map, title=f"Contact Map: {struct.name}")

        if output:
            fig.savefig(output, dpi=150, bbox_inches="tight")
            click.echo(f"Contact map saved to: {output}")

        if show:
            plt.show()
        else:
            plt.close(fig)


@cli.command()
@click.argument("structure", type=click.Path(exists=True))
def info(structure):
    """Display information about a protein structure."""
    loader = StructureLoader()
    struct = loader.load(structure)

    # Detect structure type
    structure_type = loader.detect_structure_type(struct)
    is_predicted = structure_type == "predicted"

    click.echo(f"\nStructure: {struct.name}")
    click.echo(f"File: {structure}")
    click.echo(f"\nBasic Info:")
    click.echo(f"  Residues: {struct.n_residues}")
    click.echo(f"  Sequence: {struct.sequence[:50]}..." if len(struct.sequence) > 50 else f"  Sequence: {struct.sequence}")

    if is_predicted:
        click.echo(f"\nConfidence (pLDDT):")
        click.echo(f"  Mean:   {struct.mean_plddt:.1f}")
        click.echo(f"  Min:    {struct.plddt.min():.1f}")
        click.echo(f"  Max:    {struct.plddt.max():.1f}")

        n_high = sum(struct.high_confidence_mask)
        n_low = sum(struct.low_confidence_mask)
        click.echo(f"  High confidence (≥70): {n_high} ({100*n_high/struct.n_residues:.1f}%)")
        click.echo(f"  Low confidence (<50):  {n_low} ({100*n_low/struct.n_residues:.1f}%)")

        # Detect source
        source = loader.detect_prediction_source(struct)
        click.echo(f"\nPredicted source: {source}")
    else:
        click.echo(f"\nB-factors (Å²):")
        click.echo(f"  Mean:   {struct.mean_plddt:.1f}")
        click.echo(f"  Min:    {struct.plddt.min():.1f}")
        click.echo(f"  Max:    {struct.plddt.max():.1f}")

        # B-factor interpretation
        n_rigid = sum(struct.plddt < 20)
        n_flexible = sum(struct.plddt > 40)
        click.echo(f"  Rigid regions (B<20):    {n_rigid} ({100*n_rigid/struct.n_residues:.1f}%)")
        click.echo(f"  Flexible regions (B>40): {n_flexible} ({100*n_flexible/struct.n_residues:.1f}%)")

        click.echo(f"\nStructure type: Experimental")


@cli.command()
@click.argument("results_csv", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.option("--format", "fmt", type=click.Choice(["text", "html", "json"]),
              default="text", help="Output format")
@click.option("--min-tm", type=float, help="Filter by minimum TM-score")
@click.option("--max-rmsd", type=float, help="Filter by maximum RMSD")
def report(results_csv, output, fmt, min_tm, max_rmsd):
    """Generate report from batch comparison results.

    Reads a CSV file from the 'batch' command and generates
    formatted reports or applies filters.
    """
    import pandas as pd

    results = pd.read_csv(results_csv)
    reporter = ComparisonReporter(results)

    # Apply filters
    if min_tm or max_rmsd:
        results = reporter.filter_results(
            min_tm_score=min_tm,
            max_rmsd=max_rmsd,
        )
        reporter.set_results(results)
        click.echo(f"Filtered to {len(results)} comparisons")

    if fmt == "text":
        report_text = reporter.summary_report()
        if output:
            Path(output).write_text(report_text)
            click.echo(f"Report saved to: {output}")
        else:
            click.echo(report_text)

    elif fmt == "html":
        out_path = output or "report.html"
        reporter.generate_html_report(out_path)
        click.echo(f"HTML report saved to: {out_path}")

    elif fmt == "json":
        out_path = output or "report.json"
        reporter.to_json(out_path)
        click.echo(f"JSON saved to: {out_path}")


@cli.command()
@click.argument("structure", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path (without extension for 'both' format)")
@click.option("--format", "fmt", type=click.Choice(["html", "pdf", "both", "json", "all"]), default="both",
              help="Output format: html, pdf, both, json, or all (html+pdf+json)")
@click.option("--contact-cutoff", default=8.0, help="Contact map distance cutoff (Å)")
@click.option("--dpi", default=150, help="Image resolution for plots")
@click.option("--experimental", "is_experimental", is_flag=True, default=False,
              help="Treat as experimental structure (B-factors instead of pLDDT)")
@click.option("--predicted", "is_predicted", is_flag=True, default=False,
              help="Treat as predicted structure (pLDDT confidence scores)")
@click.option("--pae", "pae_path", type=click.Path(exists=True), default=None,
              help="Path to PAE JSON file (AlphaFold predicted aligned error)")
@click.option("--chai-scores", "chai_scores_path", type=click.Path(exists=True), default=None,
              help="Path to Chai scores NPZ file (scores.model_idx_*.npz)")
@click.option("--msa", "msa_path", type=click.Path(exists=True), default=None,
              help="Path to MSA parquet file (Chai .aligned.pqt, requires pyarrow)")
def characterize(structure, output, fmt, contact_cutoff, dpi, is_experimental, is_predicted, pae_path, chai_scores_path, msa_path):
    """Generate comprehensive characterization report for a structure.

    Analyzes confidence scores, contacts, secondary structure, and
    sequence composition. Outputs HTML and/or PDF with embedded figures.

    By default, auto-detects whether the structure is predicted (pLDDT)
    or experimental (B-factors). Use --experimental or --predicted to override.

    For AlphaFold structures, use --pae to provide the PAE JSON file for
    domain analysis and confidence visualization.

    For Chai structures, use --chai-scores to provide the scores NPZ file
    for pTM/ipTM display. Use --msa to provide MSA parquet files for
    MSA depth visualization (requires pyarrow).

    Examples:

        protein_compare characterize structure.pdb -o report

        protein_compare characterize structure.pdb -o report.html --format html

        protein_compare characterize experimental.pdb --experimental

        protein_compare characterize alphafold.pdb --predicted --dpi 300

        protein_compare characterize alphafold.pdb --pae alphafold_scores.json

        protein_compare characterize chai.cif --chai-scores scores.model_idx_0.npz

        protein_compare characterize chai.cif --chai-scores scores.npz --msa msas/seq.aligned.pqt
    """
    click.echo("Loading structure...")

    loader = StructureLoader()
    try:
        struct = loader.load(structure)
    except Exception as e:
        click.echo(f"Error loading structure: {e}", err=True)
        sys.exit(1)

    # Determine structure type
    if is_experimental and is_predicted:
        click.echo("Error: Cannot specify both --experimental and --predicted", err=True)
        sys.exit(1)
    elif is_experimental:
        structure_type = "experimental"
    elif is_predicted:
        structure_type = "predicted"
    else:
        structure_type = None  # Auto-detect

    # Create characterizer (will auto-detect if structure_type is None)
    characterizer = StructureCharacterizer(
        structure=struct,
        contact_cutoff=contact_cutoff,
        dpi=dpi,
        structure_type=structure_type,
        pae_path=pae_path,
        chai_scores_path=chai_scores_path,
        msa_path=msa_path,
    )

    # Show structure info with appropriate terminology
    if characterizer.is_predicted:
        click.echo(f"  {struct.name}: {struct.n_residues} residues, mean pLDDT: {struct.mean_plddt:.1f}")
        click.echo(f"  Structure type: Predicted (pLDDT confidence scores)")
        if characterizer.has_pae:
            pae_analysis = characterizer.analyze_pae()
            click.echo(f"  PAE loaded: mean {pae_analysis.mean_pae:.1f} Å, {pae_analysis.n_domains} domain(s) detected")
            if pae_analysis.pae_data.ptm is not None:
                click.echo(f"  pTM: {pae_analysis.pae_data.ptm:.3f}")
        if characterizer.has_chai_scores:
            scores = characterizer.chai_scores
            click.echo(f"  Chai scores loaded: pTM={scores.ptm:.3f}, ipTM={scores.iptm:.3f}")
        if characterizer.has_msa_depth:
            msa = characterizer.msa_depth
            click.echo(f"  MSA depth loaded: mean={msa.mean_depth:.0f}, max={msa.max_depth}")
    else:
        click.echo(f"  {struct.name}: {struct.n_residues} residues, mean B-factor: {struct.mean_plddt:.1f} Ų")
        click.echo(f"  Structure type: Experimental (B-factor flexibility)")

    click.echo("\nGenerating characterization report...")

    # Determine output paths
    if output is None:
        output = struct.name

    # Remove extension if provided for "both" format
    output_base = output
    if output.endswith(".html") or output.endswith(".pdf"):
        output_base = output.rsplit(".", 1)[0]

    try:
        if fmt in ("html", "both", "all"):
            html_path = f"{output_base}.html" if not output.endswith(".html") else output
            click.echo("  Generating HTML report...")
            characterizer.generate_html_report(html_path)
            click.echo(f"  HTML report saved to: {html_path}")

        if fmt in ("pdf", "both", "all"):
            pdf_path = f"{output_base}.pdf" if not output.endswith(".pdf") else output
            click.echo("  Generating PDF report...")
            characterizer.generate_pdf_report(pdf_path)
            click.echo(f"  PDF report saved to: {pdf_path}")

        if fmt in ("json", "all"):
            json_path = f"{output_base}.json" if not output.endswith(".json") else output
            click.echo("  Generating JSON analysis data...")
            characterizer.generate_json_report(json_path)
            click.echo(f"  JSON analysis saved to: {json_path}")

    except Exception as e:
        click.echo(f"Error generating report: {e}", err=True)
        sys.exit(1)

    click.echo("\nCharacterization complete!")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
