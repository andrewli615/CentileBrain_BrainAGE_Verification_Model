"""Generate the pinned-source BrainAGE comparison, figures, and verification."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent/"results/.mplconfig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify saved results without rebuilding")
    parser.add_argument("--skip-plots", action="store_true", help="Rebuild tables/workbook without redrawing figures")
    args = parser.parse_args()
    from scripts.shared.pipeline import build
    from scripts.statistics.workbook import build_workbook
    from scripts.validation.verify_results import verify
    if not args.verify_only:
        cohorts, training = build()
        if not args.skip_plots:
            from scripts.plots.method_comparisons import generate
            print(f"Generated {len(generate(cohorts,training))} combined comparison figures")
        build_workbook()
    verify()


if __name__ == "__main__":
    main()
