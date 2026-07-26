"""
run_ablation.py — Unified entry point for ablation experiments.

Using predefined datasets (recommended):
  python -m ablation.run_ablation --dataset agent72
  python -m ablation.run_ablation --dataset twibot1000 --repeats 3 --folds 5

Custom data paths:
  python -m ablation.run_ablation --db "path/to/data.db" --csv "path/to/labels.csv"

Independence verification only:
  python -m ablation.run_ablation --dataset agent72 --skip-ablation

Ablation only (skip independence check):
  python -m ablation.run_ablation --dataset agent72 --skip-independence
"""

import argparse
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Emotion module ablation experiment & independence verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Data paths (three specification modes)
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["agent72", "72", "test72", "twibot120", "twibot",
                                  "twibot1000", "sim1000", "sim"],
                        help="Predefined dataset name (e.g., agent72, twibot1000)")
    parser.add_argument("--db", type=str, default=None,
                        help="SQLite database path (overrides preset)")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV label file path (overrides preset)")
    
    # Cross-validation
    parser.add_argument("--repeats", type=int, default=5,
                        help="RepeatedStratifiedKFold repeats (default: 5)")
    parser.add_argument("--folds", type=int, default=5,
                        help="K-fold splits (default: 5)")
    
    # Skip options
    parser.add_argument("--skip-independence", action="store_true",
                        help="Skip independence verification, run ablation only")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation experiment, run independence verification only")
    
    # Output
    parser.add_argument("--output", type=str, default="./ablation_results",
                        help="Output directory (default: ./ablation_results)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("   Emotion Module Ablation Study & Independence Verification Tool")
    print("=" * 70)
    
    from ablation.ablation_experiment import run_ablation
    from ablation.ablation_plot import plot_all
    
    # Execute ablation experiment (internally calls independence verification first)
    ablation_results = run_ablation(
        db_path=args.db,
        csv_path=args.csv,
        dataset=args.dataset,
        n_repeats=args.repeats,
        n_splits=args.folds,
        save_dir=args.output,
        skip_independence=args.skip_independence,
        skip_ablation=args.skip_ablation,
    )
    
    # If ablation results exist, auto-generate bar chart
    if ablation_results:
        from ablation.ablation_plot import plot_ablation_bar_chart
        plot_ablation_bar_chart(
            ablation_results,
            save_path=os.path.join(args.output, "ablation_bar_chart.png")
        )
    
    print("\n" + "=" * 70)
    print(f"   All completed. Results saved to: {os.path.abspath(args.output)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()