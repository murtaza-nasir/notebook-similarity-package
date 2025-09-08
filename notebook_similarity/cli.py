#!/usr/bin/env python3
"""
Command-line interface for Notebook Similarity Detector.
"""

import argparse
import sys
from pathlib import Path
from .analyzer import analyze_directory, quick_check

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Detect similar or identical submissions in Jupyter notebooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all notebooks in current directory
  notebook-similarity .
  
  # Analyze with custom threshold
  notebook-similarity /path/to/notebooks --threshold 0.8
  
  # Compare two specific notebooks
  notebook-similarity --compare notebook1.ipynb notebook2.ipynb
  
  # Analyze without text files (just use notebook filenames)
  notebook-similarity /path/to/notebooks --no-txt
  
  # Use fewer workers for parallel processing
  notebook-similarity /path/to/notebooks --workers 4
        """
    )
    
    # Add arguments
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to directory containing notebooks (default: current directory)"
    )
    
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("NOTEBOOK1", "NOTEBOOK2"),
        help="Compare two specific notebooks"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for flagging (0-1, default: 0.7)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for reports (default: input directory)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers (default: CPU count)"
    )
    
    parser.add_argument(
        "--no-txt",
        action="store_true",
        help="Don't look for accompanying .txt files with metadata"
    )
    
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Compare two specific notebooks
            notebook1, notebook2 = args.compare
            
            # Check files exist
            if not Path(notebook1).exists():
                print(f"Error: File not found: {notebook1}")
                sys.exit(1)
            if not Path(notebook2).exists():
                print(f"Error: File not found: {notebook2}")
                sys.exit(1)
            
            # Run comparison
            quick_check(notebook1, notebook2)
        else:
            # Analyze directory
            path = Path(args.path)
            
            if not path.exists():
                print(f"Error: Path not found: {path}")
                sys.exit(1)
            
            if not path.is_dir():
                print(f"Error: Path is not a directory: {path}")
                sys.exit(1)
            
            # Run analysis
            detector, df, clusters = analyze_directory(
                directory_path=str(path),
                output_dir=args.output,
                similarity_threshold=args.threshold,
                num_workers=args.workers,
                include_txt_files=not args.no_txt,
                verbose=not args.quiet
            )
            
            if df is not None and not args.quiet:
                # Print summary
                flagged = df[df['flagged']].shape[0]
                exact = df[df['exact_match']].sum()
                
                print(f"\nSummary:")
                print(f"  • Flagged pairs: {flagged}")
                print(f"  • Exact matches: {exact}")
                print(f"  • Clusters found: {len(clusters)}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()