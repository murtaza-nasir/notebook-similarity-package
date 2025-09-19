#!/usr/bin/env python3
"""
Standalone runner script for Notebook Similarity Detector.
This script can be run directly without installing the package.
"""

import sys
import os
from pathlib import Path
import argparse

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from notebook_similarity import analyze_directory
from notebook_similarity.analyzer import quick_check

def main():
    """Main function for standalone execution."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Notebook Similarity Detector')
    parser.add_argument('path', nargs='?', help='Path to directory or notebook file to analyze')
    parser.add_argument('second_file', nargs='?', help='Second notebook for comparison (optional)')
    parser.add_argument('--threshold', '-t', type=float, default=None, 
                       help='Similarity threshold (0.5-1.0, default: 0.7)')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Run non-interactively with defaults')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NOTEBOOK SIMILARITY DETECTOR")
    print("=" * 60)
    
    # Get target path
    if args.path:
        target_path = args.path
    elif not args.no_interactive:
        # Ask user for path
        target_path = input("\nEnter path to analyze (or press Enter for current directory): ").strip()
        if not target_path:
            target_path = "."
    else:
        target_path = "."
    
    # Check if path exists
    path = Path(target_path)
    if not path.exists():
        print(f"Error: Path not found: {path}")
        sys.exit(1)
    
    # Check if it's a comparison of two files
    if path.is_file() and path.suffix == '.ipynb':
        if args.second_file:
            second_file = args.second_file
            if Path(second_file).exists():
                print(f"\nComparing two notebooks:")
                print(f"  • {path.name}")
                print(f"  • {Path(second_file).name}")
                quick_check(str(path), second_file)
            else:
                print(f"Error: Second file not found: {second_file}")
                sys.exit(1)
        else:
            print("Error: Please provide a second notebook to compare")
            print("Usage: python run_similarity_check.py notebook1.ipynb notebook2.ipynb")
            sys.exit(1)
    elif path.is_dir():
        # Analyze directory
        print(f"\nAnalyzing directory: {path.absolute()}")
        
        # Get threshold
        if args.threshold is not None:
            threshold = args.threshold
            if not 0.5 <= threshold <= 1.0:
                print(f"Invalid threshold {threshold}. Using default 0.7")
                threshold = 0.7
        elif not args.no_interactive:
            # Ask for threshold
            threshold_input = input("Enter similarity threshold (0.5-1.0, default 0.7): ").strip()
            try:
                threshold = float(threshold_input) if threshold_input else 0.7
                if not 0.5 <= threshold <= 1.0:
                    print("Invalid threshold. Using default 0.7")
                    threshold = 0.7
            except ValueError:
                print("Invalid threshold. Using default 0.7")
                threshold = 0.7
        else:
            threshold = 0.7
        
        print(f"\nUsing similarity threshold: {threshold:.0%}")
        print("-" * 40)
        
        # Run analysis
        detector, df, clusters = analyze_directory(
            directory_path=str(path),
            similarity_threshold=threshold,
            verbose=True
        )
        
        if df is not None:
            # Show additional summary
            print("\n" + "=" * 60)
            print("ANALYSIS SUMMARY")
            print("=" * 60)
            
            flagged = df[df['flagged']].shape[0]
            exact = df[df['exact_match']].sum()
            
            if flagged > 0:
                print(f"\n⚠️  Found {flagged} flagged submission pairs")
                print(f"   Including {exact} exact matches")
                
                # Show top flagged pairs
                print("\nTop flagged pairs:")
                top_flagged = df[df['flagged']].head(5)
                for _, row in top_flagged.iterrows():
                    similarity = row['overall_similarity']
                    print(f"  • {row['student1_name']} ↔ {row['student2_name']}: {similarity:.1%}")
            else:
                print("\n✓ No suspicious similarities found!")
            
            print("\nReports generated:")
            print(f"  • HTML: similarity_report.html")
            print(f"  • CSV:  similarity_report.csv")
    else:
        print(f"Error: {path} is neither a directory nor a notebook file")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()