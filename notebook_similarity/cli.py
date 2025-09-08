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
        description="Detect similar or identical submissions in code files (Jupyter notebooks and Python files)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all notebooks and Python files in current directory
  notebook-similarity .
  
  # Analyze only Python files
  notebook-similarity /path/to/files --types .py
  
  # Analyze both notebooks and Python files
  notebook-similarity /path/to/files --types .ipynb .py
  
  # Analyze with custom threshold
  notebook-similarity /path/to/files --threshold 0.8
  
  # Compare two specific files
  notebook-similarity --compare file1.py file2.ipynb
  
  # Analyze without text files (just use filenames)
  notebook-similarity /path/to/files --no-txt
  
  # Use fewer workers for parallel processing
  notebook-similarity /path/to/files --workers 4
        """
    )
    
    # Add arguments
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to directory containing code files (default: current directory)"
    )
    
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE1", "FILE2"),
        help="Compare two specific code files (.ipynb or .py)"
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
        "--types",
        nargs="+",
        default=None,
        help="File types to analyze (e.g., .ipynb .py, default: both)"
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
            # Compare two specific files
            file1, file2 = args.compare
            
            # Check files exist
            if not Path(file1).exists():
                print(f"Error: File not found: {file1}")
                sys.exit(1)
            if not Path(file2).exists():
                print(f"Error: File not found: {file2}")
                sys.exit(1)
            
            # Check file types
            valid_extensions = ['.ipynb', '.py']
            if Path(file1).suffix not in valid_extensions:
                print(f"Error: Unsupported file type: {file1}")
                print(f"Supported types: {', '.join(valid_extensions)}")
                sys.exit(1)
            if Path(file2).suffix not in valid_extensions:
                print(f"Error: Unsupported file type: {file2}")
                print(f"Supported types: {', '.join(valid_extensions)}")
                sys.exit(1)
            
            # Run comparison
            quick_check(file1, file2)
        else:
            # Analyze directory
            path = Path(args.path)
            
            if not path.exists():
                print(f"Error: Path not found: {path}")
                sys.exit(1)
            
            if not path.is_dir():
                print(f"Error: Path is not a directory: {path}")
                sys.exit(1)
            
            # Process file types argument
            file_types = args.types
            if file_types:
                # Ensure all types start with a dot
                file_types = [t if t.startswith('.') else f'.{t}' for t in file_types]
                # Validate file types
                valid_types = ['.ipynb', '.py']
                for ft in file_types:
                    if ft not in valid_types:
                        print(f"Error: Unsupported file type: {ft}")
                        print(f"Supported types: {', '.join(valid_types)}")
                        sys.exit(1)
            
            # Run analysis
            detector, df, clusters = analyze_directory(
                directory_path=str(path),
                output_dir=args.output,
                similarity_threshold=args.threshold,
                num_workers=args.workers,
                include_txt_files=not args.no_txt,
                file_types=file_types,
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