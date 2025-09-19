"""
High-level analyzer functions for easy usage.
"""

from pathlib import Path
from .detector import NotebookSimilarityDetector

def analyze_directory(
    directory_path: str,
    output_dir: str = None,
    similarity_threshold: float = 0.7,
    num_workers: int = None,
    include_txt_files: bool = True,
    file_types: list = None,
    verbose: bool = True
):
    """
    Analyze all code files (notebooks and Python files) in a directory for similarity.
    
    Args:
        directory_path: Path to directory containing code files
        output_dir: Directory for output files (defaults to input directory)
        similarity_threshold: Threshold for flagging similar submissions (0-1)
        num_workers: Number of parallel workers (defaults to CPU count)
        include_txt_files: Whether to look for accompanying .txt files with metadata
        file_types: List of file extensions to analyze (defaults to ['.ipynb', '.py'])
        verbose: Whether to print progress messages
    
    Returns:
        tuple: (detector instance, DataFrame of results, list of clusters)
    """
    # Set output directory
    if output_dir is None:
        output_dir = directory_path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = NotebookSimilarityDetector(
        folder_path=directory_path,
        num_workers=num_workers,
        include_txt_files=include_txt_files,
        file_types=file_types
    )
    
    # Load and analyze submissions
    detector.load_submissions(verbose=verbose)
    
    if len(detector.submissions) == 0:
        print(f"No submissions found in {directory_path}")
        return detector, None, []
    
    detector.analyze_similarities(
        similarity_threshold=similarity_threshold,
        verbose=verbose
    )
    
    # Generate report
    report_path = output_path / "similarity_report.html"
    df = detector.generate_report(str(report_path), verbose=verbose)
    
    # Find clusters
    clusters = detector.find_clusters(verbose=verbose)
    
    if verbose:
        print(f"\n[OK] Analysis complete!")
        print(f"  - Report: {report_path}")
        print(f"  - CSV: {str(report_path).replace('.html', '.csv')}")
    
    return detector, df, clusters

def quick_check(file1: str, file2: str):
    """
    Quick similarity check between two specific code files.
    
    Args:
        file1: Path to first file (.ipynb or .py)
        file2: Path to second file (.ipynb or .py)
    
    Returns:
        dict: Similarity metrics between the two files
    """
    from .detector import FileParser, CodeAnalyzer, StudentSubmission, compare_submissions
    
    # Parse first file
    code1, markdown1, file_type1 = FileParser.parse_file(file1)
    all_code1 = '\n'.join(code1)
    normalized1 = CodeAnalyzer.normalize_code(all_code1)
    vars1, funcs1, imports1 = CodeAnalyzer.extract_identifiers(all_code1)
    
    sub1 = StudentSubmission(
        student_id="student1",
        student_name=Path(file1).stem,
        file_path=file1,
        submission_time="",
        code_cells=code1,
        markdown_cells=markdown1,
        code_hash=CodeAnalyzer.compute_code_hash(all_code1),
        normalized_code=normalized1,
        ast_structure=CodeAnalyzer.extract_ast_structure(all_code1),
        variable_names=vars1,
        function_names=funcs1,
        import_statements=imports1,
        file_type=file_type1
    )
    
    # Parse second file
    code2, markdown2, file_type2 = FileParser.parse_file(file2)
    all_code2 = '\n'.join(code2)
    normalized2 = CodeAnalyzer.normalize_code(all_code2)
    vars2, funcs2, imports2 = CodeAnalyzer.extract_identifiers(all_code2)
    
    sub2 = StudentSubmission(
        student_id="student2",
        student_name=Path(file2).stem,
        file_path=file2,
        submission_time="",
        code_cells=code2,
        markdown_cells=markdown2,
        code_hash=CodeAnalyzer.compute_code_hash(all_code2),
        normalized_code=normalized2,
        ast_structure=CodeAnalyzer.extract_ast_structure(all_code2),
        variable_names=vars2,
        function_names=funcs2,
        import_statements=imports2,
        file_type=file_type2
    )
    
    # Compare
    result = compare_submissions((sub1, sub2), similarity_threshold=0.7)
    
    # Print results
    print(f"\nSimilarity Analysis:")
    print(f"  File 1: {Path(file1).name} ({file_type1})")
    print(f"  File 2: {Path(file2).name} ({file_type2})")
    print(f"  File types: {file_type1}, {file_type2}")
    print(f"\nResults:")
    print(f"  Overall Similarity: {result['overall_similarity']:.2%}")
    print(f"  Sequence Similarity: {result['sequence_similarity']:.2%}")
    print(f"  AST Similarity: {result['ast_similarity']:.2%}")
    print(f"  Variable Overlap: {result.get('variable_overlap', 0):.2%}")
    print(f"  Function Overlap: {result.get('function_overlap', 0):.2%}")
    print(f"  Exact Match: {'Yes' if result['exact_match'] else 'No'}")
    
    return result