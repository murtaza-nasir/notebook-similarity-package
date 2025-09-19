#!/usr/bin/env python3
"""
Core detector module for notebook similarity detection.
"""

import json
import os
import re
import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
import difflib
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

@dataclass
class StudentSubmission:
    """Represents a student submission."""
    student_id: str
    student_name: str
    file_path: str  # Changed from notebook_path to be more generic
    submission_time: str
    code_cells: List[str]
    markdown_cells: List[str]
    code_hash: str
    normalized_code: str
    ast_structure: str
    variable_names: Set[str]
    function_names: Set[str]
    import_statements: Set[str]
    file_type: str  # Added to track if it's .ipynb or .py

class FileParser:
    """Parses Jupyter notebooks and extracts relevant content."""
    
    @staticmethod
    def parse_txt_file(txt_path: str) -> Dict[str, str]:
        """Parse the text file to get student information."""
        info = {}
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract student ID from filename
        student_id_match = re.search(r'_([a-z]\d{3}[a-z]\d{3})_', txt_path)
        if student_id_match:
            info['student_id'] = student_id_match.group(1)
        
        # Extract name
        name_match = re.search(r'Name:\s*(.+?)\s*\(', content)
        if name_match:
            info['student_name'] = name_match.group(1)
        
        # Extract submission time
        date_match = re.search(r'Date Submitted:\s*(.+)', content)
        if date_match:
            info['submission_time'] = date_match.group(1)
            
        return info
    
    @staticmethod
    def parse_notebook(notebook_path: str) -> Tuple[List[str], List[str]]:
        """Parse notebook and extract code and markdown cells."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    print(f"  Warning: Empty notebook file: {Path(notebook_path).name}")
                    return [], []
                notebook = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  Warning: Invalid JSON in notebook {Path(notebook_path).name}: {str(e)}")
            return [], []
        except Exception as e:
            print(f"  Warning: Error reading notebook {Path(notebook_path).name}: {str(e)}")
            return [], []
        
        code_cells = []
        markdown_cells = []
        
        cells = notebook.get('cells', [])
        for cell in cells:
            cell_type = cell.get('cell_type', '')
            source = cell.get('source', [])
            
            # Handle both list and string formats
            if isinstance(source, list):
                content = ''.join(source)
            else:
                content = source
            
            if cell_type == 'code':
                code_cells.append(content)
            elif cell_type == 'markdown':
                markdown_cells.append(content)
        
        return code_cells, markdown_cells
    
    @staticmethod
    def parse_python_file(py_path: str) -> Tuple[List[str], List[str]]:
        """Parse Python file and extract code and docstrings."""
        with open(py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract docstrings as markdown cells
        markdown_cells = []
        # Find module-level docstrings
        docstring_pattern = r'^"""([\s\S]*?)"""'
        docstrings = re.findall(docstring_pattern, content, re.MULTILINE)
        markdown_cells.extend(docstrings)
        
        # Treat the entire file content as one code cell
        code_cells = [content]
        
        return code_cells, markdown_cells
    
    @staticmethod
    def parse_file(file_path: str) -> Tuple[List[str], List[str], str]:
        """Parse a file based on its extension."""
        path = Path(file_path)
        if path.suffix == '.ipynb':
            code_cells, markdown_cells = FileParser.parse_notebook(file_path)
            return code_cells, markdown_cells, 'notebook'
        elif path.suffix == '.py':
            code_cells, markdown_cells = FileParser.parse_python_file(file_path)
            return code_cells, markdown_cells, 'python'
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

class CodeAnalyzer:
    """Analyzes code for similarity detection."""
    
    @staticmethod
    def normalize_code(code: str) -> str:
        """Normalize code by removing comments, extra whitespace, and standardizing."""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'"""[\s\S]*?"""', '', code)
        code = re.sub(r"'''[\s\S]*?'''", '', code)
        
        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code)
        code = code.strip()
        
        # Standardize quotes
        code = code.replace('"', "'")
        
        return code
    
    @staticmethod
    def extract_ast_structure(code: str) -> str:
        """Extract AST structure for structural comparison."""
        try:
            tree = ast.parse(code)
            return ast.dump(tree, annotate_fields=False)
        except:
            return ""
    
    @staticmethod
    def extract_identifiers(code: str) -> Tuple[Set[str], Set[str], Set[str]]:
        """Extract variable names, function names, and imports from code."""
        variables = set()
        functions = set()
        imports = set()
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    variables.add(node.id)
                elif isinstance(node, ast.FunctionDef):
                    functions.add(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        except:
            pass
        
        return variables, functions, imports
    
    @staticmethod
    def compute_code_hash(code: str) -> str:
        """Compute hash of normalized code."""
        normalized = CodeAnalyzer.normalize_code(code)
        return hashlib.md5(normalized.encode()).hexdigest()

class SimilarityDetector:
    """Detects similarities between submissions."""
    
    @staticmethod
    def exact_match(sub1: StudentSubmission, sub2: StudentSubmission) -> bool:
        """Check if two submissions are exactly the same."""
        return sub1.code_hash == sub2.code_hash
    
    @staticmethod
    def sequence_similarity(sub1: StudentSubmission, sub2: StudentSubmission) -> float:
        """Calculate sequence similarity using SequenceMatcher."""
        return difflib.SequenceMatcher(None, 
                                      sub1.normalized_code, 
                                      sub2.normalized_code).ratio()
    
    @staticmethod
    def ast_similarity(sub1: StudentSubmission, sub2: StudentSubmission) -> float:
        """Calculate AST structure similarity."""
        if not sub1.ast_structure or not sub2.ast_structure:
            return 0.0
        return difflib.SequenceMatcher(None, 
                                      sub1.ast_structure, 
                                      sub2.ast_structure).ratio()
    
    @staticmethod
    def token_overlap(sub1: StudentSubmission, sub2: StudentSubmission) -> Dict[str, float]:
        """Calculate overlap in identifiers."""
        results = {}
        
        # Variable overlap
        if sub1.variable_names or sub2.variable_names:
            var_intersection = len(sub1.variable_names & sub2.variable_names)
            var_union = len(sub1.variable_names | sub2.variable_names)
            results['variable_overlap'] = var_intersection / var_union if var_union > 0 else 0
        
        # Function overlap
        if sub1.function_names or sub2.function_names:
            func_intersection = len(sub1.function_names & sub2.function_names)
            func_union = len(sub1.function_names | sub2.function_names)
            results['function_overlap'] = func_intersection / func_union if func_union > 0 else 0
        
        # Import overlap
        if sub1.import_statements or sub2.import_statements:
            import_intersection = len(sub1.import_statements & sub2.import_statements)
            import_union = len(sub1.import_statements | sub2.import_statements)
            results['import_overlap'] = import_intersection / import_union if import_union > 0 else 0
        
        return results
    
    @staticmethod
    def longest_common_substring(sub1: StudentSubmission, sub2: StudentSubmission) -> int:
        """Find the length of the longest common substring."""
        s1 = sub1.normalized_code
        s2 = sub2.normalized_code
        
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_length = max(max_length, dp[i][j])
        
        return max_length

def process_submission(file_path: Path, folder_path: Path, include_txt: bool = True) -> StudentSubmission:
    """Process a single submission - used for parallel processing."""
    student_info = {'student_id': 'unknown', 'student_name': 'Unknown', 'submission_time': ''}
    
    if include_txt:
        # Try to find corresponding txt file for both notebooks and Python files
        base_name = file_path.stem
        txt_files = []
        
        # Strategy 1: Look for exact match with .txt extension
        txt_pattern = f"{base_name}.txt"
        txt_files = list(folder_path.glob(txt_pattern))
        
        # Strategy 2: Extract the student ID pattern and attempt date
        # Pattern: CA3.1 - Dropbox_STUDENTID_attempt_DATE_*
        if not txt_files:
            import re
            # Match pattern like: _x123y456_attempt_2025-09-07-18-27-28_
            pattern = r'(.+_[a-z]\d{3}[a-z]\d{3}_attempt_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})'
            match = re.match(pattern, base_name)
            if match:
                txt_pattern = f"{match.group(1)}*.txt"
                txt_files = list(folder_path.glob(txt_pattern))
        
        # Strategy 3: Remove everything after the attempt timestamp
        if not txt_files and 'attempt_' in base_name:
            # Split on 'attempt_' and reconstruct up to the timestamp
            parts = base_name.split('attempt_')
            if len(parts) >= 2:
                # Get the timestamp part (2025-09-07-18-27-28)
                timestamp_and_rest = parts[1]
                # Extract just the timestamp (first 19 characters: 2025-09-07-18-27-28)
                timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', timestamp_and_rest)
                if timestamp_match:
                    txt_pattern = f"{parts[0]}attempt_{timestamp_match.group(1)}.txt"
                    txt_files = list(folder_path.glob(txt_pattern))
        
        # Strategy 4: For files with _CA or other suffix at the end, remove it
        if not txt_files and '_CA' in base_name:
            txt_pattern = base_name.split('_CA')[0]
            txt_files = list(folder_path.glob(f"{txt_pattern}*.txt"))
        
        # Strategy 5: Try removing the last segment after underscore
        if not txt_files and '_' in base_name:
            parts = base_name.rsplit('_', 1)
            if len(parts) > 1:
                txt_pattern = parts[0]
                txt_files = list(folder_path.glob(f"{txt_pattern}*.txt"))
        
        if txt_files:
            # Use the first matching txt file
            txt_path = txt_files[0]
            student_info = FileParser.parse_txt_file(str(txt_path))
    
    if student_info['student_id'] == 'unknown':
        # Extract student info from filename if no txt file found
        filename = file_path.stem
        student_info['student_name'] = filename
        student_info['student_id'] = filename.lower().replace(' ', '_')
    
    # Parse file based on type
    code_cells, markdown_cells, file_type = FileParser.parse_file(str(file_path))
    
    # Skip files with no content
    if not code_cells and not markdown_cells:
        return None
    
    # Combine all code
    all_code = '\n'.join(code_cells)
    
    # Analyze code
    normalized = CodeAnalyzer.normalize_code(all_code)
    code_hash = CodeAnalyzer.compute_code_hash(all_code)
    ast_structure = CodeAnalyzer.extract_ast_structure(all_code)
    variables, functions, imports = CodeAnalyzer.extract_identifiers(all_code)
    
    return StudentSubmission(
        student_id=student_info.get('student_id', 'unknown'),
        student_name=student_info.get('student_name', 'Unknown'),
        file_path=str(file_path),
        submission_time=student_info.get('submission_time', ''),
        code_cells=code_cells,
        markdown_cells=markdown_cells,
        code_hash=code_hash,
        normalized_code=normalized,
        ast_structure=ast_structure,
        variable_names=variables,
        function_names=functions,
        import_statements=imports,
        file_type=file_type
    )

def compare_submissions(pair: Tuple[StudentSubmission, StudentSubmission], 
                        similarity_threshold: float) -> Dict:
    """Compare two submissions - used for parallel processing."""
    sub1, sub2 = pair
    
    # Calculate various similarity metrics
    is_exact = SimilarityDetector.exact_match(sub1, sub2)
    seq_sim = SimilarityDetector.sequence_similarity(sub1, sub2)
    ast_sim = SimilarityDetector.ast_similarity(sub1, sub2)
    token_overlaps = SimilarityDetector.token_overlap(sub1, sub2)
    lcs_length = SimilarityDetector.longest_common_substring(sub1, sub2)
    
    # Calculate overall similarity score
    scores = [seq_sim, ast_sim]
    scores.extend(token_overlaps.values())
    overall_similarity = np.mean([s for s in scores if s is not None])
    
    return {
        'student1_id': sub1.student_id,
        'student1_name': sub1.student_name,
        'student2_id': sub2.student_id,
        'student2_name': sub2.student_name,
        'exact_match': is_exact,
        'sequence_similarity': seq_sim,
        'ast_similarity': ast_sim,
        'variable_overlap': token_overlaps.get('variable_overlap', 0),
        'function_overlap': token_overlaps.get('function_overlap', 0),
        'import_overlap': token_overlaps.get('import_overlap', 0),
        'lcs_length': lcs_length,
        'overall_similarity': overall_similarity,
        'flagged': overall_similarity >= similarity_threshold or is_exact
    }

class NotebookSimilarityDetector:
    """Main analyzer for detecting similar code submissions (notebooks and Python files)."""
    
    def __init__(self, folder_path: str, num_workers: int = None, include_txt_files: bool = True, file_types: List[str] = None):
        self.folder_path = Path(folder_path)
        self.submissions = []
        self.similarity_results = []
        self.num_workers = num_workers or mp.cpu_count()
        self.include_txt_files = include_txt_files
        self.file_types = file_types or ['.ipynb', '.py']  # Support both by default
        
    def load_submissions(self, verbose: bool = True):
        """Load all submissions from the folder using parallel processing."""
        start_time = time.time()
        if verbose:
            print(f"Loading submissions using {self.num_workers} workers...")
        
        # Find all files based on specified types
        all_files = []
        for file_type in self.file_types:
            pattern = f"*{file_type}"
            files = list(self.folder_path.glob(pattern))
            all_files.extend(files)
            if verbose and files:
                print(f"  Found {len(files)} {file_type} files")
        
        if not all_files:
            print(f"No files found with extensions {self.file_types} in {self.folder_path}")
            return
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = []
            for file_path in all_files:
                future = executor.submit(process_submission, file_path, 
                                       self.folder_path, self.include_txt_files)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                if result:
                    self.submissions.append(result)
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Loaded {len(self.submissions)} submissions in {elapsed:.2f} seconds")
    
    def analyze_similarities(self, similarity_threshold: float = 0.7, verbose: bool = True):
        """Analyze similarities between all submissions using parallel processing."""
        start_time = time.time()
        if verbose:
            print(f"\nAnalyzing similarities (threshold: {similarity_threshold:.0%})...")
        
        # Generate all pairs to compare
        n = len(self.submissions)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((self.submissions[i], self.submissions[j]))
        
        if verbose:
            print(f"Comparing {len(pairs)} pairs...")
        
        # Process comparisons in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Create partial function with threshold
            compare_func = partial(compare_submissions, similarity_threshold=similarity_threshold)
            
            # Submit all comparison tasks
            futures = []
            for pair in pairs:
                future = executor.submit(compare_func, pair)
                futures.append(future)
            
            # Collect results with progress indicator
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                self.similarity_results.append(result)
                completed += 1
                if verbose and completed % 50 == 0:
                    print(f"  Processed {completed}/{len(pairs)} comparisons...")
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Completed {len(pairs)} comparisons in {elapsed:.2f} seconds")
    
    def generate_report(self, output_file: str = "similarity_report.html", verbose: bool = True):
        """Generate an HTML report of the findings."""
        if verbose:
            print("\nGenerating report...")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.similarity_results)
        
        # Sort by overall similarity
        df = df.sort_values('overall_similarity', ascending=False)
        
        # Identify problematic students (those involved in flagged submissions)
        problematic_students = set()
        for _, row in df[df['flagged']].iterrows():
            problematic_students.add(row['student1_id'])
            problematic_students.add(row['student2_id'])
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Notebook Similarity Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .exact-match {{ background-color: #ffcccc !important; font-weight: bold; }}
                .high-similarity {{ background-color: #ffe6cc !important; }}
                .medium-similarity {{ background-color: #ffffcc !important; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
                .timestamp {{ color: #888; font-size: 12px; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <h1>Notebook Similarity Detection Report</h1>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            <div class="summary">
                <h2>Summary Statistics</h2>
        """
        
        # Calculate summary statistics
        total_comparisons = len(df)
        exact_matches = df['exact_match'].sum()
        flagged_count = df['flagged'].sum()
        high_similarity = len(df[df['overall_similarity'] >= 0.8])
        medium_similarity = len(df[(df['overall_similarity'] >= 0.6) & (df['overall_similarity'] < 0.8)])
        
        html += f"""
                <div class="metric">
                    <div>Total Comparisons:</div>
                    <div class="metric-value">{total_comparisons}</div>
                </div>
                <div class="metric">
                    <div>Exact Matches:</div>
                    <div class="metric-value" style="color: red;">{exact_matches}</div>
                </div>
                <div class="metric">
                    <div>Flagged Submissions:</div>
                    <div class="metric-value" style="color: orange;">{flagged_count}</div>
                </div>
                <div class="metric">
                    <div>High Similarity (≥80%):</div>
                    <div class="metric-value">{high_similarity}</div>
                </div>
                <div class="metric">
                    <div>Medium Similarity (60-80%):</div>
                    <div class="metric-value">{medium_similarity}</div>
                </div>
            </div>
            
            <h2>Flagged Submissions (Require Review)</h2>
            <table>
                <tr>
                    <th>Student 1</th>
                    <th>Student 2</th>
                    <th>Overall Similarity</th>
                    <th>Sequence Similarity</th>
                    <th>AST Similarity</th>
                    <th>Variable Overlap</th>
                    <th>Function Overlap</th>
                    <th>Exact Match</th>
                </tr>
        """
        
        # Add flagged submissions to the table
        flagged_df = df[df['flagged']].head(50)  # Show top 50 flagged pairs
        
        for _, row in flagged_df.iterrows():
            row_class = ""
            if row['exact_match']:
                row_class = 'class="exact-match"'
            elif row['overall_similarity'] >= 0.8:
                row_class = 'class="high-similarity"'
            elif row['overall_similarity'] >= 0.6:
                row_class = 'class="medium-similarity"'
            
            html += f"""
                <tr {row_class}>
                    <td>{row['student1_name']} ({row['student1_id']})</td>
                    <td>{row['student2_name']} ({row['student2_id']})</td>
                    <td>{row['overall_similarity']:.2%}</td>
                    <td>{row['sequence_similarity']:.2%}</td>
                    <td>{row['ast_similarity']:.2%}</td>
                    <td>{row['variable_overlap']:.2%}</td>
                    <td>{row['function_overlap']:.2%}</td>
                    <td>{'Yes' if row['exact_match'] else 'No'}</td>
                </tr>
            """
        
        html += """
            </table>
        """
        
        # Add student submissions table
        html += """
            <h2>All Student Submissions</h2>
            <p>Submissions sorted by submission time. Highlighted rows indicate students involved in flagged similarities.</p>
            <table>
                <tr>
                    <th>Student Name</th>
                    <th>Student ID</th>
                    <th>File Type</th>
                    <th>Submission Time</th>
                    <th>Status</th>
                    <th>Similar To</th>
                </tr>
        """
        
        # Create a list of all unique students with their info
        students_info = {}
        for submission in self.submissions:
            students_info[submission.student_id] = {
                'name': submission.student_name,
                'id': submission.student_id,
                'file_type': submission.file_type,
                'time': submission.submission_time,
                'is_problematic': submission.student_id in problematic_students,
                'similar_to': []
            }
        
        # Add similarity information
        for _, row in df[df['flagged']].iterrows():
            student1_id = row['student1_id']
            student2_id = row['student2_id']
            similarity = row['overall_similarity']
            
            if student1_id in students_info:
                students_info[student1_id]['similar_to'].append(
                    f"{row['student2_name']} ({similarity:.0%})"
                )
            if student2_id in students_info:
                students_info[student2_id]['similar_to'].append(
                    f"{row['student1_name']} ({similarity:.0%})"
                )
        
        # Sort students by submission time
        def parse_submission_time(time_str):
            """Parse Blackboard submission time format."""
            if not time_str:
                return datetime.min
            try:
                # Try to parse Blackboard format: "Saturday, August 30, 2025 8:33:54 PM CDT"
                # Remove timezone abbreviation for parsing
                time_str = re.sub(r'\s+[A-Z]{3,4}$', '', time_str)
                return datetime.strptime(time_str, '%A, %B %d, %Y %I:%M:%S %p')
            except:
                return datetime.min
        
        sorted_students = sorted(
            students_info.items(),
            key=lambda x: parse_submission_time(x[1]['time'])
        )
        
        # Add students to table
        for student_id, info in sorted_students:
            row_class = ""
            status = "OK"
            
            if info['is_problematic']:
                # Determine severity
                max_similarity = 0
                for sim_info in info['similar_to']:
                    # Extract percentage from string like "Name (95%)"
                    match = re.search(r'\((\d+)%\)', sim_info)
                    if match:
                        max_similarity = max(max_similarity, int(match.group(1)))
                
                if max_similarity == 100:
                    row_class = 'class="exact-match"'
                    status = "EXACT MATCH"
                elif max_similarity >= 80:
                    row_class = 'class="high-similarity"'
                    status = "HIGH SIMILARITY"
                else:
                    row_class = 'class="medium-similarity"'
                    status = "FLAGGED"
            
            similar_to_str = '<br>'.join(info['similar_to']) if info['similar_to'] else '-'
            
            html += f"""
                <tr {row_class}>
                    <td>{info['name']}</td>
                    <td>{info['id']}</td>
                    <td>{info.get('file_type', 'unknown')}</td>
                    <td>{info['time'] if info['time'] else 'Unknown'}</td>
                    <td>{status}</td>
                    <td>{similar_to_str}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>All Comparisons</h2>
            <p>Full data exported to CSV file for detailed analysis.</p>
            
            <div style="margin-top: 30px; padding: 10px; background-color: #f0f0f0; border-left: 4px solid #4CAF50;">
                <strong>Legend:</strong><br>
                <span style="background-color: #ffcccc; padding: 2px 5px;">Red</span> = Exact Match<br>
                <span style="background-color: #ffe6cc; padding: 2px 5px;">Orange</span> = High Similarity (≥80%)<br>
                <span style="background-color: #ffffcc; padding: 2px 5px;">Yellow</span> = Medium Similarity (60-80%)
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        # Also save detailed CSV
        csv_file = output_file.replace('.html', '.csv')
        df.to_csv(csv_file, index=False)
        
        if verbose:
            print(f"Report saved to {output_file}")
            print(f"Detailed data saved to {csv_file}")
        
        return df
    
    def find_clusters(self, verbose: bool = True):
        """Find clusters of similar submissions."""
        if verbose:
            print("\nFinding submission clusters...")
        
        # Build a graph of similar submissions
        similarity_graph = defaultdict(set)
        
        for result in self.similarity_results:
            if result['flagged']:
                similarity_graph[result['student1_id']].add(result['student2_id'])
                similarity_graph[result['student2_id']].add(result['student1_id'])
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for student_id in similarity_graph:
            if student_id not in visited:
                cluster = set()
                stack = [student_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster.add(current)
                        stack.extend(similarity_graph[current] - visited)
                
                if len(cluster) > 1:
                    clusters.append(cluster)
        
        # Print clusters
        if verbose and clusters:
            print(f"\nFound {len(clusters)} clusters of similar submissions:")
            for i, cluster in enumerate(clusters, 1):
                print(f"\nCluster {i} ({len(cluster)} students):")
                for student_id in cluster:
                    student = next((s for s in self.submissions if s.student_id == student_id), None)
                    if student:
                        print(f"  - {student.student_name} ({student.student_id})")
        elif verbose:
            print("No clusters found.")
        
        return clusters