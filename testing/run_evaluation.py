#!/usr/bin/env python3
"""
RAG System Evaluation Runner

Runs comprehensive evaluation of the RAG system using test datasets
and generates detailed markdown reports.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_system import TranscriptRAG
from llm_client import get_llm_client
from rag_evaluator import EvaluationRunner
from report_generator import MarkdownReportGenerator


def load_test_dataset(dataset_path: str) -> dict:
    """Load test dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def run_evaluation(
    dataset_path: str,
    output_dir: str = "testing/evaluation_results",
    num_questions: int = None,
    author_filter: str = None,
    verbose: bool = False,
    chroma_db: str = "./chroma_db",
    max_workers: int = 4
):
    """
    Run RAG system evaluation.

    Args:
        dataset_path: Path to test dataset JSON file
        output_dir: Directory to save evaluation results
        num_questions: Limit number of questions to evaluate (None = all)
        author_filter: Filter test cases by author (None = no filter)
        verbose: Show detailed progress
        chroma_db: Path to ChromaDB database directory (default: ./chroma_db)
        max_workers: Maximum number of parallel workers (default: 4, use 1 for sequential)
    """
    print("=" * 70)
    print("RAG System Evaluation Runner")
    print("=" * 70)
    print()

    # Load test dataset
    print(f"Loading test dataset from: {dataset_path}")
    try:
        dataset = load_test_dataset(dataset_path)
        test_cases = dataset['test_cases']
        print(f"✓ Loaded {len(test_cases)} test cases")
    except FileNotFoundError:
        print(f"Error: Dataset file not found: {dataset_path}")
        print("\nGenerate a dataset first using:")
        print("  uv run python src/test_generator.py --output testing/test_datasets/dataset_v1.json")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Filter by author if specified
    if author_filter:
        test_cases = [tc for tc in test_cases if tc.get('author_filter') == author_filter]
        print(f"✓ Filtered to {len(test_cases)} test cases for author: {author_filter}")

    # Limit number of questions if specified
    if num_questions:
        test_cases = test_cases[:num_questions]
        print(f"✓ Limited to {num_questions} test cases")

    if not test_cases:
        print("Error: No test cases to evaluate")
        return

    print()

    # Initialize RAG system and LLM clients
    print("Initializing RAG system and LLM clients...")
    print(f"  - Using ChromaDB directory: {chroma_db}")
    try:
        rag = TranscriptRAG(persist_directory=chroma_db)
        llm_eval = get_llm_client()  # For evaluation
        llm_response = get_llm_client()  # For response generation

        print("✓ RAG system initialized")
        print("✓ LLM clients initialized")

        # Show RAG collection stats
        stats = rag.get_collection_stats()
        print(f"  - Total chunks in collection: {stats['total_chunks']}")
        print(f"  - Available authors: {', '.join(rag.get_authors())}")
    except Exception as e:
        print(f"Error initializing systems: {e}")
        return

    print()

    # Initialize evaluation runner
    print("Setting up evaluation runner...")
    runner = EvaluationRunner(rag, llm_eval, llm_response)
    print("✓ Evaluation runner ready")
    print()

    # Run evaluation
    print(f"Running evaluation on {len(test_cases)} test cases...")
    if max_workers > 1:
        print(f"  - Running with {max_workers} parallel workers")
    else:
        print("  - Running sequentially (max_workers=1)")
    print("This may take several minutes depending on the number of test cases.")
    print()

    def progress_callback(current, total):
        """Display progress."""
        percent = (current / total) * 100
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\rProgress: [{bar}] {current}/{total} ({percent:.1f}%)", end='', flush=True)

    # Run batch evaluation
    results = runner.run_batch_evaluation(
        test_cases,
        progress_callback=progress_callback,
        max_workers=max_workers
    )

    print()  # New line after progress bar
    print()
    print("✓ Evaluation complete!")
    print()

    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(output_dir) / f"{timestamp}_evaluation.md"

    print(f"Generating markdown report...")
    report_gen = MarkdownReportGenerator()
    report_file = report_gen.generate_report(results, str(output_path))

    print(f"✓ Report generated: {report_file}")
    print()

    # Show summary
    valid_results = [r for r in results if 'error' not in r]
    error_count = len(results) - len(valid_results)

    if valid_results:
        avg_retrieval = sum(r['retrieval']['avg_relevance'] for r in valid_results) / len(valid_results)
        avg_response = sum(r['response']['avg_score'] for r in valid_results) / len(valid_results)
        hallucination_count = sum(1 for r in valid_results if r['hallucinations']['has_hallucinations'])
        hallucination_rate = (hallucination_count / len(valid_results)) * 100

        print("=" * 70)
        print("Evaluation Summary")
        print("=" * 70)
        print(f"Total Test Cases: {len(results)}")
        print(f"Successful: {len(valid_results)}")
        print(f"Errors: {error_count}")
        print()
        print(f"Average Retrieval Quality: {avg_retrieval:.2f}/5.0")
        print(f"Average Response Quality: {avg_response:.2f}/5.0")
        print(f"Hallucination Rate: {hallucination_rate:.1f}%")
        print("=" * 70)
        print()
        print(f"📊 View full report: {report_file}")
        print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive RAG system evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation with default dataset
  uv run python testing/run_evaluation.py

  # Run with specific dataset
  uv run python testing/run_evaluation.py --dataset testing/test_datasets/dataset_v1.json

  # Run limited test (first 10 questions)
  uv run python testing/run_evaluation.py --num-questions 10

  # Filter by author
  uv run python testing/run_evaluation.py --author-filter "Andrew Huberman"

  # Use custom ChromaDB database
  uv run python testing/run_evaluation.py --chroma-db ./custom_chroma_db

  # Run with 8 parallel workers (faster for large datasets)
  uv run python testing/run_evaluation.py --workers 8

  # Run sequentially (1 worker, slower but uses less resources)
  uv run python testing/run_evaluation.py --workers 1

  # Verbose mode
  uv run python testing/run_evaluation.py --num-questions 5 --verbose
        """
    )

    parser.add_argument(
        "--dataset",
        default="testing/test_datasets/dataset_v1.json",
        help="Path to test dataset JSON file (default: testing/test_datasets/dataset_v1.json)"
    )

    parser.add_argument(
        "--output",
        default="testing/evaluation_results",
        help="Output directory for evaluation reports (default: testing/evaluation_results/)"
    )

    parser.add_argument(
        "--num-questions",
        type=int,
        help="Limit number of questions to evaluate (default: all)"
    )

    parser.add_argument(
        "--author-filter",
        help="Filter test cases by author name"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress for each test case"
    )

    parser.add_argument(
        "--chroma-db",
        default="./chroma_db",
        help="Path to ChromaDB database directory (default: ./chroma_db)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for evaluation (default: 4, use 1 for sequential)"
    )

    args = parser.parse_args()

    # Run evaluation
    run_evaluation(
        dataset_path=args.dataset,
        output_dir=args.output,
        num_questions=args.num_questions,
        author_filter=args.author_filter,
        verbose=args.verbose,
        chroma_db=args.chroma_db,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
