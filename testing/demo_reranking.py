"""
Demo script showing reranking functionality in the RAG system.

This script demonstrates:
1. How to enable/disable reranking
2. Comparison of results with and without reranking
3. Viewing relevance scores
"""

import sys
sys.path.insert(0, 'src')

from rag_system import TranscriptRAG


def demo_basic_reranking():
    """Demonstrate basic reranking functionality."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Reranking")
    print("=" * 80)

    # Initialize RAG with reranking enabled (default)
    print("\nInitializing RAG system with reranking enabled...")
    rag_with_rerank = TranscriptRAG(use_reranking=True)

    # Check if collection has data
    stats = rag_with_rerank.get_collection_stats()
    if stats['total_chunks'] == 0:
        print("\n⚠️  No data in collection. Please add transcripts first:")
        print("   python src/smart_add_to_rag.py --data-dir")
        return

    # Query with reranking
    query = "What are the effects of ketamine on depression?"
    print(f"\nQuery: '{query}'")
    print("\nRetrieving results with reranking...")

    results = rag_with_rerank.query(query, n_results=3, return_scores=True)

    print("\nTop 3 results (with relevance scores):")
    for i, (text, metadata, score) in enumerate(results, 1):
        source = metadata.get('source', 'unknown')
        author = metadata.get('author', 'unknown')
        print(f"\n{i}. Score: {score:.4f} | Author: {author} | Source: {source}")
        print(f"   Preview: {text[:150]}...")


def demo_comparison():
    """Compare results with and without reranking."""
    print("\n" + "=" * 80)
    print("DEMO 2: Comparison - With vs Without Reranking")
    print("=" * 80)

    # Initialize both versions
    print("\nInitializing RAG systems...")
    rag_with_rerank = TranscriptRAG(use_reranking=True)
    rag_without_rerank = TranscriptRAG(use_reranking=False)

    # Check if collection has data
    stats = rag_with_rerank.get_collection_stats()
    if stats['total_chunks'] == 0:
        print("\n⚠️  No data in collection. Please add transcripts first:")
        print("   python src/smart_add_to_rag.py --data-dir")
        return

    query = "How does sleep affect mental health?"
    print(f"\nQuery: '{query}'")

    # Without reranking
    print("\n" + "-" * 80)
    print("WITHOUT RERANKING (standard semantic search):")
    print("-" * 80)
    results_no_rerank = rag_without_rerank.query(query, n_results=3)
    for i, (text, metadata) in enumerate(results_no_rerank, 1):
        source = metadata.get('source', 'unknown')
        author = metadata.get('author', 'unknown')
        print(f"\n{i}. Author: {author} | Source: {source}")
        print(f"   Preview: {text[:150]}...")

    # With reranking
    print("\n" + "-" * 80)
    print("WITH RERANKING (cross-encoder refinement):")
    print("-" * 80)
    results_with_rerank = rag_with_rerank.query(query, n_results=3, return_scores=True)
    for i, (text, metadata, score) in enumerate(results_with_rerank, 1):
        source = metadata.get('source', 'unknown')
        author = metadata.get('author', 'unknown')
        print(f"\n{i}. Score: {score:.4f} | Author: {author} | Source: {source}")
        print(f"   Preview: {text[:150]}...")


def demo_context_with_scores():
    """Demonstrate getting formatted context with relevance scores."""
    print("\n" + "=" * 80)
    print("DEMO 3: Formatted Context with Relevance Scores")
    print("=" * 80)

    rag = TranscriptRAG(use_reranking=True)

    # Check if collection has data
    stats = rag.get_collection_stats()
    if stats['total_chunks'] == 0:
        print("\n⚠️  No data in collection. Please add transcripts first:")
        print("   python src/smart_add_to_rag.py --data-dir")
        return

    query = "What are practical tips for improving focus?"
    print(f"\nQuery: '{query}'")

    # Get context with scores visible
    print("\nFormatted context (with relevance scores):")
    print("-" * 80)
    context = rag.get_context_for_query(query, n_results=3, show_scores=True)
    print(context)


def demo_author_filtering_with_reranking():
    """Demonstrate author filtering combined with reranking."""
    print("\n" + "=" * 80)
    print("DEMO 4: Author Filtering + Reranking")
    print("=" * 80)

    rag = TranscriptRAG(use_reranking=True)

    # Check if collection has data and get authors
    stats = rag.get_collection_stats()
    if stats['total_chunks'] == 0:
        print("\n⚠️  No data in collection. Please add transcripts first:")
        print("   python src/smart_add_to_rag.py --data-dir")
        return

    authors = rag.get_authors()
    if not authors:
        print("\n⚠️  No author metadata found in collection.")
        return

    print(f"\nAvailable authors: {', '.join(authors)}")

    # Use first author for demo
    target_author = authors[0]
    query = "What are the key insights about health and performance?"

    print(f"\nQuery: '{query}'")
    print(f"Filtering by author: {target_author}")

    results = rag.query(query, n_results=3, author_filter=target_author, return_scores=True)

    print(f"\nTop 3 results from {target_author} (with reranking):")
    for i, (text, metadata, score) in enumerate(results, 1):
        source = metadata.get('source', 'unknown')
        print(f"\n{i}. Score: {score:.4f} | Source: {source}")
        print(f"   Preview: {text[:150]}...")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("RAG RERANKING DEMO")
    print("=" * 80)
    print("\nThis demo shows how cross-encoder reranking improves retrieval quality.")
    print("The system retrieves more candidates (20+) then reranks them using a")
    print("cross-encoder model for better relevance.\n")

    try:
        demo_basic_reranking()
        demo_comparison()
        demo_context_with_scores()
        demo_author_filtering_with_reranking()

        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("• Reranking is enabled by default (use_reranking=True)")
        print("• Cross-encoder provides more accurate relevance scores")
        print("• Works seamlessly with author filtering")
        print("• Scores can be displayed for debugging and evaluation")
        print("• To disable: TranscriptRAG(use_reranking=False)")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
