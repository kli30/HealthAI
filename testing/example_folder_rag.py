#!/usr/bin/env python3
"""
Example script demonstrating how to use the folder-based RAG system
with author metadata.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_system import TranscriptRAG
from add_folder_to_rag import add_folder_to_rag


def example_usage():
    """Demonstrate the folder-based RAG system."""

    print("=" * 70)
    print("Example: Using Folder-based RAG with Author Metadata")
    print("=" * 70)
    print()

    # Example 1: Add a folder of transcripts with author metadata
    print("Example 1: Adding transcripts from a folder")
    print("-" * 70)
    print("""
# Add Huberman Lab transcripts
add_folder_to_rag(
    folder_path="data/huberman_transcripts",
    author="Andrew Huberman",
    podcast="Huberman Lab",
    topic="neuroscience"
)

# Add Lex Fridman podcast transcripts
add_folder_to_rag(
    folder_path="data/lex_transcripts",
    author="Lex Fridman",
    podcast="Lex Fridman Podcast",
    topic="technology"
)
    """)

    # Example 2: Query with author filtering
    print("\nExample 2: Querying with author filter")
    print("-" * 70)
    print("""
# Initialize RAG system
rag = TranscriptRAG()

# Query only Huberman's transcripts
results = rag.query(
    "What are the benefits of cold exposure?",
    n_results=3,
    author_filter="Andrew Huberman"
)

# Query only Lex's transcripts
results = rag.query(
    "What is artificial general intelligence?",
    n_results=3,
    author_filter="Lex Fridman"
)

# Query all transcripts (no filter)
results = rag.query(
    "What is the future of AI?",
    n_results=5
)
    """)

    # Example 3: Get collection statistics
    print("\nExample 3: View collection statistics")
    print("-" * 70)
    print("""
# Initialize RAG system
rag = TranscriptRAG()

# Print statistics
rag.print_collection_stats()

# Get list of all authors
authors = rag.get_authors()
print("Available authors:", authors)

# Get detailed statistics
stats = rag.get_collection_stats()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Chunks by author: {stats['chunks_by_author']}")
    """)

    # Example 4: Practical demonstration (if data exists)
    print("\nExample 4: Practical Demonstration")
    print("-" * 70)

    try:
        # Initialize RAG system
        rag = TranscriptRAG()

        # Print current statistics
        print("\nCurrent RAG Collection:")
        rag.print_collection_stats()

        # List available authors
        authors = rag.get_authors()
        if authors:
            print(f"Available authors: {', '.join(authors)}\n")

            # Example query with author filter
            if len(authors) > 0:
                print(f"Example query filtered by author '{authors[0]}':")
                print("-" * 70)
                results = rag.query(
                    "What are the main topics discussed?",
                    n_results=2,
                    author_filter=authors[0]
                )

                for i, (text, metadata) in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"  Author: {metadata.get('author', 'Unknown')}")
                    print(f"  Source: {metadata.get('source', 'Unknown')}")
                    print(f"  Text preview: {text[:200]}...")
                    print()
        else:
            print("No transcripts found in the collection.")
            print("Add transcripts using: python add_folder_to_rag.py")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("\nNote: Make sure you have added transcripts to the RAG system first.")
        print("Use: python add_folder_to_rag.py --help")

    print("\n" + "=" * 70)
    print("For more information, see CLAUDE.md")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()
