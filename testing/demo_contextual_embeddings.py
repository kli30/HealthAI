"""
Demo script showing the benefits of contextual embeddings.

This script demonstrates how adding contextual information (author, topic, keywords)
to chunks before embedding improves retrieval accuracy.
"""

import sys
sys.path.insert(0, 'src')

from rag_system import TranscriptRAG
import tempfile
import os


def create_sample_transcripts():
    """Create sample transcript files for demonstration."""
    transcripts = {
        "sleep_huberman.txt": """
Sleep is fundamental to cognitive function and overall health. Dr. Andrew Huberman
discusses how sleep cycles work, including the importance of REM sleep and deep sleep.
The circadian rhythm regulates when we feel sleepy and alert. Light exposure in the
morning helps set your circadian clock. Avoiding blue light at night can improve
sleep quality. Temperature also plays a crucial role - cooler temperatures promote
better sleep. Regular sleep schedules help maintain healthy sleep patterns.
        """,
        "sleep_walker.txt": """
Sleep research by Dr. Matthew Walker shows that sleep deprivation has serious health
consequences. Lack of sleep affects memory consolidation and learning. The glymphatic
system cleans the brain during sleep. Chronic sleep loss increases risk of
neurodegenerative diseases. Both REM and non-REM sleep are essential. Sleep helps
process emotions and consolidate memories. Adults need 7-9 hours of sleep per night.
        """,
        "exercise_huberman.txt": """
Dr. Andrew Huberman explains how exercise affects the brain and body. Physical activity
releases endorphins and dopamine. Resistance training builds muscle and bone density.
Cardiovascular exercise improves heart health and endurance. Zone 2 cardio is optimal
for fat burning. High-intensity interval training (HIIT) provides time-efficient workouts.
Exercise improves mood and reduces anxiety. Regular movement is essential for longevity.
        """
    }

    temp_dir = tempfile.mkdtemp()
    file_paths = {}

    for filename, content in transcripts.items():
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content.strip())
        file_paths[filename] = filepath

    return temp_dir, file_paths


def demo_without_context():
    """Demonstrate RAG without contextual embeddings."""
    print("\n" + "=" * 80)
    print("DEMO 1: WITHOUT Contextual Embeddings")
    print("=" * 80)

    temp_dir, file_paths = create_sample_transcripts()

    try:
        # Create RAG system with temporary collection
        rag = TranscriptRAG(collection_name="demo_no_context", persist_directory=temp_dir)

        # Add transcripts WITHOUT contextual embeddings
        rag.add_transcript(
            file_paths["sleep_huberman.txt"],
            metadata={"author": "Andrew Huberman"},
            use_contextual_embeddings=False  # Disabled
        )

        rag.add_transcript(
            file_paths["sleep_walker.txt"],
            metadata={"author": "Matthew Walker"},
            use_contextual_embeddings=False  # Disabled
        )

        rag.add_transcript(
            file_paths["exercise_huberman.txt"],
            metadata={"author": "Andrew Huberman"},
            use_contextual_embeddings=False  # Disabled
        )

        # Query: User asks about Huberman's sleep advice
        query = "What does Huberman say about sleep?"
        print(f"\nQuery: '{query}'")
        print("\nTop 2 Results:")

        results = rag.query(query, n_results=2)
        for i, (chunk, metadata) in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Author: {metadata.get('author', 'Unknown')}")
            print(f"Text: {chunk[:150]}...")

        return rag, temp_dir

    except Exception as e:
        print(f"Error: {e}")
        import shutil
        shutil.rmtree(temp_dir)
        raise


def demo_with_context():
    """Demonstrate RAG WITH contextual embeddings."""
    print("\n" + "=" * 80)
    print("DEMO 2: WITH Contextual Embeddings")
    print("=" * 80)

    temp_dir, file_paths = create_sample_transcripts()

    try:
        # Create RAG system with temporary collection
        rag = TranscriptRAG(collection_name="demo_with_context", persist_directory=temp_dir)

        # Add transcripts WITH contextual embeddings
        rag.add_transcript(
            file_paths["sleep_huberman.txt"],
            metadata={
                "author": "Andrew Huberman",
                "keywords": ["sleep", "circadian", "rhythm"]
            },
            use_contextual_embeddings=True  # Enabled!
        )

        rag.add_transcript(
            file_paths["sleep_walker.txt"],
            metadata={
                "author": "Matthew Walker",
                "keywords": ["sleep", "memory", "health"]
            },
            use_contextual_embeddings=True  # Enabled!
        )

        rag.add_transcript(
            file_paths["exercise_huberman.txt"],
            metadata={
                "author": "Andrew Huberman",
                "keywords": ["exercise", "fitness", "training"]
            },
            use_contextual_embeddings=True  # Enabled!
        )

        # Query: User asks about Huberman's sleep advice
        query = "What does Huberman say about sleep?"
        print(f"\nQuery: '{query}'")
        print("\nTop 2 Results:")

        results = rag.query(query, n_results=2, return_original=True)
        for i, (chunk, metadata) in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Author: {metadata.get('author', 'Unknown')}")
            print(f"Text: {chunk[:150]}...")

        # Show what the embedded text looks like
        print("\n" + "=" * 80)
        print("Example of Contextual Chunk (what gets embedded):")
        print("=" * 80)
        sample_metadata = {
            "author": "Andrew Huberman",
            "keywords": ["sleep", "circadian"],
            "source": "sleep_huberman.txt"
        }
        sample_chunk = "Sleep is fundamental to cognitive function..."
        contextual = rag.create_contextual_chunk(sample_chunk, sample_metadata)
        print(contextual)

        return rag, temp_dir

    except Exception as e:
        print(f"Error: {e}")
        import shutil
        shutil.rmtree(temp_dir)
        raise


def cleanup(temp_dir):
    """Clean up temporary files."""
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Contextual Embeddings Demo")
    print("=" * 80)
    print("\nThis demo shows how contextual embeddings improve retrieval accuracy.")
    print("By adding metadata (author, keywords, source) to chunks before embedding,")
    print("the embedding model better understands the context and returns more relevant results.")

    # Demo without contextual embeddings
    rag1, temp_dir1 = demo_without_context()
    cleanup(temp_dir1)

    # Demo with contextual embeddings
    rag2, temp_dir2 = demo_with_context()
    cleanup(temp_dir2)

    print("\n" + "=" * 80)
    print("Key Takeaway:")
    print("=" * 80)
    print("With contextual embeddings, the system can better distinguish between:")
    print("- Different authors discussing the same topic")
    print("- Same author discussing different topics")
    print("- Specific keywords and themes")
    print("\nThis leads to more accurate and relevant search results!")
    print("=" * 80 + "\n")
