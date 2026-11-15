#!/usr/bin/env python3
"""
Demonstration of automatic metadata extraction from file paths.

This script shows how the system automatically extracts:
- Author from folder structure
- Keywords from filename
- Topic inference from keywords
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metadata_extractor import (
    extract_author_from_path,
    extract_keywords_from_filename,
    infer_topic_from_keywords,
    extract_all_metadata,
    print_metadata
)


def demo_individual_functions():
    """Demonstrate each metadata extraction function individually."""
    print("\n" + "=" * 70)
    print("DEMO: Individual Metadata Extraction Functions")
    print("=" * 70)

    # Example file path
    file_path = "data/andrew_huberman/ketamine_benefits_depression_ptsd_neuroplasticity.txt"

    print(f"\nExample file: {file_path}\n")
    print("-" * 70)

    # Extract author
    author = extract_author_from_path(file_path)
    print(f"1. Author extracted from path:")
    print(f"   Input: {file_path}")
    print(f"   Output: {author}")

    # Extract keywords
    keywords = extract_keywords_from_filename("ketamine_benefits_depression_ptsd_neuroplasticity.txt")
    print(f"\n2. Keywords extracted from filename:")
    print(f"   Input: ketamine_benefits_depression_ptsd_neuroplasticity.txt")
    print(f"   Output: {keywords}")

    # Infer topic
    topic = infer_topic_from_keywords(keywords, file_path)
    print(f"\n3. Topic inferred from keywords:")
    print(f"   Input keywords: {keywords}")
    print(f"   Output topic: {topic}")

    print("-" * 70)


def demo_various_filenames():
    """Demonstrate metadata extraction with various filename patterns."""
    print("\n" + "=" * 70)
    print("DEMO: Automatic Metadata from Different Filename Patterns")
    print("=" * 70)

    test_cases = [
        # Neuroscience/Psychology
        "data/andrew_huberman/dopamine_motivation_focus_optimization.txt",
        "data/andrew_huberman/cold_exposure_benefits_immune_system.txt",
        "data/andrew_huberman/sleep_circadian_rhythm_melatonin.txt",

        # Technology/AI
        "data/lex_fridman/artificial_general_intelligence_future.txt",
        "data/lex_fridman/machine_learning_neural_networks.txt",

        # Health/Performance
        "data/tim_ferriss/productivity_performance_optimization.txt",
        "data/peter_attia/longevity_exercise_nutrition.txt",

        # Science
        "data/research_papers/clinical_trial_findings_study.txt",
    ]

    for file_path in test_cases:
        metadata = extract_all_metadata(file_path)
        print(f"\nFile: {metadata['filename']}")
        print(f"  Author:   {metadata['author']}")
        print(f"  Keywords: {metadata['keywords']}")
        print(f"  Topic:    {metadata['topic']}")


def demo_topic_inference():
    """Demonstrate how topics are inferred from different keyword combinations."""
    print("\n" + "=" * 70)
    print("DEMO: Topic Inference from Keywords")
    print("=" * 70)

    test_cases = [
        (["ketamine", "depression", "therapy"], "medicine"),
        (["sleep", "circadian", "melatonin"], "sleep"),
        (["ai", "machine", "learning"], "technology"),
        (["brain", "neuroplasticity", "learning"], "neuroscience"),
        (["focus", "productivity", "performance"], "performance"),
        (["nutrition", "diet", "supplements"], "health"),
        (["anxiety", "stress", "meditation"], "psychology"),
    ]

    for keywords, expected_topic in test_cases:
        inferred_topic = infer_topic_from_keywords(keywords)
        match = "✓" if inferred_topic == expected_topic else "✗"
        print(f"{match} Keywords: {keywords}")
        print(f"  → Inferred topic: '{inferred_topic}' (expected: '{expected_topic}')")


def demo_full_extraction():
    """Demonstrate complete metadata extraction from realistic file paths."""
    print("\n" + "=" * 70)
    print("DEMO: Complete Metadata Extraction")
    print("=" * 70)

    # Realistic examples
    examples = [
        "data/andrew_huberman/Ketamine Benefits and Risks for Depression, PTSD & Neuroplasticity - Huberman Lab Podcast.txt",
        "data/lex_fridman/Artificial Intelligence and the Future of Humanity.txt",
        "data/tim_ferriss/The-4-Hour-Body-Sleep-Optimization.txt",
        "data/joe_rogan/mushrooms_psychedelics_therapy_mental_health.txt",
    ]

    for file_path in examples:
        metadata = extract_all_metadata(file_path)
        print_metadata(metadata, f"File Path: {file_path}")


def demo_usage_in_code():
    """Show how to use metadata extraction in actual code."""
    print("\n" + "=" * 70)
    print("DEMO: Usage in Code")
    print("=" * 70)

    code_example = '''
# Example 1: Extract metadata for a single file
from metadata_extractor import extract_all_metadata

file_path = "data/andrew_huberman/sleep_optimization.txt"
metadata = extract_all_metadata(file_path)

print(f"Author: {metadata['author']}")
print(f"Keywords: {metadata['keywords']}")
print(f"Topic: {metadata['topic']}")

# Example 2: Use with RAG system
from rag_system import TranscriptRAG

rag = TranscriptRAG()
rag.add_transcript(file_path, metadata=metadata)

# Example 3: Batch process with custom metadata
from pathlib import Path

data_dir = Path("data/huberman_lab")
for file in data_dir.glob("*.txt"):
    metadata = extract_all_metadata(str(file))
    metadata['podcast'] = 'Huberman Lab'  # Add custom field
    rag.add_transcript(str(file), metadata=metadata)

# Example 4: Query by extracted metadata
results = rag.query(
    "What helps with sleep?",
    author_filter="Andrew Huberman"
)
    '''

    print(code_example)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("AUTOMATIC METADATA EXTRACTION - DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how metadata is automatically extracted from:")
    print("  1. Folder structure → Author name")
    print("  2. Filename → Keywords")
    print("  3. Keywords → Topic inference")

    # Run demos
    demo_individual_functions()
    demo_various_filenames()
    demo_topic_inference()
    demo_full_extraction()
    demo_usage_in_code()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The automatic metadata extraction system:

1. AUTHOR EXTRACTION
   - Looks at folder structure: data/AUTHOR_NAME/file.txt
   - Converts snake_case to Title Case: "andrew_huberman" → "Andrew Huberman"

2. KEYWORD EXTRACTION
   - Parses filename by separators: _, -, spaces
   - Filters out stop words and short words
   - Returns meaningful keywords in order

3. TOPIC INFERENCE
   - Matches keywords against topic dictionaries
   - Scores each topic based on keyword matches
   - Returns the best matching topic

4. USAGE
   Use smart_add_to_rag.py to automatically process files:

   # Add entire data directory
   python smart_add_to_rag.py --data-dir

   # Add specific folder
   python smart_add_to_rag.py --folder data/andrew_huberman

   # Add single file
   python smart_add_to_rag.py --file data/huberman/sleep.txt

All metadata is automatically extracted and stored in the RAG database!
    """)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
