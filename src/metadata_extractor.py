#!/usr/bin/env python3
"""
Metadata extraction utilities for automatic metadata generation from file paths.

This module provides functions to extract author, keywords, and topics from
file paths and filenames automatically.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional


# Common stop words to filter from keywords
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
    'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
    'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'huberman', 'lab', 'podcast', 'transcript', 'episode', 'interview'
}

# Topic mapping based on common keywords
TOPIC_KEYWORDS = {
    'neuroscience': [
        'brain', 'neural', 'neuron', 'dopamine', 'serotonin', 'cortisol',
        'neuroplasticity', 'cognitive', 'memory', 'learning', 'attention',
        'consciousness', 'neurotransmitter', 'synapse', 'cerebral'
    ],
    'health': [
        'fitness', 'exercise', 'nutrition', 'diet', 'supplement', 'vitamin',
        'wellness', 'healthy', 'immune', 'metabolism', 'disease', 'treatment'
    ],
    'psychology': [
        'mental', 'emotion', 'behavior', 'mood', 'anxiety', 'depression',
        'stress', 'therapy', 'psychological', 'mindfulness', 'meditation',
        'ptsd', 'trauma', 'adhd'
    ],
    'sleep': [
        'sleep', 'circadian', 'insomnia', 'rem', 'melatonin', 'adenosine'
    ],
    'performance': [
        'performance', 'productivity', 'focus', 'concentration', 'optimization',
        'enhancement', 'peak', 'flow', 'training'
    ],
    'technology': [
        'ai', 'artificial', 'intelligence', 'machine', 'learning', 'computer',
        'software', 'algorithm', 'data', 'neural', 'network', 'programming'
    ],
    'science': [
        'research', 'study', 'scientific', 'experiment', 'evidence', 'clinical',
        'trial', 'findings', 'discovery'
    ],
    'medicine': [
        'medical', 'clinical', 'drug', 'medication', 'treatment', 'therapy',
        'ketamine', 'pharmaceutical', 'diagnosis', 'patient', 'doctor'
    ]
}


def format_name(name_str: str) -> str:
    """
    Format a name string from snake_case or kebab-case to Title Case.

    Args:
        name_str: String in snake_case or kebab-case format

    Returns:
        Formatted name in Title Case

    Examples:
        'andrew_huberman' -> 'Andrew Huberman'
        'lex-fridman' -> 'Lex Fridman'
        'tim_ferriss_show' -> 'Tim Ferriss Show'
    """
    # Replace underscores and hyphens with spaces
    name = name_str.replace('_', ' ').replace('-', ' ')

    # Title case
    name = name.title()

    return name


def extract_author_from_path(file_path: str, data_dir: str = "data") -> Optional[str]:
    """
    Extract author name from the subfolder structure under data directory.

    Args:
        file_path: Full path to the file
        data_dir: Name of the data directory (default: "data")

    Returns:
        Author name in Title Case, or None if not found

    Examples:
        'data/andrew_huberman/transcript.txt' -> 'Andrew Huberman'
        'data/lex_fridman/episode_123.txt' -> 'Lex Fridman'
        '/home/user/data/tim_ferriss/file.txt' -> 'Tim Ferriss'
    """
    path = Path(file_path)
    parts = path.parts

    # Find the data directory in the path
    try:
        data_index = parts.index(data_dir)
        # The next part after 'data' should be the author folder
        if data_index + 1 < len(parts):
            author_folder = parts[data_index + 1]
            return format_name(author_folder)
    except ValueError:
        # 'data' not in path, try to use parent directory
        if path.parent.name and path.parent.name != '.':
            return format_name(path.parent.name)

    return None


def extract_keywords_from_filename(filename: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from filename by parsing and filtering.

    Args:
        filename: The filename (with or without extension)
        max_keywords: Maximum number of keywords to return

    Returns:
        List of extracted keywords

    Examples:
        'ketamine_benefits_depression_ptsd.txt' -> ['ketamine', 'benefits', 'depression', 'ptsd']
        'Sleep-Optimization-Guide.txt' -> ['sleep', 'optimization', 'guide']
    """
    # Remove file extension
    name = Path(filename).stem

    # Replace common separators with spaces
    name = re.sub(r'[_\-\.\,\;\:\|]', ' ', name)

    # Remove special characters and numbers at the start/end
    name = re.sub(r'^[\d\W]+|[\d\W]+$', '', name)

    # Split into words
    words = name.lower().split()

    # Filter out stop words and short words
    keywords = [
        word for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]

    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)

    return unique_keywords[:max_keywords]


def infer_topic_from_keywords(keywords: List[str], filename: str = "") -> str:
    """
    Infer the main topic from keywords using a scoring system.

    Args:
        keywords: List of keywords extracted from filename
        filename: Optional full filename for additional context

    Returns:
        Inferred topic name

    Examples:
        ['ketamine', 'depression', 'therapy'] -> 'medicine'
        ['sleep', 'optimization', 'circadian'] -> 'sleep'
        ['focus', 'productivity', 'performance'] -> 'performance'
    """
    if not keywords:
        return "general"

    # Convert keywords to lowercase for matching
    keywords_lower = [k.lower() for k in keywords]

    # Also check filename
    filename_lower = filename.lower()

    # Score each topic based on keyword matches
    topic_scores = {}

    for topic, topic_keywords in TOPIC_KEYWORDS.items():
        score = 0

        # Check keywords
        for keyword in keywords_lower:
            for topic_keyword in topic_keywords:
                if topic_keyword in keyword or keyword in topic_keyword:
                    score += 2  # Exact or partial match in keywords

        # Check filename for additional context
        for topic_keyword in topic_keywords:
            if topic_keyword in filename_lower:
                score += 1  # Match in full filename

        topic_scores[topic] = score

    # Get the topic with highest score
    if topic_scores:
        max_score = max(topic_scores.values())
        if max_score > 0:
            # Return the topic with the highest score
            best_topic = max(topic_scores.items(), key=lambda x: x[1])[0]
            return best_topic

    # Default fallback: use first keyword as topic if it's substantial
    if keywords and len(keywords[0]) > 4:
        return keywords[0]

    return "general"


def extract_all_metadata(
    file_path: str,
    data_dir: str = "data",
    custom_metadata: Optional[Dict[str, str]] = None
) -> Dict[str, any]:
    """
    Extract all metadata from a file path automatically.

    Args:
        file_path: Full path to the file
        data_dir: Name of the data directory (default: "data")
        custom_metadata: Optional dictionary of custom metadata to merge

    Returns:
        Dictionary containing all extracted metadata

    Examples:
        extract_all_metadata('data/andrew_huberman/ketamine_depression.txt')
        -> {
            'author': 'Andrew Huberman',
            'keywords': ['ketamine', 'depression'],
            'topic': 'medicine',
            'filename': 'ketamine_depression.txt',
            'source_path': 'data/andrew_huberman'
        }
    """
    path = Path(file_path)

    # Extract components
    author = extract_author_from_path(file_path, data_dir)
    keywords = extract_keywords_from_filename(path.name)
    topic = infer_topic_from_keywords(keywords, path.name)

    # Build metadata dictionary
    metadata = {
        'author': author or 'Unknown',
        'keywords': ', '.join(keywords) if keywords else '',
        'topic': topic,
        'filename': path.name,
        'source_path': str(path.parent)
    }

    # Add custom metadata
    if custom_metadata:
        metadata.update(custom_metadata)

    return metadata


def print_metadata(metadata: Dict[str, any], title: str = "Extracted Metadata"):
    """
    Print metadata in a formatted way.

    Args:
        metadata: Metadata dictionary
        title: Title for the output
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for key, value in metadata.items():
        print(f"  {key.capitalize()}: {value}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Example usage and tests
    print("Metadata Extractor - Example Usage")
    print("=" * 60)

    # Test cases
    test_files = [
        "data/Andrew Huberman/Adderall, Stimulants & Modafinil for ADHD Short- & Long-Term Effects   Huberman Lab Podcast.txt",
        "data/Lex Fridman/Artificial Intelligence and the Future of Humanity.txt",
        "data/Tim Ferriss/The-4-Hour-Body-Sleep-Optimization.txt",
        "data/Neuroscience Podcasts/Brain Plasticity Learning.txt",
    ]

    for test_file in test_files:
        metadata = extract_all_metadata(test_file)
        print_metadata(metadata, f"File: {test_file}")
