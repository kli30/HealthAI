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
# Standardized 8 topics for health/wellness content
TOPIC_KEYWORDS = {
    'neuroscience': [
        'brain', 'neural', 'neuron', 'dopamine', 'serotonin', 'cortisol',
        'neuroplasticity', 'cognitive', 'memory', 'learning', 'attention',
        'consciousness', 'neurotransmitter', 'synapse', 'cerebral'
    ],
    'health': [
        'health', 'wellness', 'healthy', 'immune', 'immunity', 'disease',
        'prevention', 'longevity', 'aging', 'metabolic', 'metabolism'
    ],
    'psychology': [
        'mental', 'emotion', 'behavior', 'mood', 'anxiety', 'depression',
        'stress', 'therapy', 'psychological', 'mindfulness', 'meditation',
        'ptsd', 'trauma', 'adhd'
    ],
    'sleep': [
        'sleep', 'circadian', 'insomnia', 'rem', 'melatonin', 'adenosine',
        'nap', 'rest', 'wake', 'drowsy'
    ],
    'performance': [
        'performance', 'productivity', 'focus', 'concentration', 'optimization',
        'enhancement', 'peak', 'flow', 'efficiency', 'energy'
    ],
    'medicine': [
        'medical', 'clinical', 'drug', 'medication', 'treatment', 'therapy',
        'ketamine', 'pharmaceutical', 'diagnosis', 'patient', 'doctor',
        'prescription', 'dose', 'protocol'
    ],
    'nutrition': [
        'nutrition', 'diet', 'food', 'eating', 'meal', 'nutrient', 'vitamin',
        'supplement', 'fasting', 'keto', 'carb', 'protein', 'fat', 'calorie',
        'macronutrient', 'micronutrient'
    ],
    'exercise': [
        'exercise', 'workout', 'training', 'fitness', 'strength', 'cardio',
        'muscle', 'weight', 'lift', 'run', 'physical', 'movement', 'activity',
        'hiit', 'resistance', 'endurance'
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


def extract_author_from_path(file_path: str) -> Optional[str]:
    """
    Extract author name from the parent folder of the transcript file.

    Args:
        file_path: Full path to the file

    Returns:
        Author name in Title Case, or None if not found

    Examples:
        'data/andrew_huberman/transcript.txt' -> 'Andrew Huberman'
        'data/lex_fridman/episode_123.txt' -> 'Lex Fridman'
        '/home/user/data/tim_ferriss/file.txt' -> 'Tim Ferriss'
        'any_folder/jane_doe/transcript.txt' -> 'Jane Doe'
    """
    path = Path(file_path)

    # Use parent directory as author
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
    custom_metadata: Optional[Dict[str, str]] = None
) -> Dict[str, any]:
    """
    Extract all metadata from a file path automatically.

    The author is inferred from the parent folder name of the transcript file.

    Args:
        file_path: Full path to the file
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
    author = extract_author_from_path(file_path)
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
        "/home/user/healthAI/data/Lex Fridman/Artificial Intelligence and the Future of Humanity.txt",
        "/home/user/healthAI/data/Tim Ferriss/The-4-Hour-Body-Sleep-Optimization.txt",
        "/home/user/healthAI/data/Neuroscience Podcasts/Brain Plasticity Learning.txt",
        "/home/user/healthAI/data/Andrew Huberman/data/andrew_huberman/xWelcome to the Huberman Lab Podcast.txt",
    ]

    for test_file in test_files:
        metadata = extract_all_metadata(test_file)
        print_metadata(metadata, f"File: {test_file}")
