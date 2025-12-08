#!/usr/bin/env python3
"""
Add a folder of transcripts to the RAG vector database.

This script scans a folder for transcript files and adds them to the
ChromaDB vector database with author metadata.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Dict
from rag_system import TranscriptRAG


def add_folder_to_rag(
    folder_path: str,
    author: str,
    file_extension: str = ".txt",
    topic: Optional[str] = None,
    podcast: Optional[str] = None,
    additional_metadata: Optional[Dict[str, str]] = None,
    collection_name: str = "transcripts",
    persist_directory: str = "./chroma_db"
):
    """
    Add all transcript files from a folder to the RAG system.

    Args:
        folder_path: Path to the folder containing transcript files
        author: Author of the transcripts (e.g., "Andrew Huberman", "Lex Fridman")
        file_extension: File extension to look for (default: ".txt")
        topic: Optional topic/category for the transcripts
        podcast: Optional podcast name
        additional_metadata: Optional dictionary of additional metadata fields
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory to persist the vector database

    Returns:
        Number of files successfully added
    """
    # Initialize RAG system
    print(f"Initializing RAG system...")
    print(f"Collection: {collection_name}")
    print(f"Persist directory: {persist_directory}\n")
    rag = TranscriptRAG(collection_name=collection_name, persist_directory=persist_directory)

    # Get all transcript files from the folder
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    # Find all files with the specified extension
    transcript_files = list(folder.glob(f"*{file_extension}"))

    if not transcript_files:
        print(f"No {file_extension} files found in {folder_path}")
        return 0

    print(f"Found {len(transcript_files)} transcript file(s) in {folder_path}\n")

    # Process each file
    added_count = 0
    for file_path in transcript_files:
        print(f"Processing: {file_path.name}")

        # Build metadata for this file
        metadata = {
            "author": author,
            "filename": file_path.name,
        }

        # Add optional metadata fields
        if topic:
            metadata["topic"] = topic
        if podcast:
            metadata["podcast"] = podcast
        if additional_metadata:
            metadata.update(additional_metadata)

        try:
            # Add transcript to RAG system
            rag.add_transcript(str(file_path), metadata=metadata)
            added_count += 1
            print(f"  ✓ Successfully added\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    # Print summary
    print("=" * 60)
    print(f"Summary:")
    print(f"  Total files found: {len(transcript_files)}")
    print(f"  Successfully added: {added_count}")
    print(f"  Failed: {len(transcript_files) - added_count}")
    print(f"  Total chunks in collection: {rag.collection.count()}")
    print("=" * 60)

    return added_count


def interactive_mode():
    """Run the script in interactive mode, prompting for inputs."""
    print("=" * 60)
    print("Add Folder to RAG - Interactive Mode")
    print("=" * 60)
    print()

    # Get folder path
    folder_path = input("Enter the folder path containing transcripts: ").strip()
    if not folder_path:
        print("Error: Folder path is required")
        return

    # Get author
    author = input("Enter the author name (e.g., Andrew Huberman): ").strip()
    if not author:
        print("Error: Author name is required")
        return

    # Get optional metadata
    topic = input("Enter topic/category (optional, press Enter to skip): ").strip() or None
    podcast = input("Enter podcast name (optional, press Enter to skip): ").strip() or None
    file_extension = input("Enter file extension [.txt]: ").strip() or ".txt"

    # Confirm
    print("\n" + "=" * 60)
    print("Configuration:")
    print(f"  Folder: {folder_path}")
    print(f"  Author: {author}")
    print(f"  Topic: {topic or 'Not specified'}")
    print(f"  Podcast: {podcast or 'Not specified'}")
    print(f"  File extension: {file_extension}")
    print("=" * 60)
    confirm = input("\nProceed? (yes/no): ").strip().lower()

    if confirm not in ['yes', 'y']:
        print("Cancelled.")
        return

    print()
    try:
        add_folder_to_rag(
            folder_path=folder_path,
            author=author,
            file_extension=file_extension,
            topic=topic,
            podcast=podcast
        )
    except Exception as e:
        print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Add a folder of transcripts to the RAG vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python add_folder_to_rag.py

  # Add Huberman Lab transcripts
  python add_folder_to_rag.py --folder data/huberman --author "Andrew Huberman" --podcast "Huberman Lab"

  # Add transcripts with specific topic
  python add_folder_to_rag.py --folder data/transcripts --author "Lex Fridman" --podcast "Lex Fridman Podcast" --topic "AI"

  # Add markdown files instead of txt
  python add_folder_to_rag.py --folder data/notes --author "Author Name" --extension .md
        """
    )

    parser.add_argument(
        '--folder', '-f',
        type=str,
        help='Path to folder containing transcript files'
    )

    parser.add_argument(
        '--author', '-a',
        type=str,
        help='Author of the transcripts (e.g., "Andrew Huberman")'
    )

    parser.add_argument(
        '--topic', '-t',
        type=str,
        help='Topic or category for the transcripts'
    )

    parser.add_argument(
        '--podcast', '-p',
        type=str,
        help='Podcast name'
    )

    parser.add_argument(
        '--extension', '-e',
        type=str,
        default='.txt',
        help='File extension to look for (default: .txt)'
    )

    parser.add_argument(
        '--collection',
        type=str,
        default='transcripts',
        help='ChromaDB collection name (default: transcripts)'
    )

    parser.add_argument(
        '--persist-dir',
        type=str,
        default='./chroma_db',
        help='Directory to persist the vector database (default: ./chroma_db)'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )

    args = parser.parse_args()

    # If no arguments or interactive flag, run interactive mode
    if args.interactive or (not args.folder and not args.author):
        interactive_mode()
        return

    # Validate required arguments
    if not args.folder:
        parser.error("--folder is required (or use --interactive)")
    if not args.author:
        parser.error("--author is required (or use --interactive)")

    try:
        add_folder_to_rag(
            folder_path=args.folder,
            author=args.author,
            file_extension=args.extension,
            topic=args.topic,
            podcast=args.podcast,
            collection_name=args.collection,
            persist_directory=args.persist_dir
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
