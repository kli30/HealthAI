#!/usr/bin/env python3
"""
Smart RAG file/folder adder with automatic metadata extraction.

This script automatically extracts metadata from file paths and filenames:
- Author: from parent folder name
- Keywords: parsed from filename
- Topic: inferred from keywords and filename
"""

import os
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, List
from rag_system import TranscriptRAG
from metadata_extractor import extract_all_metadata, print_metadata


def add_file_with_auto_metadata(
    file_path: str,
    rag: TranscriptRAG,
    custom_metadata: Optional[Dict[str, str]] = None,
    verbose: bool = True,
    use_contextual_embeddings: bool = True,
) -> bool:
    """
    Add a single file to RAG with automatically extracted metadata.

    Args:
        file_path: Path to the file
        rag: TranscriptRAG instance
        custom_metadata: Optional custom metadata to override/add
        verbose: Print extraction details

    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract metadata automatically
        metadata = extract_all_metadata(file_path, custom_metadata)

        if verbose:
            print_metadata(metadata, f"Processing: {Path(file_path).name}")
            print(f"Use contextual embeddings: {use_contextual_embeddings}\n")

        # Add to RAG system
        rag.add_transcript(file_path, metadata=metadata, use_contextual_embeddings=use_contextual_embeddings)

        if verbose:
            print(f"✓ Successfully added to RAG\n")

        return True

    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}\n")
        return False


def add_folder_with_auto_metadata(
    folder_path: str,
    file_extension: str = ".txt",
    custom_metadata: Optional[Dict[str, str]] = None,
    collection_name: str = "transcripts",
    persist_directory: str = "./chroma_db",
    recursive: bool = False,
    verbose: bool = True,
    use_contextual_embeddings: bool = True,
) -> int:
    """
    Add all files from a folder to RAG with automatic metadata extraction.

    Args:
        folder_path: Path to the folder containing files
        file_extension: File extension to look for
        custom_metadata: Optional custom metadata to add to all files
        collection_name: ChromaDB collection name
        persist_directory: Directory to persist the vector database
        recursive: Search subdirectories recursively
        verbose: Print detailed output

    Returns:
        Number of files successfully added
    """
    # Initialize RAG system
    if verbose:
        print(f"Initializing RAG system...")
        print(f"Collection: {collection_name}")
        print(f"Persist directory: {persist_directory}\n")
        print(f"Use contextual embeddings: {use_contextual_embeddings}\n")

    rag = TranscriptRAG(collection_name=collection_name, persist_directory=persist_directory)

    # Get all files
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    # Find files
    if recursive:
        files = list(folder.rglob(f"*{file_extension}"))
    else:
        files = list(folder.glob(f"*{file_extension}"))

    if not files:
        print(f"No {file_extension} files found in {folder_path}")
        return 0

    if verbose:
        print(f"Found {len(files)} file(s) to process\n")
        print("=" * 60)

    # Process each file
    added_count = 0
    for file_path in files:
        if add_file_with_auto_metadata(
            str(file_path),
            rag,
            custom_metadata=custom_metadata,
            verbose=verbose,
            use_contextual_embeddings=use_contextual_embeddings,
        ):
            added_count += 1

    # Print summary (timing will be added by caller)
    if verbose:
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Total files found: {len(files)}")
        print(f"  Successfully added: {added_count}")
        print(f"  Failed: {len(files) - added_count}")
        print(f"  Total chunks in collection: {rag.collection.count()}")
        print("=" * 60)
        print("\nCollection Statistics:")
        rag.print_collection_stats()

    return added_count


def add_data_directory(
    data_dir: str = "./data",
    file_extension: str = ".txt",
    collection_name: str = "transcripts",
    persist_directory: str = "./chroma_db300",
    verbose: bool = True,
    use_contextual_embeddings: bool = True,
) -> int:
    """
    Add all files from the data directory and its subfolders.

    This is the simplest way to add all your transcripts - just organize
    them in folders by author and run this function.

    Directory structure expected:
        data/
        ├── author_name_1/
        │   ├── transcript1.txt
        │   └── transcript2.txt
        └── author_name_2/
            └── transcript3.txt

    Args:
        data_dir: Path to the data directory
        file_extension: File extension to look for
        collection_name: ChromaDB collection name
        persist_directory: Directory to persist the vector database
        verbose: Print detailed output
        use_contextual_embeddings: Use contextual embeddings for embedding

    Returns:
        Total number of files added
    """
    return add_folder_with_auto_metadata(
        folder_path=data_dir,
        file_extension=file_extension,
        collection_name=collection_name,
        persist_directory=persist_directory,
        recursive=True,
        verbose=verbose,
        use_contextual_embeddings=use_contextual_embeddings,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Smart RAG adder with automatic metadata extraction from file paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add entire data directory (recommended - simplest approach)
  python smart_add_to_rag.py --data-dir

  # Add a specific folder
  python smart_add_to_rag.py --folder data/andrew_huberman

  # Add folder recursively (including subfolders)
  python smart_add_to_rag.py --folder data --recursive

  # Add single file
  python smart_add_to_rag.py --file data/andrew_huberman/ketamine_depression.txt

  # Add with custom metadata
  python smart_add_to_rag.py --folder data/huberman --podcast "Huberman Lab"

Directory structure for automatic author extraction:
  data/
  ├── andrew_huberman/
  │   ├── ketamine_depression_ptsd.txt  # Author: Andrew Huberman
  │   └── sleep_optimization.txt        # Keywords: sleep, optimization
  └── lex_fridman/
      └── ai_future.txt                 # Topic: technology

Metadata is automatically extracted:
  - Author: from parent folder name (e.g., 'andrew_huberman/file.txt' -> 'Andrew Huberman')
  - Keywords: from filename (e.g., 'ketamine_depression' -> ['ketamine', 'depression'])
  - Topic: inferred from keywords (e.g., ['ketamine', 'depression'] -> 'medicine')
        """
    )

    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Add a single file with automatic metadata extraction'
    )

    parser.add_argument(
        '--folder',
        type=str,
        help='Add all files from a folder with automatic metadata extraction'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Add all files from the data directory recursively (default: ./data)'
    )

    parser.add_argument(
        '--extension', '-e',
        type=str,
        default='.txt',
        help='File extension to look for (default: .txt)'
    )

    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search subdirectories recursively'
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
        default='./chroma_db_context',
        help='Directory to persist the vector database (default: ./chroma_db_context)'
    )

    # Optional custom metadata
    parser.add_argument(
        '--podcast',
        type=str,
        help='Override podcast name in metadata'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    parser.add_argument(
        "--use-contextual-embeddings",
        action='store_true',
        help='Use contextual embeddings for embedding'
    )

    args = parser.parse_args()

    # Build custom metadata if provided
    custom_metadata = {}
    if args.podcast:
        custom_metadata['podcast'] = args.podcast

    verbose = not args.quiet

    print("use_contextual_embeddings", args.use_contextual_embeddings)

    # Start timer
    start_time = time.time()

    try:
        # Handle data directory mode (simplest)
        if args.data_dir:
            if verbose:
                print(f"Adding all files from '{args.data_dir}/' recursively...")
                print()
            add_data_directory(
                data_dir=args.data_dir,
                file_extension=args.extension,
                collection_name=args.collection,
                persist_directory=args.persist_dir,
                verbose=verbose,
                use_contextual_embeddings=args.use_contextual_embeddings,
            )

        # Handle single file
        elif args.file:
            rag = TranscriptRAG(
                collection_name=args.collection,
                persist_directory=args.persist_dir
            )
            add_file_with_auto_metadata(
                args.file,
                rag,
                custom_metadata=custom_metadata if custom_metadata else None,
                verbose=verbose,
                use_contextual_embeddings=args.use_contextual_embeddings,
            )
            if verbose:
                print("\nCollection Statistics:")
                rag.print_collection_stats()

        # Handle folder
        elif args.folder:
            add_folder_with_auto_metadata(
                folder_path=args.folder,
                file_extension=args.extension,
                custom_metadata=custom_metadata if custom_metadata else None,
                collection_name=args.collection,
                persist_directory=args.persist_dir,
                recursive=args.recursive,
                verbose=verbose,
                use_contextual_embeddings=args.use_contextual_embeddings,
            )

        else:
            parser.print_help()
            print("\nError: Please specify --file, --folder, or --data-dir")
            return 1

    except Exception as e:
        print(f"Error: {e}")
        return 1

    finally:
        # Calculate and display elapsed time
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60

        if verbose:
            print("\n" + "=" * 60)
            print("⏱️  Execution Time:")
            if hours > 0:
                print(f"  Total time: {hours}h {minutes}m {seconds:.2f}s")
            elif minutes > 0:
                print(f"  Total time: {minutes}m {seconds:.2f}s")
            else:
                print(f"  Total time: {seconds:.2f}s")
            print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
