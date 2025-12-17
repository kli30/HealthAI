import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Tuple

class TranscriptRAG:
    """RAG system for querying transcript data using semantic search."""

    def __init__(self, collection_name: str = "transcripts", persist_directory: str = "./chroma_db"):
        """Initialize the RAG system with ChromaDB.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
        """
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Use sentence transformers for embeddings (all-MiniLM-L6-v2 is fast and effective)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: The text to chunk
            chunk_size: Approximate number of words per chunk
            overlap: Number of words to overlap between chunks

        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:  # Don't add empty chunks
                chunks.append(chunk)

        return chunks

    def create_contextual_chunk(self, chunk: str, metadata: dict) -> str:
        """Create a contextual version of a chunk by prepending metadata.

        This helps the embedding model better understand the context of each chunk,
        leading to more accurate semantic search.

        Args:
            chunk: The original text chunk
            metadata: Metadata dictionary containing author, topic, keywords, etc.

        Returns:
            Chunk with contextual header prepended
        """
        context_parts = []

        # Add author context
        if 'author' in metadata:
            context_parts.append(f"Author: {metadata['author']}")

        # Add topic context
        if 'topic' in metadata:
            context_parts.append(f"Topic: {metadata['topic']}")

        # Add keywords context
        if 'keywords' in metadata:
            keywords = metadata['keywords']
            if isinstance(keywords, list):
                keywords = ', '.join(keywords)
            context_parts.append(f"Keywords: {keywords}")

        # Add podcast/source context
        if 'podcast' in metadata:
            context_parts.append(f"Podcast: {metadata['podcast']}")

        # Build the contextual header
        if context_parts:
            header = ' | '.join(context_parts)
            return f"[{header}]\n\n{chunk}"
        else:
            return chunk

    def add_transcript(self, transcript_path: str, metadata: dict = None, use_contextual_embeddings: bool = True):
        """Add a transcript to the vector database.

        Args:
            transcript_path: Path to the transcript file
            metadata: Optional metadata to associate with the transcript
            use_contextual_embeddings: If True, prepend contextual information to chunks before embedding
        """
        with open(transcript_path, 'r') as f:
            text = f.read()

        chunks = self.chunk_text(text)

        # Create IDs and metadata for each chunk
        base_name = os.path.basename(transcript_path)
        ids = [f"{base_name}_chunk_{i}" for i in range(len(chunks))]

        # Add source file to metadata for each chunk
        # Convert list values to strings for ChromaDB compatibility
        metadatas = []
        for i in range(len(chunks)):
            chunk_metadata = {"source": base_name, "chunk_index": i}
            if metadata:
                # Deep copy metadata and convert lists to comma-separated strings
                for key, value in metadata.items():
                    if isinstance(value, list):
                        chunk_metadata[key] = ', '.join(str(v) for v in value)
                    else:
                        chunk_metadata[key] = value
            metadatas.append(chunk_metadata)

        # Create contextual versions of chunks if enabled
        if use_contextual_embeddings and metadata:
            contextual_chunks = [
                self.create_contextual_chunk(chunk, metadata)
                for chunk in chunks
            ]
        else:
            contextual_chunks = chunks

        # Add to collection (contextual chunks are embedded, original chunks stored in metadata)
        # Store original chunks in metadata for clean retrieval
        for i, chunk_metadata in enumerate(metadatas):
            chunk_metadata['original_chunk'] = chunks[i]

        self.collection.add(
            documents=contextual_chunks,
            ids=ids,
            metadatas=metadatas
        )

        context_note = " with contextual embeddings" if use_contextual_embeddings else ""
        print(f"Added {len(chunks)} chunks from {base_name}{context_note}")

    def query(self, query_text: str, n_results: int = 3, author_filter=None, return_original: bool = True) -> List[Tuple[str, dict]]:
        """Query the vector database for relevant chunks.

        Args:
            query_text: The query string
            n_results: Number of results to return
            author_filter: Optional author name(s) to filter results. Can be a string or list of strings.
            return_original: If True, return original chunks without context headers (cleaner for display)

        Returns:
            List of tuples (chunk_text, metadata)
        """
        # Build where clause for filtering by author
        where_clause = None
        if author_filter:
            if isinstance(author_filter, list) and len(author_filter) > 0:
                # Use $in operator for multiple authors
                where_clause = {"author": {"$in": author_filter}}
            elif isinstance(author_filter, str):
                # Single author filter
                where_clause = {"author": author_filter}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_clause
        )

        # Format results as list of (text, metadata) tuples
        chunks = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                # Return original chunk if available and requested (for cleaner display)
                if return_original and 'original_chunk' in metadata:
                    chunk_text = metadata['original_chunk']
                else:
                    chunk_text = doc

                chunks.append((chunk_text, metadata))

        return chunks

    def get_context_for_query(self, query_text: str, n_results: int = 3, author_filter=None) -> str:
        """Get formatted context string for a query.

        Args:
            query_text: The query string
            n_results: Number of chunks to retrieve
            author_filter: Optional author name(s) to filter results. Can be a string or list of strings.

        Returns:
            Formatted context string to include in a prompt
        """
        # Query with author filter (handles both single author and list of authors)
        chunks = self.query(query_text, n_results, author_filter=author_filter)

        if not chunks:
            return ""

        context_parts = ["Here is relevant context from the transcripts:\n"]

        for i, (text, metadata) in enumerate(chunks, 1):
            source = metadata.get('source', 'unknown')
            context_parts.append(f"\n[Source: {source}]")
            context_parts.append(text)
            context_parts.append("")

        return '\n'.join(context_parts)

    def clear_collection(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_function
        )
        print("Collection cleared")

    def get_authors(self) -> List[str]:
        """Get a list of all unique authors in the collection.

        Returns:
            List of unique author names
        """
        all_items = self.collection.get()
        authors = set()

        if all_items['metadatas']:
            for metadata in all_items['metadatas']:
                if 'author' in metadata:
                    authors.add(metadata['author'])

        return sorted(list(authors))

    def get_collection_stats(self) -> dict:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        all_items = self.collection.get()
        total_chunks = len(all_items['ids']) if all_items['ids'] else 0

        # Count by author
        author_counts = {}
        source_counts = {}

        if all_items['metadatas']:
            for metadata in all_items['metadatas']:
                author = metadata.get('author', 'Unknown')
                source = metadata.get('source', 'Unknown')

                author_counts[author] = author_counts.get(author, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1

        return {
            'total_chunks': total_chunks,
            'total_authors': len(author_counts),
            'total_sources': len(source_counts),
            'chunks_by_author': author_counts,
            'chunks_by_source': source_counts
        }

    def print_collection_stats(self):
        """Print formatted collection statistics."""
        stats = self.get_collection_stats()

        print("\n" + "=" * 60)
        print("RAG Collection Statistics")
        print("=" * 60)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Unique authors: {stats['total_authors']}")
        print(f"Unique sources: {stats['total_sources']}")

        if stats['chunks_by_author']:
            print("\nChunks by Author:")
            for author, count in sorted(stats['chunks_by_author'].items()):
                print(f"  {author}: {count} chunks")

        print("=" * 60 + "\n")


def initialize_rag_with_transcripts():
    """Initialize RAG system and load all transcripts."""
    rag = TranscriptRAG()

    # Check if collection is already populated
    count = rag.collection.count()
    if count > 0:
        print(f"Collection already contains {count} chunks")
        return rag

    # Add transcripts
    transcripts = [
        ("transcript_ketamine.txt", {"topic": "ketamine", "podcast": "Huberman Lab"}),
        ("transcript_depression.txt", {"topic": "depression", "podcast": "Huberman Lab"})
    ]

    for transcript_file, metadata in transcripts:
        if os.path.exists(transcript_file):
            rag.add_transcript(transcript_file, metadata)
        else:
            print(f"Warning: {transcript_file} not found")

    return rag


if __name__ == "__main__":
    # Example usage
    rag = initialize_rag_with_transcripts()

    # Test query
    query = "What are the benefits of ketamine for depression?"
    print(f"\nQuery: {query}")
    print("\n" + "="*80)
    context = rag.get_context_for_query(query, n_results=3)
    print(context)
