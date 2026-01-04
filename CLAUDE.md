# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Structure

```
healthAI/
├── src/                          # Source code
│   ├── rag_system.py            # Core RAG system with ChromaDB
│   ├── llm_client.py            # Unified LLM client (OpenAI/Anthropic)
│   ├── metadata_extractor.py   # Automatic metadata extraction
│   ├── chat_rag.py              # RAG-enhanced chat interface
│   ├── web_chat.py              # Flask web interface
│   ├── smart_add_to_rag.py     # Smart RAG adder with auto metadata
│   ├── add_folder_to_rag.py    # Manual folder-based RAG adder
│   └── templates/               # HTML templates for web UI
├── testing/                      # Demos and examples
│   ├── example_folder_rag.py   # RAG collection statistics
│   ├── demo_auto_metadata.py   # Metadata extraction demo
│   └── demo_reranking.py       # Cross-encoder reranking demo
├── data/                         # Transcript data (organized by author)
└── chroma_db/                    # Vector database storage
```

## Overview

A RAG-powered chatbot system for querying health and science transcripts from domain experts.

**Key Components:**
- **RAG System**: ChromaDB vector database with contextual embeddings and cross-encoder reranking
- **Chat Interface**: Terminal-based (`chat_rag.py`) and web-based (`web_chat.py`) interfaces
- **Smart Metadata**: Automatic extraction from folder/file names (author, keywords, topics)

## Quick Start

### RAG-Enhanced Chat
```bash
# Using OpenAI with default database (./chroma_db_context) - reranking enabled by default
uv run python src/chat_rag.py

# Using custom database
uv run python src/chat_rag.py --db ./chroma_db_context

# Disable reranking for faster queries (slightly lower accuracy)
uv run python src/chat_rag.py --no-reranking

# Using Anthropic Claude with custom database
export LLM_PROVIDER=anthropic
uv run python src/chat_rag.py --db ./my_custom_db

# View all options
uv run python src/chat_rag.py --help
```
- Type messages to chat with AI (OpenAI gpt-5-mini by default, or Claude Sonnet 4.5)
- Type 'quit' to exit
- In RAG mode, relevant transcript chunks are automatically retrieved for each query
- Use `--db` to specify which ChromaDB database to use (default: `./chroma_db_context`)

### Run Web Chatbox
Web-based chat interface (with RAG):
```bash
# Using default database on default port (5000) - reranking enabled by default
uv run python src/web_chat.py

# Using custom database
uv run python src/web_chat.py --db ./chroma_db_context

# Disable reranking for faster queries
uv run python src/web_chat.py --no-reranking

# Using custom port
uv run python src/web_chat.py --port 8080

# Using custom database and port
uv run python src/web_chat.py --db ./my_custom_db --port 8080

# View all options
uv run python src/web_chat.py --help
```
Then open your browser to: http://localhost:5000 (or your custom port)
- Modern, responsive chat interface
- Real-time streaming responses
- Automatic context retrieval from transcripts
- Session-based conversation history
- Clear conversation button
- Use `--db` to specify which ChromaDB database to use
- Use `--port` to change the web server port

### Add Folder of Transcripts to RAG

**Option 1: Automatic metadata extraction (recommended)**
```bash
# Add entire data directory - metadata extracted automatically!
uv run python src/smart_add_to_rag.py --data-dir

# Add specific folder
uv run python src/smart_add_to_rag.py --folder data/andrew_huberman

# Add single file
uv run python src/smart_add_to_rag.py --file data/huberman/ketamine_depression.txt
```
Metadata is automatically extracted from:
- **Author**: parent folder name (e.g., `andrew_huberman/file.txt` → "Andrew Huberman")
- **Keywords**: filename (e.g., `ketamine_depression.txt` → ["ketamine", "depression"])
- **Topic**: inferred from keywords (e.g., ["ketamine", "depression"] → "medicine")

**Option 2: Manual metadata entry**
```bash
# Interactive mode (recommended for first-time users)
uv run python src/add_folder_to_rag.py --interactive

# Command-line mode
uv run python src/add_folder_to_rag.py --folder data/huberman_transcripts --author "Andrew Huberman" --podcast "Huberman Lab"

# With topic metadata
uv run python src/add_folder_to_rag.py --folder data/transcripts --author "Author Name" --topic "neuroscience" --podcast "Podcast Name"
```

### View RAG Collection Statistics
```bash
uv run python testing/example_folder_rag.py
```
This displays statistics about your RAG collection, including authors and chunk counts.

### Test Metadata Extraction
```bash
uv run python testing/demo_auto_metadata.py
```
This demonstrates how automatic metadata extraction works with various filename patterns.

### Test Cross-Encoder Reranking
```bash
uv run python testing/demo_reranking.py
```
This demonstrates the reranking system with comparisons, relevance scores, and performance impact.

## Architecture

### LLM Client (`src/llm_client.py`)
- **Unified Interface**: Single interface for both OpenAI and Anthropic APIs
- **Default Provider**: OpenAI gpt-5-mini (can be changed via `LLM_PROVIDER` env var)
- **Streaming Support**: Consistent streaming interface across providers
- **Model Configuration**: Default models (gpt-5-mini for OpenAI, Claude Sonnet 4.5 for Anthropic)
- **Easy Switching**: Change providers via environment variable without code changes

### RAG System (`src/rag_system.py`)
- **Vector Database**: ChromaDB with persistent storage in `./chroma_db/`
- **Embeddings**: Uses `sentence-transformers` with the `all-MiniLM-L6-v2` model (local, no API required)
- **Cross-Encoder Reranking**: Two-stage retrieval using `cross-encoder/ms-marco-MiniLM-L6-v2` for improved relevance (enabled by default)
- **Contextual Embeddings**: Prepends metadata (author, topic, keywords) to chunks before embedding for better retrieval
- **Chunking Strategy**: Splits transcripts into ~300 word chunks with 30 word overlap
- **Retrieval**: Semantic search returns top-k most relevant chunks with source metadata
- **Key Class**: `TranscriptRAG` handles indexing, querying, and context formatting
- **Author Filtering**: Query results can be filtered by author metadata
- **Statistics**: Built-in methods to view collection statistics and available authors

### Folder-based RAG (`src/add_folder_to_rag.py`)
- **Batch Processing**: Add entire folders of transcripts at once
- **Author Metadata**: Each transcript is tagged with author information
- **Additional Metadata**: Support for topic, podcast name, and custom metadata fields
- **Interactive Mode**: User-friendly prompts for configuration
- **Command-line Mode**: Script-friendly for automation
- **Flexible File Types**: Support for .txt, .md, and other text formats
- **Progress Tracking**: Shows processing status for each file

### Automatic Metadata Extraction (`src/metadata_extractor.py` & `src/smart_add_to_rag.py`)
- **Smart Author Extraction**: Infers author from parent folder name
  - `andrew_huberman/file.txt` → "Andrew Huberman"
  - `any_folder/author_name/file.txt` → "Author Name"
  - Handles snake_case, kebab-case, and spaces
- **Keyword Parsing**: Extracts meaningful keywords from filenames
  - `ketamine_benefits_depression.txt` → ["ketamine", "benefits", "depression"]
  - Filters stop words and short words
  - Handles various separators (_, -, spaces)
- **Topic Inference**: AI-powered topic classification from keywords
  - Uses keyword matching against topic dictionaries
  - Categories: neuroscience, psychology, medicine, sleep, performance, technology, health, science
  - Scoring system for best match
- **Zero Configuration**: Just organize files and run - metadata is automatic
- **Customizable**: Can override or add custom metadata fields

### RAG-Enhanced Chat (`src/chat_rag.py`)
- Automatically retrieves top 3 relevant chunks from transcripts for each user query
- Uses unified LLM client (default: OpenAI gpt-5-mini, or Claude Sonnet 4.5) with 2000 max tokens
- Injects retrieved context into messages before sending to the LLM
- Visual indicator when context is retrieved (yellow "[Retrieved context from transcripts]")
- Maintains full conversation history for multi-turn dialogue

### Web Chatbox (`src/web_chat.py`)
- **Framework**: Flask web server with RESTful API endpoints
- **Frontend**: Responsive HTML/CSS/JavaScript interface with modern design (in `src/templates/`)
- **Streaming**: Server-Sent Events (SSE) for real-time response streaming
- **RAG Integration**: Automatically retrieves relevant context for each query
- **Session Management**: Maintains separate conversation history per user session
- **Features**:
  - Real-time typing indicators
  - Visual context indicators when transcripts are retrieved
  - Clear conversation functionality
  - Mobile-responsive design with gradient UI
  - Smooth animations and modern UX

### Data Flow
 

**RAG workflow:**
1. `src/rag_system.py` chunks transcript text into overlapping segments
2. Each chunk is embedded using sentence-transformers
3. Embeddings and chunks are stored in ChromaDB (persists to disk)
4. When user asks a question in `src/chat_rag.py`:
   - Query is embedded using the same model
   - Top-20+ semantically similar candidates are retrieved via vector similarity search (when reranking is enabled)
   - Cross-encoder reranks candidates by relevance scores (when reranking is enabled)
   - Top-k highest-scoring chunks are selected and prepended to the user's message as context
   - LLM (OpenAI or Anthropic) generates a response informed by relevant transcript excerpts

## Dependencies

This project uses `uv` for fast, reliable Python package management.

Install all dependencies:
```bash
uv sync
```

Dependencies include:
- `openai` - OpenAI API client (default LLM provider)
- `anthropic` - Anthropic Claude API client (alternative LLM provider)
- `chromadb` - Vector database for RAG
- `sentence-transformers` - Local embeddings model (includes both bi-encoders and cross-encoders)
- `flask` - Web framework for the chatbox interface

**API Key Configuration:**
- For OpenAI (default): Set `OPENAI_API_KEY` environment variable
- For Anthropic: Set `ANTHROPIC_API_KEY` and `LLM_PROVIDER=anthropic` environment variables

## RAG System Notes

- **First Run**: On first execution, `src/chat_rag.py` will download the sentence-transformers models (~100MB for bi-encoder, ~80MB for cross-encoder) and index transcripts.
- **Persistence**: Vector databases are persisted to disk (default varies by tool), so subsequent runs are instant.
- **Database Defaults**: Different tools use different default databases:
  - `chat_rag.py` and `web_chat.py`: `./chroma_db_context`
  - `smart_add_to_rag.py`: `./chroma_db300`
  - `add_folder_to_rag.py`: `./chroma_db`
  - **Important**: Ensure you're adding data to and querying from the same database using the `--db` flag.
- **Reindexing**: To reindex transcripts, delete the `./chroma_db/` directory or call `rag.clear_collection()`.
- **Chunk Metadata**: Each chunk includes `source` (filename), `chunk_index` (position), and custom metadata like `keywords` and `author`.
- **Customization**: Adjust chunk size, overlap, or number of retrieved results in `src/rag_system.py`.
- **Reranking**: Enabled by default. Disable with `--no-reranking` flag for faster queries (with slightly lower accuracy).

## Contextual Embeddings

### What are Contextual Embeddings?

Contextual embeddings enhance retrieval accuracy by prepending metadata to each chunk before embedding. Instead of embedding just the raw text, we embed:

```
[Author: Andrew Huberman | Keywords: circadian, rhythm | Source: sleep_huberman.txt]

Sleep is fundamental to cognitive function and overall health...
```

This helps the embedding model better understand:
- **Who** is speaking (author)
- **What** keywords are relevant (keywords)
- **Where** it comes from (source filename)

### Benefits

1. **Better Author Disambiguation**: Distinguishes between different experts discussing similar topics
2. **Keyword Enhancement**: Keywords boost relevance for specific search terms
3. **Source Context**: Filename/source information helps identify the origin of content
4. **Cleaner Results**: Original chunks are stored separately, so search results remain clean

### How It Works

**When adding transcripts:**
```python
from rag_system import TranscriptRAG

rag = TranscriptRAG()

# With contextual embeddings (default, recommended)
rag.add_transcript(
    "data/huberman/sleep.txt",
    metadata={
        "author": "Andrew Huberman",
        "keywords": ["circadian", "rhythm", "melatonin"]
    },
    use_contextual_embeddings=True  # Default
)
# Note: 'source' (filename) is automatically added by the system

# Without contextual embeddings (legacy mode)
rag.add_transcript(
    "data/huberman/sleep.txt",
    metadata={"author": "Andrew Huberman"},
    use_contextual_embeddings=False
)
```

**When querying:**
```python
# Returns clean chunks (original text) by default
results = rag.query("What does Huberman say about sleep?", return_original=True)

# Or return the full contextual chunks
results = rag.query("What does Huberman say about sleep?", return_original=False)
```

### Testing Contextual Embeddings

Run the demo to see contextual embeddings in action:
```bash
uv run python testing/demo_contextual_embeddings.py
```

This compares retrieval accuracy with and without contextual embeddings.

## Cross-Encoder Reranking

### What is Cross-Encoder Reranking?

Cross-encoder reranking is a **two-stage retrieval approach** that significantly improves result quality:

1. **Stage 1 - Fast Retrieval (Bi-Encoder)**: Retrieve 20+ candidate chunks using fast bi-encoder embeddings
2. **Stage 2 - Precise Ranking (Cross-Encoder)**: Re-score all candidates using a more accurate cross-encoder model
3. **Final Selection**: Return top N chunks sorted by cross-encoder scores

### Why Reranking?

**Bi-Encoders** (used in Stage 1):
- Encode query and documents independently
- Very fast for initial retrieval
- Good recall, but moderate precision

**Cross-Encoders** (used in Stage 2):
- Jointly encode query and document together
- Much more accurate at measuring relevance
- Captures subtle semantic relationships
- Slower, so only used on a small candidate set

By combining both, we get the speed of bi-encoders with the accuracy of cross-encoders.

### Benefits

1. **Higher Accuracy**: Cross-encoders are ~10-20% more accurate than bi-encoders for ranking
2. **Better Context Quality**: More relevant chunks lead to better LLM responses
3. **Nuanced Understanding**: Captures query-document interactions that bi-encoders miss
4. **Minimal Speed Impact**: Only reranks a small candidate set (~20 chunks)

### How It Works

**Initialization:**
```python
from rag_system import TranscriptRAG

# With reranking (default, recommended)
rag = TranscriptRAG(use_reranking=True)

# Without reranking (faster, slightly lower accuracy)
rag = TranscriptRAG(use_reranking=False)
```

**Querying with scores:**
```python
# Basic query (reranking happens automatically if enabled)
results = rag.query("What are the benefits of sleep?", n_results=3)

# Query with relevance scores
results = rag.query("What are the benefits of sleep?", n_results=3, return_scores=True)
for text, metadata, score in results:
    print(f"Score: {score:.4f} | {text[:100]}...")

# Get formatted context with scores
context = rag.get_context_for_query("What are the benefits of sleep?", show_scores=True)
```

**Command-line usage:**
```bash
# With reranking (default)
uv run python src/chat_rag.py

# Without reranking (faster)
uv run python src/chat_rag.py --no-reranking

# Web interface with reranking
uv run python src/web_chat.py

# Web interface without reranking
uv run python src/web_chat.py --no-reranking
```

### Model Details

- **Bi-Encoder**: `all-MiniLM-L6-v2` (384 dimensions, ~80MB)
  - Fast encoding (~1000 docs/sec on CPU)
  - Used for initial candidate retrieval

- **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L6-v2` (~80MB)
  - Trained on MS MARCO passage ranking dataset
  - Optimized for question-answering and passage retrieval
  - Scores range from ~-10 to +10 (higher = more relevant)

### Testing Reranking

Run the demo to see reranking in action:
```bash
uv run python testing/demo_reranking.py
```

This demonstrates:
1. **Basic reranking**: Query with relevance scores
2. **Comparison**: Results with vs without reranking
3. **Context with scores**: Formatted context showing relevance
4. **Author filtering**: Reranking combined with author filters

### Performance Impact

**With Reranking:**
- Initial retrieval: ~20-50ms (retrieve 20 candidates)
- Reranking: ~10-30ms (score 20 candidates with cross-encoder)
- Total: ~30-80ms per query

**Without Reranking:**
- Initial retrieval: ~20-50ms (retrieve 3 results directly)
- Total: ~20-50ms per query

The small speed decrease (~30ms) is usually worth it for significantly better accuracy.

## Automatic Metadata Extraction

### Overview

The system can automatically extract metadata from your file organization without manual entry. Simply organize your files logically, and the system handles the rest!

### How It Works

**Step 1: Organize your files by author**
```
data/
├── andrew_huberman/
│   ├── ketamine_benefits_depression_ptsd.txt
│   ├── sleep_optimization_circadian_rhythm.txt
│   └── dopamine_motivation_focus.txt
├── lex_fridman/
│   ├── artificial_intelligence_future.txt
│   └── machine_learning_neural_networks.txt
└── tim_ferriss/
    └── productivity_performance_optimization.txt
```

**Step 2: Run the smart add command**
```bash
python src/smart_add_to_rag.py --data-dir
```

**Step 3: Done!** Metadata is automatically extracted:

| File | Author | Keywords | Topic |
|------|--------|----------|-------|
| `data/andrew_huberman/ketamine_benefits_depression_ptsd.txt` | Andrew Huberman | ketamine, benefits, depression, ptsd | medicine |
| `data/lex_fridman/artificial_intelligence_future.txt` | Lex Fridman | artificial, intelligence, future | technology |
| `data/andrew_huberman/sleep_optimization_circadian_rhythm.txt` | Andrew Huberman | sleep, optimization, circadian, rhythm | sleep |

### Metadata Extraction Rules

1. **Author Extraction**
   - Extracted from parent folder name of the transcript file
   - Formatting: `snake_case` → `Title Case`
   - Examples:
     - `andrew_huberman/file.txt` → "Andrew Huberman"
     - `lex_fridman/file.txt` → "Lex Fridman"
     - `tim-ferriss/file.txt` → "Tim Ferriss"
     - `data/author_name/file.txt` → "Author Name"

2. **Keyword Extraction**
   - Parsed from filename (before `.txt` extension)
   - Separators: underscores, hyphens, spaces
   - Filters: stop words removed, min length 3 chars
   - Examples:
     - `ketamine_benefits_depression.txt` → ["ketamine", "benefits", "depression"]
     - `The-Art-of-Learning.txt` → ["art", "learning"]
     - `AI and Machine Learning.txt` → ["machine", "learning"]

3. **Topic Inference**
   - Inferred from keywords using topic dictionaries
   - 8 categories: neuroscience, psychology, medicine, sleep, performance, technology, health, science
   - Scoring system finds best match
   - Examples:
     - Keywords ["ketamine", "depression"] → Topic: "medicine"
     - Keywords ["sleep", "circadian"] → Topic: "sleep"
     - Keywords ["ai", "learning"] → Topic: "technology"

### Demonstration

Run the demo to see extraction in action:
```bash
python testing/demo_auto_metadata.py
```

This shows:
- Individual extraction functions
- Various filename patterns
- Topic inference logic
- Code usage examples

### Advanced Usage

**Add with custom override:**
```bash
# Auto-extract metadata but add podcast field
uv run python src/smart_add_to_rag.py --data-dir --podcast "Huberman Lab"
```

**Programmatic usage:**
```python
import sys
sys.path.insert(0, 'src')

from metadata_extractor import extract_all_metadata
from rag_system import TranscriptRAG

# Extract metadata for a file
metadata = extract_all_metadata("data/andrew_huberman/sleep.txt")
print(metadata)
# {
#   'author': 'Andrew Huberman',
#   'keywords': 'sleep',
#   'topic': 'sleep',
#   'filename': 'sleep.txt',
#   'source_path': 'data/andrew_huberman'
# }

# Add to RAG with auto metadata
rag = TranscriptRAG()
rag.add_transcript("data/andrew_huberman/sleep.txt", metadata=metadata)
```

## Working with Author Metadata

### Adding Transcripts with Author Information

The folder-based approach allows you to organize transcripts by author:

```python
import sys
sys.path.insert(0, 'src')

from add_folder_to_rag import add_folder_to_rag

# Add Huberman Lab transcripts
add_folder_to_rag(
    folder_path="data/huberman_lab",
    author="Andrew Huberman",
    podcast="Huberman Lab",
    topic="neuroscience"
)

# Add Lex Fridman podcast transcripts
add_folder_to_rag(
    folder_path="data/lex_fridman",
    author="Lex Fridman",
    podcast="Lex Fridman Podcast",
    topic="technology"
)
```

### Querying by Author

Filter search results to specific authors:

```python
import sys
sys.path.insert(0, 'src')

from rag_system import TranscriptRAG

rag = TranscriptRAG()

# Query only Huberman's transcripts
results = rag.query(
    "What are the benefits of cold exposure?",
    n_results=3,
    author_filter="Andrew Huberman"
)

# Query all transcripts (no filter)
results = rag.query("What is AI?", n_results=5)
```

### Viewing Collection Statistics

```python
import sys
sys.path.insert(0, 'src')

from rag_system import TranscriptRAG

rag = TranscriptRAG()

# Print formatted statistics
rag.print_collection_stats()

# Get list of authors
authors = rag.get_authors()
print(f"Available authors: {authors}")

# Get detailed statistics
stats = rag.get_collection_stats()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Chunks by author: {stats['chunks_by_author']}")
```

### Organizing Your Data

Recommended folder structure:
```
data/
├── huberman_lab/
│   ├── transcript1.txt
│   ├── transcript2.txt
│   └── ...
├── lex_fridman/
│   ├── transcript1.txt
│   └── ...
└── other_author/
    └── ...
```

Then add each folder with its respective author metadata:
```bash
uv run python src/add_folder_to_rag.py --folder data/huberman_lab --author "Andrew Huberman"
uv run python src/add_folder_to_rag.py --folder data/lex_fridman --author "Lex Fridman"
```
