# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

```
openai/
├── src/                          # Essential source code
│   ├── rag_system.py            # Core RAG system
│   ├── llm_client.py            # Unified LLM client (OpenAI/Anthropic)
│   ├── metadata_extractor.py   # Automatic metadata extraction
│   ├── chat.py                  # Basic chat interface
│   ├── chat_rag.py              # RAG-enhanced chat
│   ├── web_chat.py              # Web interface
│   ├── smart_add_to_rag.py     # Smart RAG file/folder adder
│   ├── add_folder_to_rag.py    # Manual folder-based RAG adder
│   └── templates/               # HTML templates for web interface
├── testing/                      # Testing and example code
│   ├── example_folder_rag.py   # Usage examples
│   └── demo_auto_metadata.py   # Metadata extraction demo
├── data/                         # Transcript data (organized by author)
├── chroma_db/                    # Vector database storage
└── [documentation files]         # README.md, CLAUDE.md, etc.
```

## Overview

This is a Python project for interacting with transcripts from multiple domain experts. It consists of three main components:

1. **RAG System** (`src/smart_add_to_rag.py`): Add transcripts to the RAG system with automatic metadata extraction
2. **RAG-Enhanced Chatbot** (`src/chat_rag.py`): Chatbot with automatic context retrieval from transcripts
3. **Web Chatbox** (`src/web_chat.py`): Browser-based chatbox interface with RAG support

## Running the Code

RAG-enhanced chat (with transcript context):
```bash
# Using OpenAI (default)
uv run python src/chat_rag.py

# Using Anthropic Claude
export LLM_PROVIDER=anthropic
uv run python src/chat_rag.py
```
- Type messages to chat with AI (OpenAI GPT-4o by default, or Claude)
- Type 'quit' to exit
- In RAG mode, relevant transcript chunks are automatically retrieved for each query

### Run Web Chatbox
Web-based chat interface (with RAG):
```bash
uv run python src/web_chat.py
```
Then open your browser to: http://localhost:5000
- Modern, responsive chat interface
- Real-time streaming responses
- Automatic context retrieval from transcripts
- Session-based conversation history
- Clear conversation button

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
- **Author**: folder name (e.g., `data/andrew_huberman/` → "Andrew Huberman")
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

## Architecture

### LLM Client (`src/llm_client.py`)
- **Unified Interface**: Single interface for both OpenAI and Anthropic APIs
- **Default Provider**: OpenAI GPT-4o (can be changed via `LLM_PROVIDER` env var)
- **Streaming Support**: Consistent streaming interface across providers
- **Model Configuration**: Default models (GPT-4o for OpenAI, Claude Sonnet 4.5 for Anthropic)
- **Easy Switching**: Change providers via environment variable without code changes

### Chat Interface (`src/chat.py`)
- Uses the unified LLM client to interface with OpenAI or Anthropic
- Maintains conversation history in memory for context across messages
- Streams responses for real-time output
- Default: OpenAI GPT-4o with 1000 max tokens
- Uses ANSI color codes for terminal formatting (blue for user, green for AI)

### RAG System (`src/rag_system.py`)
- **Vector Database**: ChromaDB with persistent storage in `./chroma_db/`
- **Embeddings**: Uses `sentence-transformers` with the `all-MiniLM-L6-v2` model (local, no API required)
- **Contextual Embeddings**: Prepends metadata (author, topic, keywords) to chunks before embedding for better retrieval
- **Chunking Strategy**: Splits transcripts into ~500 word chunks with 50 word overlap
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
- **Smart Author Extraction**: Parses folder structure to extract author names
  - `data/andrew_huberman/` → "Andrew Huberman"
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
- Uses unified LLM client (default: OpenAI GPT-4o, or Claude Sonnet 4.5) with 2000 max tokens
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
   - Top-k semantically similar chunks are retrieved via vector similarity search
   - Retrieved chunks are prepended to the user's message as context
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
- `sentence-transformers` - Local embeddings model
- `flask` - Web framework for the chatbox interface

**API Key Configuration:**
- For OpenAI (default): Set `OPENAI_API_KEY` environment variable
- For Anthropic: Set `ANTHROPIC_API_KEY` and `LLM_PROVIDER=anthropic` environment variables

## RAG System Notes

- **First Run**: On first execution, `src/chat_rag.py` will download the sentence-transformers model (~100MB) and index both transcripts. This may take 30-60 seconds.
- **Persistence**: The vector database is persisted to `./chroma_db/`, so subsequent runs are instant.
- **Reindexing**: To reindex transcripts, delete the `./chroma_db/` directory or call `rag.clear_collection()`.
- **Chunk Metadata**: Each chunk includes `source` (filename), `chunk_index` (position), and custom metadata like `topic`, `podcast`, and `author`.
- **Customization**: Adjust chunk size, overlap, or number of retrieved results in `src/rag_system.py`.

## Contextual Embeddings

### What are Contextual Embeddings?

Contextual embeddings enhance retrieval accuracy by prepending metadata to each chunk before embedding. Instead of embedding just the raw text, we embed:

```
[Author: Andrew Huberman | Topic: sleep | Keywords: circadian, rhythm | Podcast: Huberman Lab]

Sleep is fundamental to cognitive function and overall health...
```

This helps the embedding model better understand:
- **Who** is speaking (author)
- **What** the topic is about (topic/keywords)
- **Where** it comes from (podcast/source)

### Benefits

1. **Better Author Disambiguation**: Distinguishes between different experts discussing the same topic
2. **Improved Topic Matching**: Helps find content about specific subjects more accurately
3. **Keyword Enhancement**: Keywords boost relevance for specific search terms
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
        "topic": "sleep",
        "keywords": ["circadian", "rhythm", "melatonin"],
        "podcast": "Huberman Lab"
    },
    use_contextual_embeddings=True  # Default
)

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
   - Extracted from subfolder name under `data/`
   - Formatting: `snake_case` → `Title Case`
   - Examples:
     - `data/andrew_huberman/` → "Andrew Huberman"
     - `data/lex_fridman/` → "Lex Fridman"
     - `data/tim-ferriss/` → "Tim Ferriss"

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
