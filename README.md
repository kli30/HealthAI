# Transcript RAG System

A Retrieval-Augmented Generation (RAG) system for querying podcast transcripts using Claude with automatic metadata extraction.

## Demo

![Web Interface Demo](pics/demo.png)

## Project Structure

```
openai/
├── src/                          # Essential source code
│   ├── rag_system.py            # Core RAG system
│   ├── metadata_extractor.py   # Automatic metadata extraction
│   ├── chat.py                  # Basic chat interface
│   ├── chat_rag.py              # RAG-enhanced chat
│   ├── web_chat.py              # Web interface
│   ├── smart_add_to_rag.py     # Smart RAG adder (auto metadata)
│   ├── add_folder_to_rag.py    # Manual RAG adder
│   └── templates/               # HTML templates for web interface
├── testing/                      # Testing and example code
│   ├── example_folder_rag.py   # Usage examples
│   └── demo_auto_metadata.py   # Metadata extraction demo
├── data/                         # Transcript data (organize by author)
├── chroma_db/                    # Vector database storage
└── requirements.txt             # Python dependencies
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your API key:**
   ```bash
   # For OpenAI (default)
   export OPENAI_API_KEY='your-openai-key-here'

   # Or for Anthropic Claude
   export ANTHROPIC_API_KEY='your-anthropic-key-here'
   export LLM_PROVIDER='anthropic'
   ```

3. **Add transcripts to RAG system:**
   ```bash
   # Add entire data directory with automatic metadata extraction
   python src/smart_add_to_rag.py --data-dir
   ```

4. **Run the RAG-enhanced chatbot:**
   ```bash
   python src/chat_rag.py
   ```

5. **Or run the web interface:**
   ```bash
   python src/web_chat.py
   ```
   Then open http://localhost:5000 in your browser.

## Features

- **Multiple LLM Providers**: Supports both OpenAI (default) and Anthropic Claude
- **Semantic Search**: Automatically finds relevant transcript excerpts for your questions
- **Automatic Metadata Extraction**: Extracts author, keywords, and topics from file organization
- **Local Embeddings**: Uses sentence-transformers (no external API needed for embeddings)
- **Persistent Storage**: Vector database saves to disk for fast subsequent runs
- **Multiple Authors**: Support for transcripts from multiple podcast hosts
- **Interactive Chat**: Natural conversation with AI, enhanced by transcript context
- **Web Interface**: Modern, responsive web UI with real-time streaming

## Core Files

### Source Code (`src/`)
- `chat_rag.py` - RAG-enhanced terminal chatbot (recommended)
- `web_chat.py` - Web-based chat interface
- `rag_system.py` - Core RAG functionality
- `llm_client.py` - Unified LLM client supporting OpenAI and Anthropic
- `smart_add_to_rag.py` - Add transcripts with automatic metadata extraction
- `add_folder_to_rag.py` - Add transcripts with manual metadata entry
- `metadata_extractor.py` - Automatic metadata extraction logic
- `chat.py` - Basic chatbot without RAG

### Testing & Examples (`testing/`)
- `example_folder_rag.py` - View collection statistics and usage examples
- `demo_auto_metadata.py` - Demonstrate metadata extraction

## Example Usage

### Terminal Chat
```bash
$ python src/chat_rag.py
Welcome to the AI Chatbot with RAG! (Using OPENAI: gpt-4o)
Loading RAG system...
RAG system ready!

You: What are the benefits of ketamine for depression?
[Retrieved context from transcripts]
AI: Based on the transcripts, ketamine has several notable benefits for treating depression:

1. **Rapid Relief**: Ketamine provides immediate relief from depressive symptoms...
[response continues with context from transcripts]
```

### Switching LLM Providers
```bash
# Use OpenAI (default)
python src/chat_rag.py

# Use Anthropic Claude
export LLM_PROVIDER=anthropic
python src/chat_rag.py
```

### Adding Transcripts
```bash
# Automatic metadata extraction (recommended)
$ python src/smart_add_to_rag.py --data-dir

# Add specific folder
$ python src/smart_add_to_rag.py --folder data/andrew_huberman

# Manual metadata entry
$ python src/add_folder_to_rag.py --folder data/transcripts --author "Author Name"
```

## Architecture

The RAG system uses:
- **OpenAI GPT-4o** (default) or **Anthropic Claude Sonnet 4.5** for chat responses
- **ChromaDB** for vector storage
- **sentence-transformers** (`all-MiniLM-L6-v2`) for embeddings
- **~500 word chunks** with 50 word overlap for optimal context retrieval
- **Flask** for the web interface

See `CLAUDE.md` for detailed architecture documentation.
