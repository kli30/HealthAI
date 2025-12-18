# Automatic Metadata Extraction for RAG

Automatically extract author, keywords, and topic metadata from your file organization—no manual entry required!

## Quick Start

### 1. Organize Your Files

Create a folder structure where each author has their own subfolder:

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

### 2. Run One Command

```bash
python src/smart_add_to_rag.py --data-dir
```

### 3. Done!

All metadata is automatically extracted and stored:

- **Author**: From parent folder name (`andrew_huberman/file.txt` → "Andrew Huberman")
- **Keywords**: From filename (`ketamine_benefits_depression` → ["ketamine", "benefits", "depression"])
- **Topic**: Inferred from keywords (["ketamine", "depression"] → "medicine")

## What Gets Extracted

### Author
Extracted from the parent folder name:
- `andrew_huberman/file.txt` → Author: "Andrew Huberman"
- `lex-fridman/file.txt` → Author: "Lex Fridman"
- `tim_ferriss_show/file.txt` → Author: "Tim Ferriss Show"
- `data/author_name/file.txt` → Author: "Author Name"

### Keywords
Parsed from the filename:
- `ketamine_benefits_depression_ptsd.txt` → ["ketamine", "benefits", "depression", "ptsd"]
- `Sleep-Optimization-Guide.txt` → ["sleep", "optimization", "guide"]
- `AI and Machine Learning.txt` → ["machine", "learning"]

Stop words and short words are automatically filtered out.

### Topic
Inferred from keywords using smart topic matching:

| Keywords | Inferred Topic |
|----------|---------------|
| ketamine, depression, therapy | medicine |
| sleep, circadian, melatonin | sleep |
| ai, machine, learning | technology |
| brain, neural, plasticity | neuroscience |
| anxiety, stress, meditation | psychology |
| focus, productivity, performance | performance |
| nutrition, diet, supplements | health |

**Available Topics**: neuroscience, psychology, medicine, sleep, performance, technology, health, science

## Usage Examples

### Add Entire Data Directory (Recommended)
```bash
# Process all files in data/ and subdirectories
python src/smart_add_to_rag.py --data-dir
```

### Add Specific Folder
```bash
# Process all files in a specific folder
python src/smart_add_to_rag.py --folder data/andrew_huberman
```

### Add Single File
```bash
# Process a single file
python src/smart_add_to_rag.py --file data/andrew_huberman/sleep.txt
```

### Add with Custom Metadata Override
```bash
# Auto-extract metadata but add podcast name
python src/smart_add_to_rag.py --data-dir --podcast "Huberman Lab"
```

### Recursive Processing
```bash
# Process folder and all subdirectories
python src/smart_add_to_rag.py --folder data --recursive
```

## View Extracted Metadata

### See Demonstration
```bash
python testing/demo_auto_metadata.py
```

This shows:
- How each extraction function works
- Examples with different filename patterns
- Topic inference logic
- Code usage examples

## Programmatic Usage

### Extract Metadata for a File
```python
import sys
sys.path.insert(0, 'src')

from metadata_extractor import extract_all_metadata

file_path = "data/andrew_huberman/ketamine_depression_ptsd.txt"
metadata = extract_all_metadata(file_path)

print(metadata)
# Output:
# {
#     'author': 'Andrew Huberman',
#     'keywords': 'ketamine, depression, ptsd',
#     'topic': 'medicine',
#     'filename': 'ketamine_depression_ptsd.txt',
#     'source_path': 'data/andrew_huberman'
# }
```

### Add to RAG with Auto Metadata
```python
import sys
sys.path.insert(0, 'src')

from metadata_extractor import extract_all_metadata
from rag_system import TranscriptRAG

rag = TranscriptRAG()

# Process multiple files
from pathlib import Path

for file in Path("data/andrew_huberman").glob("*.txt"):
    metadata = extract_all_metadata(str(file))
    metadata['podcast'] = 'Huberman Lab'  # Add custom field
    rag.add_transcript(str(file), metadata=metadata)
```

### Query by Extracted Metadata
```python
import sys
sys.path.insert(0, 'src')

from rag_system import TranscriptRAG

rag = TranscriptRAG()

# Query by author
results = rag.query(
    "What helps with sleep?",
    author_filter="Andrew Huberman"
)

# View all authors
authors = rag.get_authors()
print(f"Available authors: {authors}")
```

## Naming Best Practices

### Folder Names (Authors)
- Use snake_case or kebab-case: `andrew_huberman` or `andrew-huberman`
- Or use spaces: `Andrew Huberman` (will be preserved)
- Avoid special characters

### File Names (Keywords & Topics)
- Use descriptive names: `ketamine_benefits_depression_ptsd.txt`
- Separate words with underscores, hyphens, or spaces
- Include key topics and concepts
- Avoid generic names like `transcript_1.txt`

**Good examples:**
```
✓ ketamine_benefits_depression_ptsd.txt
✓ sleep-optimization-circadian-rhythm.txt
✓ Artificial Intelligence and Machine Learning.txt
✓ dopamine_motivation_focus_performance.txt
```

**Poor examples:**
```
✗ transcript.txt (no keywords)
✗ 123.txt (no meaning)
✗ t1.txt (too short)
✗ file.txt (too generic)
```

## Files

- **`src/metadata_extractor.py`**: Core extraction functions
- **`src/smart_add_to_rag.py`**: Main script to add files with auto-metadata
- **`testing/demo_auto_metadata.py`**: Demonstration and examples
- **`AUTO_METADATA_README.md`**: This file

## Benefits

1. **Zero Configuration**: Just organize files and run
2. **Consistent Metadata**: Automatic extraction ensures consistency
3. **Time Saving**: No manual metadata entry
4. **Scalable**: Process hundreds of files at once
5. **Flexible**: Can override or add custom metadata
6. **Smart**: Topic inference uses keyword matching

## Next Steps

1. Organize your transcript files by author in the `data/` directory
2. Run `python src/smart_add_to_rag.py --data-dir`
3. Query your RAG system with `python src/chat_rag.py` or `python src/web_chat.py`
4. Use author filtering to get more targeted results

For more information, see `CLAUDE.md`.
