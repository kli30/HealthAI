"""
Test Dataset Generator for RAG System Evaluation.

Generates test questions based on complete transcripts rather than chunks or topics.
Each question is grounded in the actual content of specific transcript files.
"""

import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from pathlib import Path
from glob import glob

# Add src directory to path if needed
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from llm_client import get_llm_client, LLMClient
from metadata_extractor import extract_all_metadata
from tqdm import tqdm


class TranscriptBasedTestGenerator:
    """Generates test datasets based on complete transcript files."""

    def __init__(
        self,
        data_dir: str = "../data",
        llm_client: Optional[LLMClient] = None,
        max_transcript_length: int = 8000
    ):
        """
        Initialize the transcript-based test generator.

        Args:
            data_dir: Directory containing transcript files
            llm_client: LLM client for generating questions (creates default if None)
            max_transcript_length: Maximum characters to send to LLM (for long transcripts)
        """
        self.data_dir = Path(data_dir)
        self.llm = llm_client or get_llm_client()
        self.max_transcript_length = max_transcript_length

        # Author-specific question counts
        self.author_question_counts = {
            "Andrew Huberman": (5, 10),  # (min, max) questions for Huberman
            "default": (2, 3)             # (min, max) questions for other authors
        }

        # Discover transcript files
        self.transcripts = self._discover_transcripts()

        if not self.transcripts:
            print(f"Warning: No transcript files found in {data_dir}")
            print(f"Supported extensions: .txt, .md")

    def _discover_transcripts(self) -> List[Dict[str, Any]]:
        """Discover all transcript files in the data directory."""
        transcripts = []

        if not self.data_dir.exists():
            print(f"Warning: Data directory not found: {self.data_dir}")
            return transcripts

        # Search for text files recursively
        patterns = ["**/*.txt", "**/*.md"]

        for pattern in patterns:
            for file_path in self.data_dir.glob(pattern):
                # Extract metadata
                metadata = extract_all_metadata(str(file_path))

                # Read file size
                file_size = os.path.getsize(file_path)

                transcripts.append({
                    "path": str(file_path),
                    "filename": file_path.name,
                    "author": metadata.get("author", "Unknown"),
                    "keywords": metadata.get("keywords", ""),
                    "size_bytes": file_size,
                    "metadata": metadata
                })

        print(f"Discovered {len(transcripts)} transcript files")
        if transcripts:
            authors = set(t["author"] for t in transcripts)
            print(f"Authors: {', '.join(sorted(authors))}")

        return transcripts

    def generate_dataset(
        self,
        questions_per_transcript: Optional[int] = None,
        question_types: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate test dataset based on complete transcripts.

        Question counts are author-specific:
        - Andrew Huberman: 5-10 questions (randomly selected)
        - Other authors: 2-3 questions (randomly selected)

        Args:
            questions_per_transcript: Override for number of questions (optional, ignores author-specific counts)
            question_types: Types of questions to generate (default: all types)

        Returns:
            Dictionary with 'test_cases' list
        """
        if not self.transcripts:
            print("Error: No transcripts found. Cannot generate dataset.")
            return {"test_cases": []}

        if question_types is None:
            question_types = ["factual", "comprehension", "application", "specific"]

        types_count = len(question_types)

        # Display configuration
        print(f"\nQuestion Generation Strategy:")
        print(f"  - Andrew Huberman transcripts: 5-10 questions each")
        print(f"  - Other authors: 2-3 questions each")
        print(f"Question types: {', '.join(question_types)}")
        print(f"Total transcripts: {len(self.transcripts)}")
        print()

        test_cases = []

        # Use ThreadPoolExecutor for parallel processing
        max_workers = 10  # Number of parallel LLM calls

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all transcripts for processing
            future_to_transcript = {
                executor.submit(
                    self._process_single_transcript,
                    transcript_info,
                    question_types,
                    types_count,
                    questions_per_transcript
                ): transcript_info
                for transcript_info in self.transcripts
            }

            # Process completed futures with progress bar
            with tqdm(total=len(self.transcripts), desc="Processing transcripts", unit="transcript") as pbar:
                for future in as_completed(future_to_transcript):
                    transcript_info = future_to_transcript[future]
                    try:
                        result = future.result()
                        if result:
                            test_cases.extend(result)
                    except Exception as e:
                        print(f"\nError processing {transcript_info['filename']}: {e}")
                    finally:
                        pbar.update(1)

        # Assign IDs
        for i, test_case in enumerate(test_cases, 1):
            test_case['id'] = f"test_{i:04d}"

        return {"test_cases": test_cases}

    def _process_single_transcript(
        self,
        transcript_info: Dict[str, Any],
        question_types: List[str],
        types_count: int,
        questions_per_transcript: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Process a single transcript and generate questions.
        This method is designed to be called in parallel.

        Args:
            transcript_info: Transcript metadata
            question_types: Types of questions to generate
            types_count: Number of question types
            questions_per_transcript: Optional override for question count

        Returns:
            List of generated test cases for this transcript
        """
        test_cases = []

        try:
            # Read transcript content
            with open(transcript_info["path"], 'r', encoding='utf-8') as f:
                content = f.read()

            # Truncate if too long
            if len(content) > self.max_transcript_length:
                content = content[:self.max_transcript_length] + "\n\n[Content truncated for length...]"

            # Determine number of questions based on author
            author = transcript_info["author"]
            if questions_per_transcript is not None:
                # Override provided
                num_questions = questions_per_transcript
            else:
                # Use author-specific counts
                min_q, max_q = self.author_question_counts.get(
                    author,
                    self.author_question_counts["default"]
                )
                num_questions = random.randint(min_q, max_q)

            questions_per_type = max(1, num_questions // types_count)

            # Debug output (can be removed later)
            print(f"  [{author}] {transcript_info['filename'][:50]}: {num_questions} questions ({questions_per_type} per type)")

            # Generate all questions in a single API call
            all_questions = self._generate_all_questions_for_transcript(
                content=content,
                transcript_info=transcript_info,
                question_types=question_types,
                questions_per_type=questions_per_type
            )
            test_cases.extend(all_questions)

        except Exception as e:
            raise Exception(f"Failed to process {transcript_info['filename']}: {str(e)}")

        return test_cases

    def _generate_questions_for_transcript(
        self,
        content: str,
        transcript_info: Dict[str, Any],
        question_type: str,
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """Generate questions of a specific type for a transcript."""

        # Create type-specific prompt
        prompts = {
            "factual": self._create_factual_prompt(content, num_questions),
            "comprehension": self._create_comprehension_prompt(content, num_questions),
            "application": self._create_application_prompt(content, num_questions),
            "specific": self._create_specific_prompt(content, transcript_info, num_questions)
        }

        prompt = prompts.get(question_type, prompts["factual"])

        # Generate questions
        response = self._generate_with_llm(prompt)

        # Parse questions
        parsed_questions = self._parse_question_list(response)

        # Create test case objects
        test_cases = []
        for question in parsed_questions[:num_questions]:
            test_cases.append({
                "question": question,
                "category": question_type,
                "source_file": transcript_info["filename"],
                "source_path": transcript_info["path"],
                "author": transcript_info["author"],
                "expected_sources": [transcript_info["filename"]],
                "author_filter": transcript_info["author"] if transcript_info["author"] != "Unknown" else None,
                "difficulty": self._get_difficulty(question_type),
                "ground_truth_aspects": [],
                "keywords": transcript_info["keywords"]
            })

        return test_cases

    def _generate_all_questions_for_transcript(
        self,
        content: str,
        transcript_info: Dict[str, Any],
        question_types: List[str],
        questions_per_type: int
    ) -> List[Dict[str, Any]]:
        """Generate all question types for a transcript in a single API call."""
        author = transcript_info["author"]
        keywords = transcript_info["keywords"]
        source_file = transcript_info["filename"]

        # Build comprehensive prompt for all question types
        prompt = f"""Based on the following transcript, generate questions across multiple categories.

TRANSCRIPT:
{content}

---

Generate {questions_per_type} questions for EACH of the following categories:

"""

        # Add category-specific instructions
        if "factual" in question_types:
            prompt += f"""
1. FACTUAL QUESTIONS ({questions_per_type} questions):
   - Ask about specific knowledge, facts, recommendations, or mechanisms
   - Use natural, conversational phrasing - ask questions as people naturally would
   - Keep it simple - ONE concept per question, avoid chaining with "and"
   - Example: "What role do astrocytes play in synaptic transmission?"
   - Example: "What is the half-life of caffeine?"
   - Example: "How does omega-3 affect triglyceride levels?"
   - AVOID: "According to the study...", "The research shows...", etc.

"""

        if "comprehension" in question_types:
            prompt += f"""
2. COMPREHENSION QUESTIONS ({questions_per_type} questions):
   - Test understanding of relationships between concepts
   - Use simple, direct language - how would someone naturally ask this?
   - Focus on ONE relationship at a time
   - Example: "Why do certain diseases go away during pregnancy?"
   - Example: "What are the benefits of omega-3 fatty acids?"
   - Example: "How does sleep affect cognitive function?"
   - AVOID: Complex multi-part questions, contextual references

"""

        if "application" in question_types:
            prompt += f"""
3. APPLICATION QUESTIONS ({questions_per_type} questions):
   - Ask how to apply knowledge practically
   - Use conversational, everyday language
   - Focus on ONE practical application per question
   - Example: "How can someone optimize their sleep schedule?"
   - Example: "What is the recommended dose of melatonin?"
   - Example: "How can cold exposure be used safely?"
   - AVOID: "What practical steps..." - just ask directly

"""

        if "specific" in question_types:
            prompt += f"""
4. SPECIFIC QUESTIONS ({questions_per_type} questions):
   - Generate questions focusing on transcript title: {source_file}, with helps from the transcript content
   - Use natural, conversational phrasing
   - Do NOT reference {author} by name
   - Keep it simple and direct
   - Example: for "Is Medicine the Answer", ask "Is medicine the answer?"
   - Example: for "Best Exercise for Depression", ask "What is the best exercise for depression?"
   - Example: for "Using Light to Optimize Health", ask "How can light optimize health?"
   - AVOID: Complex phrasing, multiple concepts in one question

"""

        prompt += """
FORMAT YOUR RESPONSE EXACTLY AS:

FACTUAL:
1. [question]
2. [question]
...

COMPREHENSION:
1. [question]
2. [question]
...

APPLICATION:
1. [question]
2. [question]
...

SPECIFIC:
1. [question]
2. [question]
...

(Only include categories that were requested above)
"""

        # Generate questions with single API call
        response = self._generate_with_llm(prompt, max_tokens=1500)

        # Parse the categorized response
        test_cases = self._parse_categorized_questions(
            response,
            transcript_info
        )

        return test_cases

    def _parse_categorized_questions(
        self,
        response: str,
        transcript_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse questions organized by category from LLM response."""
        test_cases = []
        current_category = None

        lines = response.split('\n')

        for line in lines:
            line = line.strip()

            # Detect category headers
            line_upper = line.upper()
            if 'FACTUAL' in line_upper and ':' in line:
                current_category = 'factual'
                continue
            elif 'COMPREHENSION' in line_upper and ':' in line:
                current_category = 'comprehension'
                continue
            elif 'APPLICATION' in line_upper and ':' in line:
                current_category = 'application'
                continue
            elif 'SPECIFIC' in line_upper and ':' in line or 'AUTHOR' in line_upper and ':' in line:
                current_category = 'specific'
                continue

            # Parse question lines
            if current_category and line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                question = line.lstrip('0123456789.-) ').strip()
                if question and len(question) > 15:  # Filter out too-short questions
                    # Clean up
                    if not question.endswith('?'):
                        question += '?'

                    test_cases.append({
                        "question": question,
                        "category": current_category,
                        "source_file": transcript_info["filename"],
                        "source_path": transcript_info["path"],
                        "author": transcript_info["author"],
                        "expected_sources": [transcript_info["filename"]],
                        "author_filter": transcript_info["author"] if transcript_info["author"] != "Unknown" else None,
                        "difficulty": self._get_difficulty(current_category),
                        "ground_truth_aspects": [],
                        "keywords": transcript_info["keywords"]
                    })

        return test_cases

    def _create_factual_prompt(self, content: str, num: int) -> str:
        """Create prompt for factual questions."""
        return f"""Generate {num} factual questions that ask about specific knowledge, facts, recommendations, or mechanisms.

Focus on:
- Key mechanisms and how things work
- Specific recommendations or protocols
- Scientific findings or data
- Dose ranges, timings, or quantitative information

Examples:
✓ "What are the key mechanisms by which X affects Y?"
✓ "What specific recommendations are given for optimizing Z?"
✓ "What dose ranges are discussed for compound X?"

Transcript:
{content}

Generate {num} factual questions."""

    def _create_comprehension_prompt(self, content: str, num: int) -> str:
        """Create prompt for comprehension questions."""
        return f"""Generate {num} comprehension questions that test understanding of main ideas, concepts, or relationships.

Focus on:
- Relationships between concepts or mechanisms
- Synthesis of multiple ideas
- Broader implications or applications
- How different pieces of information connect

Examples:
✓ "How does X relate to Y in terms of biological mechanisms?"
✓ "What is the relationship between sleep deprivation and cognitive function?"
✓ "How do these concepts integrate to explain Z?"

Transcript:
{content}

Generate {num} comprehension questions."""

    def _create_application_prompt(self, content: str, num: int) -> str:
        """Create prompt for application questions."""
        return f"""Generate {num} application questions that ask how to apply the knowledge, implement recommendations, or use the information in practical scenarios.

Focus on:
- Practical implementation steps
- Real-world application scenarios
- How to use specific protocols or techniques
- Action-oriented recommendations

Examples:
✓ "What practical steps can someone take to optimize their sleep schedule?"
✓ "How can cold exposure be incorporated into a weekly routine safely?"
✓ "What protocol is recommended for using light therapy to adjust circadian rhythm?"

Transcript:
{content}

Generate {num} application questions."""

    def _create_specific_prompt(self, content: str, transcript_info: Dict, num: int) -> str:
        """Create prompt for author-specific questions."""
        author = transcript_info["author"]
        keywords = transcript_info["keywords"]

        return f"""Generate {num} questions that ask about {author}'s knowledge, opinions, perspectives, or recommendations related to: {keywords}

Requirements:
- Explicitly reference {author} by name in each question
- Focus on {author}'s expertise and viewpoints
- Ask one focused question at a time

Examples:
✓ "What does {author} recommend for [specific topic]?"
✓ "According to {author}, what are the key mechanisms of [concept]?"
✓ "What is {author}'s perspective on [relationship/concept]?"

Transcript:
{content}

Generate {num} author-specific questions (use {author}'s name explicitly)."""

    def _generate_with_llm(self, prompt: str, max_tokens: int = 800) -> str:
        """Generate response using LLM."""
        system_prompt = """You are an expert question generator for a RAG (Retrieval-Augmented Generation) system evaluation framework.

Your task is to generate high-quality test questions based on transcript content from domain experts.

CORE REQUIREMENTS:
1. Generate questions that test knowledge retrieval, not conversation recall
2. Focus on the CONTENT and KNOWLEDGE presented, not the format or medium
3. Questions should be timeless and transferable BEYOND a single conversation or a podcast
4. Ask about facts, concepts, mechanisms, recommendations, and applications
5. Avoid meta-questions about the episode, conversation, interview, or speaker
6. Generate clear, specific, and answerable questions
7. Each question should be self-contained and unambiguous
8. Questions shall be diverse, covering different aspects of the transcript, and not too similar to each other
9. Questions should be appropriate for testing a RAG system's retrieval accuracy

WHAT TO FOCUS ON:
- Scientific knowledge, mechanisms, and findings
- Expert recommendations and protocols
- Conceptual relationships and principles
- Practical applications and implementation
  

WHAT TO AVOID:
- Questions about "this episode", "this conversation", "this interview", "this podcast", "this transcript"
- Questions about the interviewer, guest, the speaker or conversational dynamics
- Meta-questions about the format or structure of the content
- Vague or ambiguous questions that could have multiple interpretations
- Questions that assume knowledge outside the provided transcript
- Contextual references like "according to the transcript", "in the referenced study", "this research shows", "the study mentioned"
- Overly complex multi-part questions that chain multiple concepts
- Academic/formal phrasing - use natural, conversational language instead

QUESTION STYLE:
- Write questions as a person would naturally ask them in conversation
- Keep questions simple and direct - ask about ONE thing at a time
- Avoid chaining multiple concepts with "and" - split into separate questions
- Use natural phrasing, not academic/formal language
- Bad: "How do magnesium and boron influence vitamin D function and affect lower back health?"
- Good: "How are magnesium and boron related to lower back health?"
- Bad: "According to the transcript, what mechanisms are discussed for X?"
- Good: "What mechanisms are involved in X?"

OUTPUT FORMAT:
- Return a numbered list of questions (1., 2., 3., etc.)
- Each question should be a complete, grammatically correct sentence
- Each question should end with a question mark
- Keep questions concise and natural (typically 8-15 words)

STEPS TO FOLLOW:
1. Read the transcript carefully
2. Identify the main topics and concepts discussed
3. Generate questions based on the CONTENT and KNOWLEDGE presented, following the core requirements
4. Output a numbered list of questions
5. Double check that the questions meet all the requirements


Your questions will be used to evaluate whether a RAG system can correctly retrieve and synthesize information from transcript databases."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = ""

        for chunk in self.llm.stream_chat(messages, max_tokens=max_tokens):
            response += chunk

        return response

    def _parse_question_list(self, response: str) -> List[str]:
        """Parse numbered list of questions from LLM response."""
        questions = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            # Match patterns like "1. Question" or "1) Question"
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                question = line.lstrip('0123456789.-) ').strip()
                if question and len(question) > 15:  # Filter out too-short questions
                    # Clean up any remaining artifacts
                    if not question.endswith('?'):
                        question += '?'
                    questions.append(question)

        return questions

    def _get_difficulty(self, question_type: str) -> str:
        """Get difficulty level for question type."""
        difficulty_map = {
            "factual": "easy",
            "comprehension": "medium",
            "application": "medium",
            "specific": "easy"
        }
        return difficulty_map.get(question_type, "medium")

    def save_dataset(self, dataset: Dict[str, List[Dict[str, Any]]], output_path: str):
        """Save dataset to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"\nDataset saved to: {output_path}")
        print(f"Total test cases: {len(dataset['test_cases'])}")

        # Print statistics
        if dataset['test_cases']:
            by_author = {}
            by_type = {}

            for tc in dataset['test_cases']:
                author = tc.get('author', 'Unknown')
                category = tc.get('category', 'Unknown')

                by_author[author] = by_author.get(author, 0) + 1
                by_type[category] = by_type.get(category, 0) + 1

            print("\nQuestions by author:")
            for author, count in sorted(by_author.items()):
                print(f"  {author}: {count}")

            print("\nQuestions by type:")
            for qtype, count in sorted(by_type.items()):
                print(f"  {qtype}: {count}")


def main():
    """CLI entry point for test dataset generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate test dataset from complete transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with author-specific question counts (Huberman: 5-10, Others: 2-3)
  python test_generator2.py --data-dir ../data

  # Override with fixed question count for all authors
  python test_generator2.py --data-dir ../data --questions 8

  # Generate with specific question types
  python test_generator2.py --data-dir ../data --types factual comprehension

  # Save to custom location
  python test_generator2.py --output testing/test_datasets/my_dataset.json
        """
    )

    parser.add_argument(
        "--data-dir", "-d",
        default="../data",
        help="Directory containing transcript files (default: ../data)"
    )

    parser.add_argument(
        "--output", "-o",
        default="testing/test_datasets/transcript_dataset.json",
        help="Output JSON file path"
    )

    parser.add_argument(
        "--questions", "-n",
        type=int,
        default=None,
        help="Override: Number of questions per transcript (default: author-specific - Huberman: 5-10, Others: 2-3)"
    )

    parser.add_argument(
        "--types", "-t",
        nargs='+',
        choices=["factual", "comprehension", "application", "specific"],
        help="Question types to generate (default: all types)"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=8000,
        help="Maximum transcript length in characters (default: 8000)"
    )

    args = parser.parse_args()

    # Generate dataset
    print("=" * 70)
    print("Transcript-Based Test Dataset Generator")
    print("=" * 70)
    print()

    generator = TranscriptBasedTestGenerator(
        data_dir=args.data_dir,
        max_transcript_length=args.max_length
    )

    dataset = generator.generate_dataset(
        questions_per_transcript=args.questions,
        question_types=args.types
    )

    # Save to file
    if dataset['test_cases']:
        generator.save_dataset(dataset, args.output)
        print("\n✓ Dataset generation complete!")
    else:
        print("\n✗ No test cases generated. Check your data directory and transcript files.")


if __name__ == "__main__":
    main()
