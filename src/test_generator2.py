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
        max_workers = 4  # Number of parallel LLM calls

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

            # Generate questions for each type
            for question_type in question_types:
                questions = self._generate_questions_for_transcript(
                    content=content,
                    transcript_info=transcript_info,
                    question_type=question_type,
                    num_questions=questions_per_type
                )
                test_cases.extend(questions)

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

    def _create_factual_prompt(self, content: str, num: int) -> str:
        """Create prompt for factual questions."""
        return f"""Based on the following transcript, generate {num} specific factual questions that ask about knowledge, facts, recommendations, or mechanisms.

IMPORTANT INSTRUCTIONS:
- Focus on the KNOWLEDGE and INFORMATION presented, not on the episode/conversation itself
- DO NOT assume this is from a podcast episode or interview
- DO NOT ask about "this episode", "this conversation", "the interviewer", "the guest", etc.
- Ask about the TOPIC and CONTENT, not the format or medium
- Questions should be timeless and transferable beyond a single conversation

BAD EXAMPLES (avoid these):
❌ "What company mentioned in the episode was founded by two swimmers?"
❌ "What is the primary purpose of this episode?"
❌ "What does the interviewer discuss about X?"

GOOD EXAMPLES (generate questions like these):
✓ "What are the key mechanisms by which X affects Y?"
✓ "What specific recommendations are given for optimizing Z?"
✓ "What dose ranges are discussed for compound X?"

Transcript:
{content}

Generate {num} factual questions in this format:
1. [Question about specific knowledge, facts, or mechanisms]
2. [Question about specific recommendations or protocols]
3. [Question about scientific findings or data]

Make questions focused on knowledge and information, not on the conversation format."""

    def _create_comprehension_prompt(self, content: str, num: int) -> str:
        """Create prompt for comprehension questions."""
        return f"""Based on the following transcript, generate {num} comprehension questions that test understanding of main ideas, concepts, or relationships.

IMPORTANT INSTRUCTIONS:
- Focus on UNDERSTANDING the knowledge and concepts, not on the conversation itself
- DO NOT reference "this episode", "this podcast", "the interview", "the discussion", etc.
- Ask about the IDEAS and CONCEPTS presented, not the presentation format
- Questions should focus on synthesizing knowledge, not on who said what or when
- Go beyond the current content - ask about broader implications and applications

BAD EXAMPLES (avoid these):
❌ "What is the central message of this episode?"
❌ "How does the speaker structure their argument in this conversation?"
❌ "What point does the interviewer make about X?"

GOOD EXAMPLES (generate questions like these):
✓ "How does X relate to Y in terms of biological mechanisms?"
✓ "What is the relationship between sleep deprivation and cognitive function?"
✓ "How do these concepts integrate to explain Z?"

Transcript:
{content}

Generate {num} comprehension questions in this format:
1. [Question about understanding relationships between concepts]
2. [Question requiring synthesis of multiple ideas]
3. [Question about broader implications or applications]

Make questions test deep understanding of knowledge and concepts."""

    def _create_application_prompt(self, content: str, num: int) -> str:
        """Create prompt for application questions."""
        return f"""Based on the following transcript, generate {num} application questions that ask how to apply the knowledge, implement recommendations, or use the information in practical scenarios.

IMPORTANT INSTRUCTIONS:
- Focus on APPLYING the knowledge presented, not on the conversation format
- DO NOT ask about "what was discussed in the episode" or similar meta-questions
- Ask about PRACTICAL APPLICATION of the ideas and recommendations
- Questions should be about HOW TO USE the knowledge, not about the source material
- Think beyond the specific context - ask about real-world application

BAD EXAMPLES (avoid these):
❌ "Based on this episode, what should you do about X?"
❌ "What does the guest recommend implementing from this conversation?"
❌ "What action steps are outlined in this interview?"

GOOD EXAMPLES (generate questions like these):
✓ "What practical steps can someone take to optimize their sleep schedule?"
✓ "How can cold exposure be incorporated into a weekly routine safely?"
✓ "What protocol is recommended for using light therapy to adjust circadian rhythm?"

Transcript:
{content}

Generate {num} application questions in this format:
1. [Question about practical implementation of a concept or recommendation]
2. [Question about real-world application scenarios]
3. [Question about how to use specific protocols or techniques]

Make questions practical, action-oriented, and focused on knowledge application."""

    def _create_specific_prompt(self, content: str, transcript_info: Dict, num: int) -> str:
        """Create prompt for author-specific questions."""
        author = transcript_info["author"]
        keywords = transcript_info["keywords"]

        return f"""Based on the following content from {author}, generate {num} questions that ask about {author}'s knowledge, opinions, perspectives, or recommendations on {keywords}.

IMPORTANT INSTRUCTIONS:
- Ask about {author}'s KNOWLEDGE, OPINIONS, and EXPERTISE, not about a specific episode/conversation
- DO NOT assume this is from a single podcast episode or interview
- DO NOT ask about "this episode", "this conversation", "this interview", "the guest", "the interviewer"
- Focus on {author}'s BODY OF WORK and EXPERTISE, which may span multiple discussions
- Questions should be about the substantive knowledge and viewpoints, not the format

BAD EXAMPLES (avoid these):
❌ "What company mentioned in the episode was founded by two swimmers?"
❌ "What is the primary purpose of this episode with {author}?"
❌ "What does the interviewer ask {author} about?"
❌ "What topic does {author} discuss in this conversation?"

GOOD EXAMPLES (generate questions like these):
✓ "What does {author} recommend for optimizing sleep quality?"
✓ "According to {author}, what are the key mechanisms of neuroplasticity?"
✓ "What is {author}'s view on the relationship between exercise and mental health?"
✓ "What protocols does {author} suggest for managing stress?"

Transcript:
{content}

Generate {num} author-specific questions in this format:
1. What does {author} say about [specific knowledge or mechanism]?
2. According to {author}, what are the recommendations for [specific topic]?
3. What is {author}'s perspective on [specific concept or relationship]?

Make questions reference the author's KNOWLEDGE and OPINIONS explicitly.
Use {author}'s name, not generic terms like "the speaker", "the expert", or "the interviewer".
Focus on substantive knowledge that goes beyond a single conversation."""

    def _generate_with_llm(self, prompt: str, max_tokens: int = 800) -> str:
        """Generate response using LLM."""
        messages = [{"role": "user", "content": prompt}]
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
