"""
RAG System Evaluation Engine using LLM-as-Judge methodology.

This module provides classes for evaluating both retrieval quality and response quality
in a RAG (Retrieval-Augmented Generation) system. It uses an LLM to judge the quality
of retrieved chunks and generated responses.
"""

import re
import time
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from llm_client import LLMClient, get_llm_client


class RetrievalEvaluator:
    """Evaluates the quality of retrieved chunks using LLM-as-judge."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the retrieval evaluator.

        Args:
            llm_client: LLM client to use for evaluation (creates default if None)
        """
        self.llm = llm_client or get_llm_client()

    def evaluate_chunk_relevance(self, query: str, chunk: str, metadata: dict) -> Dict[str, Any]:
        """
        Evaluate the relevance of a single retrieved chunk using LLM-as-judge.

        Args:
            query: The user's query
            chunk: The retrieved text chunk
            metadata: Metadata about the chunk (source, author, etc.)

        Returns:
            Dictionary with 'score' (0-5), 'explanation', and 'metadata'
        """
        # Create evaluation prompt
        prompt = f"""You are evaluating the relevance of retrieved text chunks for a RAG system.

Question: {query}

Retrieved Chunk:
{chunk}

Metadata:
- Source: {metadata.get('source', 'Unknown')}
- Author: {metadata.get('author', 'Unknown')}

Rate the relevance of this chunk on a scale of 0-5:
0 = Completely irrelevant
1 = Tangentially related
2 = Somewhat relevant but lacks key information
3 = Moderately relevant, contains some useful information
4 = Highly relevant, contains most needed information
5 = Perfectly relevant, directly answers the question

Provide your rating and a brief explanation (1-2 sentences).

Output format:
Score: [0-5]
Explanation: [brief explanation]"""

        # Get LLM judgment
        messages = [{"role": "user", "content": prompt}]
        response = ""
        for chunk_text in self.llm.stream_chat(messages, max_tokens=300):
            response += chunk_text

        # Parse response
        score = self._parse_score(response)
        explanation = self._parse_explanation(response)

        return {
            "score": score,
            "explanation": explanation,
            "metadata": metadata
        }

    def evaluate_retrieval_set(self, query: str, chunks: List[Tuple[str, dict]]) -> Dict[str, Any]:
        """
        Evaluate all retrieved chunks for a query.

        Args:
            query: The user's query
            chunks: List of (chunk_text, metadata) tuples

        Returns:
            Dictionary with chunk evaluations and aggregate metrics
        """
        chunk_evaluations = []

        for i, (chunk_text, metadata) in enumerate(chunks):
            eval_result = self.evaluate_chunk_relevance(query, chunk_text, metadata)
            eval_result['rank'] = i + 1
            chunk_evaluations.append(eval_result)

        # Calculate aggregate metrics
        scores = [e['score'] for e in chunk_evaluations]
        avg_relevance = sum(scores) / len(scores) if scores else 0

        # Precision@K: percentage of chunks with score >= 3
        relevant_count = sum(1 for s in scores if s >= 3)
        precision_at_k = relevant_count / len(scores) if scores else 0

        # Mean Reciprocal Rank (MRR): position of first relevant chunk
        mrr = 0
        for i, score in enumerate(scores, 1):
            if score >= 3:
                mrr = 1 / i
                break

        return {
            "chunk_evaluations": chunk_evaluations,
            "avg_relevance": avg_relevance,
            "precision_at_k": precision_at_k,
            "mrr": mrr,
            "num_chunks": len(chunks)
        }

    def calculate_recall(self, retrieved_sources: List[str], expected_sources: List[str]) -> float:
        """
        Calculate recall: percentage of expected sources that were retrieved.

        Args:
            retrieved_sources: List of source filenames retrieved
            expected_sources: List of source filenames expected

        Returns:
            Recall score (0.0 to 1.0)
        """
        if not expected_sources:
            return 1.0  # No expected sources, perfect recall

        retrieved_set = set(retrieved_sources)
        expected_set = set(expected_sources)

        matches = len(retrieved_set & expected_set)
        return matches / len(expected_set)

    def _parse_score(self, response: str) -> float:
        """Extract score from LLM response."""
        match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return max(0, min(5, score))  # Clamp to 0-5 range
        return 0.0

    def _parse_explanation(self, response: str) -> str:
        """Extract explanation from LLM response."""
        match = re.search(r'Explanation:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return "No explanation provided"


class ResponseEvaluator:
    """Evaluates the quality of generated responses using LLM-as-judge."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the response evaluator.

        Args:
            llm_client: LLM client to use for evaluation (creates default if None)
        """
        self.llm = llm_client or get_llm_client()

    def evaluate_response(
        self,
        query: str,
        context: str,
        response: str,
        ground_truth_aspects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate response quality using multi-dimensional LLM-as-judge.

        Args:
            query: The user's query
            context: The retrieved context provided to the LLM
            response: The generated response
            ground_truth_aspects: Expected aspects to cover (optional)

        Returns:
            Dictionary with scores for accuracy, completeness, faithfulness, relevance
        """
        aspects_text = ""
        if ground_truth_aspects:
            aspects_text = f"\nExpected Aspects: {', '.join(ground_truth_aspects)}"

        prompt = f"""You are evaluating an AI assistant's response quality for a RAG-enhanced Q&A system.

Question: {query}

Retrieved Context:
{context}

AI Response:
{response}{aspects_text}

Evaluate the response on these dimensions (0-5 scale each):

1. ACCURACY: Does the response correctly answer the question?
2. COMPLETENESS: Does it cover all important aspects?
3. FAITHFULNESS: Is the response grounded in the provided context (not hallucinated)?
4. RELEVANCE: Does the response directly address the question?

For each dimension, provide:
- Score (0-5)
- Brief explanation (1-2 sentences)

Output format:
ACCURACY: [score] - [explanation]
COMPLETENESS: [score] - [explanation]
FAITHFULNESS: [score] - [explanation]
RELEVANCE: [score] - [explanation]
OVERALL: [overall summary in 1-2 sentences]"""

        # Get LLM judgment
        messages = [{"role": "user", "content": prompt}]
        eval_response = ""
        for chunk in self.llm.stream_chat(messages, max_tokens=600):
            eval_response += chunk

        # Parse response
        scores = {
            "accuracy": self._parse_dimension_score(eval_response, "ACCURACY"),
            "completeness": self._parse_dimension_score(eval_response, "COMPLETENESS"),
            "faithfulness": self._parse_dimension_score(eval_response, "FAITHFULNESS"),
            "relevance": self._parse_dimension_score(eval_response, "RELEVANCE")
        }

        explanations = {
            "accuracy": self._parse_dimension_explanation(eval_response, "ACCURACY"),
            "completeness": self._parse_dimension_explanation(eval_response, "COMPLETENESS"),
            "faithfulness": self._parse_dimension_explanation(eval_response, "FAITHFULNESS"),
            "relevance": self._parse_dimension_explanation(eval_response, "RELEVANCE")
        }

        overall = self._parse_overall(eval_response)

        # Calculate average score
        avg_score = sum(scores.values()) / len(scores)

        return {
            "scores": scores,
            "explanations": explanations,
            "overall": overall,
            "avg_score": avg_score
        }

    def detect_hallucinations(self, context: str, response: str) -> Dict[str, Any]:
        """
        Detect potential hallucinations by checking if response claims are supported by context.

        Args:
            context: The retrieved context
            response: The generated response

        Returns:
            Dictionary with hallucination detection results
        """
        prompt = f"""Compare the AI response against the provided context.
Identify any claims in the response that are NOT supported by the context.

Context:
{context}

Response:
{response}

List any unsupported claims (hallucinations):
1. [claim] - NOT FOUND in context
2. [claim] - NOT FOUND in context

If all claims are supported, output: "No hallucinations detected"

Be strict: only mark something as hallucinated if it's a factual claim not present in the context.
General statements or reasoning based on the context are acceptable."""

        messages = [{"role": "user", "content": prompt}]
        detection_response = ""
        for chunk in self.llm.stream_chat(messages, max_tokens=400):
            detection_response += chunk

        # Check if hallucinations were detected
        has_hallucinations = "no hallucinations detected" not in detection_response.lower()

        # Extract hallucination list
        hallucinations = []
        if has_hallucinations:
            lines = detection_response.split('\n')
            for line in lines:
                if re.match(r'^\d+\.', line.strip()):
                    hallucinations.append(line.strip())

        return {
            "has_hallucinations": has_hallucinations,
            "hallucinations": hallucinations,
            "raw_response": detection_response
        }

    def _parse_dimension_score(self, response: str, dimension: str) -> float:
        """Extract score for a specific dimension."""
        pattern = f"{dimension}:\\s*(\\d+(?:\\.\\d+)?)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return max(0, min(5, score))
        return 0.0

    def _parse_dimension_explanation(self, response: str, dimension: str) -> str:
        """Extract explanation for a specific dimension."""
        pattern = f"{dimension}:\\s*\\d+(?:\\.\\d+)?\\s*-\\s*(.+?)(?=\\n[A-Z]+:|$)"
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return "No explanation provided"

    def _parse_overall(self, response: str) -> str:
        """Extract overall assessment."""
        match = re.search(r'OVERALL:\\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return "No overall assessment provided"


class EvaluationRunner:
    """Orchestrates the full evaluation process for RAG system."""

    def __init__(
        self,
        rag_system,
        llm_client: Optional[LLMClient] = None,
        response_llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize the evaluation runner.

        Args:
            rag_system: TranscriptRAG instance
            llm_client: LLM client for evaluation (creates default if None)
            response_llm_client: Separate LLM for response generation (uses llm_client if None)
        """
        self.rag = rag_system
        self.eval_llm = llm_client or get_llm_client()
        self.response_llm = response_llm_client or self.eval_llm

        self.retrieval_eval = RetrievalEvaluator(self.eval_llm)
        self.response_eval = ResponseEvaluator(self.eval_llm)

    def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run evaluation for a single test case.

        Args:
            test_case: Dictionary with test case details (question, category, etc.)

        Returns:
            Complete evaluation results for this test case
        """
        start_time = time.time()

        question = test_case['question']
        author_filter = test_case.get('author_filter')
        expected_sources = test_case.get('expected_sources', [])
        ground_truth_aspects = test_case.get('ground_truth_aspects', [])

        # Step 1: Retrieve chunks from RAG system
        retrieval_start = time.time()
        n_results = test_case.get('n_results', 3)
        chunks = self.rag.query(question, n_results=n_results, author_filter=author_filter)
        retrieval_time = time.time() - retrieval_start

        # Step 2: Evaluate retrieval quality
        retrieval_results = self.retrieval_eval.evaluate_retrieval_set(question, chunks)

        # Calculate recall if expected sources provided
        if expected_sources:
            retrieved_sources = [metadata.get('source', '') for _, metadata in chunks]
            retrieval_results['recall'] = self.retrieval_eval.calculate_recall(
                retrieved_sources,
                expected_sources
            )

        # Step 3: Generate response using RAG workflow
        response_start = time.time()
        context = self.rag.get_context_for_query(question, n_results=n_results, author_filter=author_filter)
        enhanced_message = f"{context}\n\nUser question: {question}"

        messages = [{"role": "user", "content": enhanced_message}]
        response = ""
        for chunk in self.response_llm.stream_chat(messages, max_tokens=2000):
            response += chunk
        response_time = time.time() - response_start

        # Step 4: Evaluate response quality
        response_results = self.response_eval.evaluate_response(
            question,
            context,
            response,
            ground_truth_aspects
        )

        # Step 5: Detect hallucinations
        hallucination_results = self.response_eval.detect_hallucinations(context, response)

        # Calculate total time
        total_time = time.time() - start_time

        # Compile full results
        return {
            "test_case": test_case,
            "retrieval": retrieval_results,
            "response": response_results,
            "hallucinations": hallucination_results,
            "generated_response": response,
            "context_used": context,
            "performance": {
                "retrieval_time_ms": retrieval_time * 1000,
                "response_time_ms": response_time * 1000,
                "total_time_ms": total_time * 1000
            }
        }

    def run_batch_evaluation(
        self,
        test_cases: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None,
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation on multiple test cases in parallel.

        Args:
            test_cases: List of test case dictionaries
            progress_callback: Optional callback function(current, total) for progress tracking
            max_workers: Maximum number of parallel workers (default: 4)

        Returns:
            List of evaluation results (in original order)
        """
        total = len(test_cases)

        # If only 1 test case or max_workers is 1, run sequentially
        if len(test_cases) == 1 or max_workers == 1:
            return self._run_sequential(test_cases, progress_callback)

        # Parallel execution
        results = [None] * total  # Pre-allocate to maintain order
        completed_count = 0
        progress_lock = Lock()

        def run_with_index(index: int, test_case: Dict[str, Any]) -> tuple:
            """Run single test and return with index for ordering."""
            try:
                result = self.run_single_test(test_case)
                return (index, result, None)
            except Exception as e:
                error_msg = f"Error evaluating test case {test_case.get('id', 'unknown')}: {e}"
                print(f"\n{error_msg}")
                return (index, {
                    "test_case": test_case,
                    "error": str(e)
                }, error_msg)

        # Submit all tasks to thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test cases with their indices
            future_to_index = {
                executor.submit(run_with_index, i, tc): i
                for i, tc in enumerate(test_cases)
            }

            # Process completed tasks
            for future in as_completed(future_to_index):
                index, result, error = future.result()
                results[index] = result

                # Update progress (thread-safe)
                with progress_lock:
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, total)

        return results

    def _run_sequential(
        self,
        test_cases: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Run evaluation sequentially (fallback for single test or max_workers=1)."""
        results = []
        total = len(test_cases)

        for i, test_case in enumerate(test_cases, 1):
            try:
                result = self.run_single_test(test_case)
                results.append(result)

                if progress_callback:
                    progress_callback(i, total)
            except Exception as e:
                # Log error but continue with other tests
                print(f"Error evaluating test case {test_case.get('id', 'unknown')}: {e}")
                results.append({
                    "test_case": test_case,
                    "error": str(e)
                })

        return results
