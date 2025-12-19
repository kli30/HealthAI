"""
Markdown Report Generator for RAG System Evaluation.

Generates comprehensive, human-readable markdown reports with evaluation
metrics, detailed breakdowns, and actionable recommendations.
"""

from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from collections import defaultdict


class MarkdownReportGenerator:
    """Generates markdown evaluation reports."""

    def generate_report(self, evaluation_results: List[Dict[str, Any]], output_path: str) -> str:
        """
        Generate a comprehensive markdown evaluation report.

        Args:
            evaluation_results: List of evaluation results from EvaluationRunner
            output_path: Path to save the markdown report

        Returns:
            Path to generated report
        """
        # Filter out results with errors
        valid_results = [r for r in evaluation_results if 'error' not in r]
        error_results = [r for r in evaluation_results if 'error' in r]

        # Build report sections
        report_parts = [
            self._create_header(),
            self._create_executive_summary(valid_results),
            self._create_retrieval_section(valid_results),
            self._create_response_section(valid_results),
            self._create_performance_section(valid_results),
            self._create_top_and_worst_cases(valid_results),
            self._create_detailed_breakdown(valid_results),
            self._create_recommendations(valid_results),
        ]

        if error_results:
            report_parts.append(self._create_error_section(error_results))

        report = "\n\n".join(report_parts)

        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write(report)

        return str(output_file)

    def _create_header(self) -> str:
        """Create report header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""# RAG System Evaluation Report

**Generated:** {timestamp}

---"""

    def _create_executive_summary(self, results: List[Dict[str, Any]]) -> str:
        """Create executive summary section."""
        if not results:
            return "## Executive Summary\n\nNo valid evaluation results."

        # Calculate overall metrics
        total_tests = len(results)

        # Retrieval metrics
        avg_relevance = sum(r['retrieval']['avg_relevance'] for r in results) / total_tests
        avg_precision = sum(r['retrieval']['precision_at_k'] for r in results) / total_tests
        avg_mrr = sum(r['retrieval']['mrr'] for r in results) / total_tests

        # Response metrics
        avg_accuracy = sum(r['response']['scores']['accuracy'] for r in results) / total_tests
        avg_completeness = sum(r['response']['scores']['completeness'] for r in results) / total_tests
        avg_faithfulness = sum(r['response']['scores']['faithfulness'] for r in results) / total_tests
        avg_relevance_resp = sum(r['response']['scores']['relevance'] for r in results) / total_tests

        # Hallucination rate
        hallucination_count = sum(1 for r in results if r['hallucinations']['has_hallucinations'])
        hallucination_rate = (hallucination_count / total_tests) * 100

        # Key findings
        findings = self._identify_key_findings(results)

        return f"""## Executive Summary

**Total Test Cases:** {total_tests}

### Overall Scores

| Metric | Score |
|--------|-------|
| **Retrieval Quality** | {avg_relevance:.2f}/5.0 |
| **Response Quality** | {(avg_accuracy + avg_completeness + avg_faithfulness + avg_relevance_resp) / 4:.2f}/5.0 |
| **Hallucination Rate** | {hallucination_rate:.1f}% |
| **Precision@3** | {avg_precision:.2f} |

### Key Findings

{findings}

---"""

    def _create_retrieval_section(self, results: List[Dict[str, Any]]) -> str:
        """Create retrieval performance section."""
        if not results:
            return ""

        total_tests = len(results)

        # Overall metrics
        avg_relevance = sum(r['retrieval']['avg_relevance'] for r in results) / total_tests
        avg_precision = sum(r['retrieval']['precision_at_k'] for r in results) / total_tests
        avg_mrr = sum(r['retrieval']['mrr'] for r in results) / total_tests

        # Calculate recall if available
        recall_results = [r for r in results if 'recall' in r['retrieval']]
        avg_recall = sum(r['retrieval']['recall'] for r in recall_results) / len(recall_results) if recall_results else None

        # Performance by query type
        by_category = defaultdict(list)
        for result in results:
            category = result['test_case'].get('category', 'unknown')
            by_category[category].append(result['retrieval']['avg_relevance'])

        category_table = "\n".join([
            f"| {cat.replace('_', ' ').title()} | {sum(scores)/len(scores):.2f} | {avg_precision:.2f} | {len(scores)} |"
            for cat, scores in sorted(by_category.items())
        ])

        recall_row = f"| Recall | {avg_recall:.2f} |" if avg_recall is not None else ""

        return f"""## Retrieval Performance

### Overall Metrics

| Metric | Score |
|--------|-------|
| Average Relevance | {avg_relevance:.2f}/5.0 |
| Precision@3 | {avg_precision:.2f} |
| Mean Reciprocal Rank (MRR) | {avg_mrr:.2f} |
{recall_row}

### Performance by Query Type

| Query Type | Avg Relevance | Precision@3 | Count |
|------------|---------------|-------------|-------|
{category_table}

---"""

    def _create_response_section(self, results: List[Dict[str, Any]]) -> str:
        """Create response quality section."""
        if not results:
            return ""

        total_tests = len(results)

        # Overall response metrics
        avg_accuracy = sum(r['response']['scores']['accuracy'] for r in results) / total_tests
        avg_completeness = sum(r['response']['scores']['completeness'] for r in results) / total_tests
        avg_faithfulness = sum(r['response']['scores']['faithfulness'] for r in results) / total_tests
        avg_relevance = sum(r['response']['scores']['relevance'] for r in results) / total_tests

        # Hallucination analysis
        hallucination_count = sum(1 for r in results if r['hallucinations']['has_hallucinations'])
        hallucination_rate = (hallucination_count / total_tests) * 100

        return f"""## Response Quality

### Overall Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| Accuracy | {avg_accuracy:.2f}/5.0 | Correctness of the answer |
| Completeness | {avg_completeness:.2f}/5.0 | Coverage of important aspects |
| Faithfulness | {avg_faithfulness:.2f}/5.0 | Grounding in retrieved context |
| Relevance | {avg_relevance:.2f}/5.0 | Addresses the question directly |

### Hallucination Analysis

- **Total Responses:** {total_tests}
- **Responses with Hallucinations:** {hallucination_count}
- **Hallucination Rate:** {hallucination_rate:.1f}%

{self._create_hallucination_details(results)}

---"""

    def _create_performance_section(self, results: List[Dict[str, Any]]) -> str:
        """Create query speed performance section."""
        if not results:
            return ""

        # Check if performance data is available
        if 'performance' not in results[0]:
            return ""

        total_tests = len(results)

        # Calculate performance metrics
        retrieval_times = [r['performance']['retrieval_time_ms'] for r in results]
        response_times = [r['performance']['response_time_ms'] for r in results]
        total_times = [r['performance']['total_time_ms'] for r in results]

        avg_retrieval = sum(retrieval_times) / len(retrieval_times)
        avg_response = sum(response_times) / len(response_times)
        avg_total = sum(total_times) / len(total_times)

        min_retrieval = min(retrieval_times)
        max_retrieval = max(retrieval_times)
        min_response = min(response_times)
        max_response = max(response_times)
        min_total = min(total_times)
        max_total = max(total_times)

        # Performance by category
        by_category = defaultdict(lambda: {'retrieval': [], 'response': [], 'total': []})
        for result in results:
            category = result['test_case'].get('category', 'unknown')
            by_category[category]['retrieval'].append(result['performance']['retrieval_time_ms'])
            by_category[category]['response'].append(result['performance']['response_time_ms'])
            by_category[category]['total'].append(result['performance']['total_time_ms'])

        category_table = "\n".join([
            f"| {cat.replace('_', ' ').title()} | "
            f"{sum(times['retrieval'])/len(times['retrieval']):.0f} | "
            f"{sum(times['response'])/len(times['response']):.0f} | "
            f"{sum(times['total'])/len(times['total']):.0f} | "
            f"{len(times['total'])} |"
            for cat, times in sorted(by_category.items())
        ])

        return f"""## Query Speed Performance

### Overall Metrics

| Metric | Avg (ms) | Min (ms) | Max (ms) |
|--------|----------|----------|----------|
| Retrieval Time | {avg_retrieval:.0f} | {min_retrieval:.0f} | {max_retrieval:.0f} |
| Response Generation | {avg_response:.0f} | {min_response:.0f} | {max_response:.0f} |
| **Total Time** | **{avg_total:.0f}** | **{min_total:.0f}** | **{max_total:.0f}** |

### Performance by Query Type

| Query Type | Avg Retrieval (ms) | Avg Response (ms) | Avg Total (ms) | Count |
|------------|-------------------|-------------------|----------------|-------|
{category_table}

**Notes:**
- Retrieval time includes vector search and chunk retrieval
- Response time includes context formatting and LLM generation
- Total time includes all evaluation steps

---"""

    def _create_hallucination_details(self, results: List[Dict[str, Any]]) -> str:
        """Create detailed hallucination breakdown."""
        hallucinations = [r for r in results if r['hallucinations']['has_hallucinations']]

        if not hallucinations:
            return "**No hallucinations detected** - All responses were grounded in retrieved context."

        details = []
        for i, result in enumerate(hallucinations[:5], 1):  # Show first 5
            test_id = result['test_case'].get('id', 'unknown')
            question = result['test_case']['question']
            hall_list = result['hallucinations']['hallucinations']

            details.append(f"{i}. **{test_id}** - {question}")
            for h in hall_list[:3]:  # Show first 3 hallucinations per case
                details.append(f"   - {h}")

        details_text = "\n".join(details)

        return f"""
**Examples of Hallucinations:**

{details_text}
"""

    def _create_top_and_worst_cases(self, results: List[Dict[str, Any]]) -> str:
        """Create section showing best and worst performing cases."""
        if not results:
            return ""

        # Sort by overall score (average of retrieval relevance and response avg_score)
        scored_results = [
            (
                r,
                (r['retrieval']['avg_relevance'] + r['response']['avg_score']) / 2
            )
            for r in results
        ]

        sorted_results = sorted(scored_results, key=lambda x: x[1], reverse=True)

        # Top 5
        top_5 = sorted_results[:5]
        top_section = "\n".join([
            f"{i}. **{r['test_case'].get('id', 'unknown')}** - \"{r['test_case']['question']}\"\n"
            f"   - Retrieval: {r['retrieval']['avg_relevance']:.1f}/5.0, "
            f"Response: {r['response']['avg_score']:.1f}/5.0, "
            f"Overall: {score:.1f}/5.0"
            for i, (r, score) in enumerate(top_5, 1)
        ])

        # Worst 5
        worst_5 = sorted_results[-5:][::-1]
        worst_section = "\n".join([
            f"{i}. **{r['test_case'].get('id', 'unknown')}** - \"{r['test_case']['question']}\"\n"
            f"   - Retrieval: {r['retrieval']['avg_relevance']:.1f}/5.0, "
            f"Response: {r['response']['avg_score']:.1f}/5.0, "
            f"Overall: {score:.1f}/5.0\n"
            f"   - *Issue:* {self._identify_issue(r)}"
            for i, (r, score) in enumerate(worst_5, 1)
        ])

        return f"""## Performance Highlights

### Top 5 Best Performing Cases

{top_section}

### Top 5 Worst Performing Cases

{worst_section}

---"""

    def _create_detailed_breakdown(self, results: List[Dict[str, Any]]) -> str:
        """Create detailed per-question breakdown (first 10 cases)."""
        if not results:
            return ""

        breakdowns = []

        for result in results[:10]:  # Show first 10 detailed cases
            test_case = result['test_case']
            test_id = test_case.get('id', 'unknown')
            question = test_case['question']
            category = test_case.get('category', 'unknown')

            # Retrieval details
            chunk_evals = result['retrieval']['chunk_evaluations']
            chunk_table = "\n".join([
                f"| {e['rank']} | {e['metadata'].get('source', 'Unknown')[:40]} | {e['score']:.1f} | {e['explanation'][:60]}... |"
                for e in chunk_evals
            ])

            # Response details
            scores = result['response']['scores']
            explanations = result['response']['explanations']

            score_table = f"""| Accuracy | {scores['accuracy']:.1f} | {explanations['accuracy']} |
| Completeness | {scores['completeness']:.1f} | {explanations['completeness']} |
| Faithfulness | {scores['faithfulness']:.1f} | {explanations['faithfulness']} |
| Relevance | {scores['relevance']:.1f} | {explanations['relevance']} |"""

            hallucination_status = "✓ No hallucinations" if not result['hallucinations']['has_hallucinations'] else "⚠ Hallucinations detected"

            # Performance metrics (if available)
            perf_section = ""
            if 'performance' in result:
                perf = result['performance']
                perf_section = f"""
**Performance:**
- Retrieval Time: {perf['retrieval_time_ms']:.0f}ms
- Response Generation: {perf['response_time_ms']:.0f}ms
- Total Time: {perf['total_time_ms']:.0f}ms
"""

            breakdown = f"""### Test Case: {test_id}

**Question:** {question}
**Category:** {category.replace('_', ' ').title()}
**Author Filter:** {test_case.get('author_filter', 'None')}

#### Retrieval Results

| Rank | Source | Relevance | Judge Comments |
|------|--------|-----------|----------------|
{chunk_table}

**Retrieval Metrics:**
- Average Relevance: {result['retrieval']['avg_relevance']:.2f}/5.0
- Precision@3: {result['retrieval']['precision_at_k']:.2f}
- MRR: {result['retrieval']['mrr']:.2f}
{perf_section}
#### Response Evaluation

**Generated Response:**
> {result['generated_response'][:500]}{"..." if len(result['generated_response']) > 500 else ""}

**Quality Scores:**

| Dimension | Score | Explanation |
|-----------|-------|-------------|
{score_table}

**Hallucinations:** {hallucination_status}

---
"""
            breakdowns.append(breakdown)

        return f"""## Detailed Test Results

{chr(10).join(breakdowns)}"""

    def _create_recommendations(self, results: List[Dict[str, Any]]) -> str:
        """Create recommendations section based on results."""
        if not results:
            return ""

        recommendations = []

        # Analyze retrieval performance
        avg_relevance = sum(r['retrieval']['avg_relevance'] for r in results) / len(results)
        if avg_relevance < 3.5:
            recommendations.append(
                "**Improve Retrieval Quality:** Average relevance score is below 3.5. "
                "Consider increasing chunk overlap, adjusting chunk size, or enabling contextual embeddings."
            )

        # Analyze precision
        avg_precision = sum(r['retrieval']['precision_at_k'] for r in results) / len(results)
        if avg_precision < 0.70:
            recommendations.append(
                "**Increase Precision:** Less than 70% of retrieved chunks are relevant. "
                "Consider fine-tuning embedding model or improving metadata tagging."
            )

        # Analyze response quality
        avg_faithfulness = sum(r['response']['scores']['faithfulness'] for r in results) / len(results)
        if avg_faithfulness < 4.0:
            recommendations.append(
                "**Reduce Hallucinations:** Faithfulness score is below 4.0. "
                "Consider adjusting prompts to emphasize grounding in context, "
                "or using a more capable LLM model."
            )

        # Analyze completeness
        avg_completeness = sum(r['response']['scores']['completeness'] for r in results) / len(results)
        if avg_completeness < 3.5:
            recommendations.append(
                "**Improve Completeness:** Responses are missing important aspects. "
                "Consider retrieving more chunks (increase n_results) or using larger context windows."
            )

        # Identify category weaknesses
        by_category = defaultdict(list)
        for result in results:
            category = result['test_case'].get('category', 'unknown')
            by_category[category].append(result['retrieval']['avg_relevance'])

        for category, scores in by_category.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 3.0:
                recommendations.append(
                    f"**Improve {category.replace('_', ' ').title()} Performance:** "
                    f"This category scores {avg_score:.2f}/5.0, indicating specific challenges "
                    f"with {category.replace('_', ' ')} queries."
                )

        # Strengths
        strengths = []
        if avg_faithfulness >= 4.5:
            strengths.append("Excellent faithfulness - very low hallucination rate")
        if avg_precision >= 0.85:
            strengths.append("High precision in retrieval - most chunks are relevant")
        if avg_relevance >= 4.0:
            strengths.append("Strong retrieval quality - chunks are highly relevant")

        recommendations_text = "\n".join([f"{i}. {rec}" for i, rec in enumerate(recommendations, 1)])
        strengths_text = "\n".join([f"- {s}" for s in strengths])

        return f"""## Recommendations

### Areas for Improvement

{recommendations_text if recommendations else "*No major issues detected - system performing well!*"}

### Strengths to Maintain

{strengths_text if strengths else "*Continue monitoring performance across all metrics.*"}

---

## Configuration Details

- **Chunk Size:** 200 words (from RAG system defaults)
- **Chunk Overlap:** 50 words
- **Embedding Model:** all-MiniLM-L6-v2
- **Top-K Results:** 3
- **Contextual Embeddings:** Enabled

*Report generated by RAG Evaluation System*
"""

    def _create_error_section(self, error_results: List[Dict[str, Any]]) -> str:
        """Create section for errors encountered during evaluation."""
        error_list = "\n".join([
            f"- **{r['test_case'].get('id', 'unknown')}:** {r['error']}"
            for r in error_results
        ])

        return f"""## Errors Encountered

The following test cases encountered errors during evaluation:

{error_list}

---"""

    def _identify_key_findings(self, results: List[Dict[str, Any]]) -> str:
        """Identify key findings from results."""
        findings = []

        # Retrieval findings
        avg_relevance = sum(r['retrieval']['avg_relevance'] for r in results) / len(results)
        if avg_relevance >= 4.0:
            findings.append("✓ **Strength:** Excellent retrieval performance with high relevance scores")
        elif avg_relevance < 3.0:
            findings.append("⚠ **Weakness:** Retrieval quality needs improvement")

        # Response findings
        avg_faithfulness = sum(r['response']['scores']['faithfulness'] for r in results) / len(results)
        if avg_faithfulness >= 4.5:
            findings.append("✓ **Strength:** Very low hallucination rate - responses well-grounded")
        elif avg_faithfulness < 3.5:
            findings.append("⚠ **Weakness:** Hallucination rate is concerning")

        # Category analysis
        by_category = defaultdict(list)
        for result in results:
            category = result['test_case'].get('category', 'unknown')
            overall_score = (result['retrieval']['avg_relevance'] + result['response']['avg_score']) / 2
            by_category[category].append(overall_score)

        best_category = max(by_category.items(), key=lambda x: sum(x[1])/len(x[1]))
        worst_category = min(by_category.items(), key=lambda x: sum(x[1])/len(x[1]))

        findings.append(f"→ **Best Performance:** {best_category[0].replace('_', ' ').title()} queries")
        findings.append(f"→ **Needs Work:** {worst_category[0].replace('_', ' ').title()} queries")

        return "\n".join(findings)

    def _identify_issue(self, result: Dict[str, Any]) -> str:
        """Identify the main issue with a poorly performing case."""
        retrieval_score = result['retrieval']['avg_relevance']
        response_scores = result['response']['scores']

        if retrieval_score < 2.0:
            return "Poor retrieval - chunks not relevant"
        elif response_scores['faithfulness'] < 3.0:
            return "Hallucinations detected"
        elif response_scores['completeness'] < 3.0:
            return "Incomplete response"
        elif response_scores['accuracy'] < 3.0:
            return "Inaccurate response"
        else:
            return "Overall low quality"
