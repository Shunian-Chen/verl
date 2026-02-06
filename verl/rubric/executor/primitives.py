# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Scoring primitives for rubric execution.

These primitives are used to score individual verification items
based on collected evidence and policy output.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from verl.rubric.schemas import EvidenceRecord, ScoringPrimitive, VerificationItem

logger = logging.getLogger(__name__)


class BaseScoringPrimitive(ABC):
    """
    Abstract base class for scoring primitives.

    A scoring primitive takes evidence and policy output and returns
    a score in [0, 1].
    """

    @abstractmethod
    def score(
        self,
        verification_item: VerificationItem,
        evidence: list[EvidenceRecord],
        policy_output: str,
    ) -> float:
        """
        Score a verification item.

        Args:
            verification_item: The item to score.
            evidence: Collected evidence for this item.
            policy_output: The policy's response.

        Returns:
            Score in [0, 1].
        """
        pass


class ExactMatchPrimitive(BaseScoringPrimitive):
    """
    Exact match scoring primitive.

    Returns 1.0 if the policy output exactly matches the expected
    value from evidence, 0.0 otherwise.

    Config options:
        - case_sensitive: Whether comparison is case-sensitive (default: True)
        - strip_whitespace: Whether to strip whitespace (default: True)
        - expected_field: Field in evidence to compare against (default: "tool_output")
    """

    def score(
        self,
        verification_item: VerificationItem,
        evidence: list[EvidenceRecord],
        policy_output: str,
    ) -> float:
        config = verification_item.scoring_config
        case_sensitive = config.get("case_sensitive", True)
        strip_whitespace = config.get("strip_whitespace", True)
        expected_field = config.get("expected_field", "tool_output")

        # Get expected value from evidence
        expected = None
        for e in evidence:
            if e.check_id == verification_item.id and e.success:
                expected = getattr(e, expected_field, e.tool_output)
                break

        if expected is None:
            return 0.0

        # Prepare strings for comparison
        actual = str(policy_output)
        expected = str(expected)

        if strip_whitespace:
            actual = actual.strip()
            expected = expected.strip()

        if not case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        return 1.0 if actual == expected else 0.0


class ContainsPrimitive(BaseScoringPrimitive):
    """
    Contains scoring primitive.

    Returns 1.0 if the policy output contains the expected substring,
    0.0 otherwise.

    Config options:
        - case_sensitive: Whether search is case-sensitive (default: False)
        - expected_value: Static expected value to search for
        - expected_field: Field in evidence containing expected value
        - all_must_match: If True, all evidence values must be found (default: False)
    """

    def score(
        self,
        verification_item: VerificationItem,
        evidence: list[EvidenceRecord],
        policy_output: str,
    ) -> float:
        config = verification_item.scoring_config
        case_sensitive = config.get("case_sensitive", False)
        expected_value = config.get("expected_value")
        expected_field = config.get("expected_field", "tool_output")
        all_must_match = config.get("all_must_match", False)

        # Prepare policy output for search
        search_text = policy_output if case_sensitive else policy_output.lower()

        # Collect expected values
        expected_values = []
        if expected_value:
            expected_values.append(expected_value)

        for e in evidence:
            if e.check_id == verification_item.id and e.success:
                value = getattr(e, expected_field, e.tool_output)
                if value:
                    expected_values.append(str(value))

        if not expected_values:
            return 0.0

        # Check for matches
        matches = 0
        for expected in expected_values:
            check_value = expected if case_sensitive else expected.lower()
            if check_value in search_text:
                matches += 1

        if all_must_match:
            return 1.0 if matches == len(expected_values) else 0.0
        else:
            return 1.0 if matches > 0 else 0.0


class RegexPrimitive(BaseScoringPrimitive):
    """
    Regex scoring primitive.

    Returns 1.0 if the policy output matches the regex pattern,
    0.0 otherwise.

    Config options:
        - pattern: Regex pattern to match
        - flags: Regex flags (e.g., "IGNORECASE", "MULTILINE")
        - partial_match: If True, use search instead of fullmatch (default: True)
    """

    def score(
        self,
        verification_item: VerificationItem,
        evidence: list[EvidenceRecord],
        policy_output: str,
    ) -> float:
        config = verification_item.scoring_config
        pattern = config.get("pattern")

        if not pattern:
            return 0.0

        # Parse flags
        flags_str = config.get("flags", "")
        flags = 0
        if "IGNORECASE" in flags_str or "I" in flags_str:
            flags |= re.IGNORECASE
        if "MULTILINE" in flags_str or "M" in flags_str:
            flags |= re.MULTILINE
        if "DOTALL" in flags_str or "S" in flags_str:
            flags |= re.DOTALL

        partial_match = config.get("partial_match", True)

        try:
            regex = re.compile(pattern, flags)
            if partial_match:
                match = regex.search(policy_output)
            else:
                match = regex.fullmatch(policy_output)

            return 1.0 if match else 0.0
        except re.error:
            return 0.0


class CodeExecutionPrimitive(BaseScoringPrimitive):
    """
    Code execution scoring primitive.

    Scores based on the result of executing the policy's code output
    against test cases.

    Config options:
        - success_field: Field in evidence indicating success (default: "success")
        - partial_credit: If True, give partial credit for partial success (default: True)
    """

    def score(
        self,
        verification_item: VerificationItem,
        evidence: list[EvidenceRecord],
        policy_output: str,
    ) -> float:
        config = verification_item.scoring_config
        success_field = config.get("success_field", "success")
        partial_credit = config.get("partial_credit", True)

        # Filter evidence for this check
        relevant_evidence = [
            e for e in evidence if e.check_id == verification_item.id
        ]

        if not relevant_evidence:
            return 0.0

        # Count successes
        successes = 0
        for e in relevant_evidence:
            # Check success from evidence record
            if e.success:
                # Also check tool output if it has a success field
                if isinstance(e.tool_output, dict):
                    tool_success = e.tool_output.get(success_field, True)
                    if tool_success:
                        successes += 1
                else:
                    successes += 1

        if partial_credit:
            return successes / len(relevant_evidence)
        else:
            return 1.0 if successes == len(relevant_evidence) else 0.0


class NumericComparisonPrimitive(BaseScoringPrimitive):
    """
    Numeric comparison scoring primitive.

    Compares numeric values from policy output with expected values.

    Config options:
        - comparison: Comparison type ("eq", "lt", "le", "gt", "ge", "approx")
        - tolerance: Tolerance for approximate comparison (default: 0.01)
        - expected_value: Static expected value
        - extract_pattern: Regex to extract number from policy output
    """

    def score(
        self,
        verification_item: VerificationItem,
        evidence: list[EvidenceRecord],
        policy_output: str,
    ) -> float:
        config = verification_item.scoring_config
        comparison = config.get("comparison", "eq")
        tolerance = config.get("tolerance", 0.01)
        expected_value = config.get("expected_value")
        extract_pattern = config.get("extract_pattern", r"[-+]?\d*\.?\d+")

        # Extract numeric value from policy output
        match = re.search(extract_pattern, policy_output)
        if not match:
            return 0.0

        try:
            actual = float(match.group())
        except ValueError:
            return 0.0

        # Get expected value from config or evidence
        if expected_value is None:
            for e in evidence:
                if e.check_id == verification_item.id and e.success:
                    if isinstance(e.tool_output, (int, float)):
                        expected_value = e.tool_output
                    elif isinstance(e.tool_output, dict):
                        expected_value = e.tool_output.get("expected")
                    break

        if expected_value is None:
            return 0.0

        try:
            expected = float(expected_value)
        except (ValueError, TypeError):
            return 0.0

        # Perform comparison
        if comparison == "eq":
            return 1.0 if actual == expected else 0.0
        elif comparison == "lt":
            return 1.0 if actual < expected else 0.0
        elif comparison == "le":
            return 1.0 if actual <= expected else 0.0
        elif comparison == "gt":
            return 1.0 if actual > expected else 0.0
        elif comparison == "ge":
            return 1.0 if actual >= expected else 0.0
        elif comparison == "approx":
            return 1.0 if abs(actual - expected) <= tolerance else 0.0
        else:
            return 0.0


class SemanticSimilarityPrimitive(BaseScoringPrimitive):
    """
    Semantic similarity scoring primitive.

    Computes semantic similarity between policy output and expected value.
    Requires an embedding model or similarity function.

    Config options:
        - threshold: Minimum similarity for score of 1.0 (default: 0.8)
        - expected_value: Static expected value to compare against
    """

    def score(
        self,
        verification_item: VerificationItem,
        evidence: list[EvidenceRecord],
        policy_output: str,
    ) -> float:
        # This is a placeholder - actual implementation would use
        # an embedding model or similarity function
        config = verification_item.scoring_config
        expected_value = config.get("expected_value", "")

        # Simple word overlap as fallback
        if not expected_value:
            for e in evidence:
                if e.check_id == verification_item.id and e.success:
                    expected_value = str(e.tool_output)
                    break

        if not expected_value:
            return 0.0

        # Simple Jaccard similarity as placeholder
        actual_words = set(policy_output.lower().split())
        expected_words = set(expected_value.lower().split())

        if not actual_words or not expected_words:
            return 0.0

        intersection = actual_words & expected_words
        union = actual_words | expected_words

        similarity = len(intersection) / len(union)
        threshold = config.get("threshold", 0.8)

        return 1.0 if similarity >= threshold else similarity / threshold


class LLMJudgePrimitive(BaseScoringPrimitive):
    """
    LLM-based judge scoring primitive.

    Uses an LLM to evaluate the policy output against specified criteria.
    Calls an OpenAI-compatible API to score the output in [0, 1].

    Config options:
        - criteria: Evaluation criteria for the LLM judge
        - rubric: Detailed scoring rubric text
        - model: Model name to use (default: env ``LLM_JUDGE_MODEL`` or "gpt-4o")
        - api_key: API key (default: env ``OPENAI_API_KEY``)
        - api_base: Custom API base URL (default: env ``OPENAI_API_BASE``)
    """

    _LLM_JUDGE_PROMPT = (
        "You are an expert evaluator. Score the following response on a scale "
        "from 0.0 to 1.0 based on the criteria below.\n\n"
        "## Criteria\n{criteria}\n\n"
        "{rubric_section}"
        "## Evidence\n{evidence_summary}\n\n"
        "## Policy Output\n{policy_output}\n\n"
        "Respond with ONLY a JSON object: {{\"score\": <float between 0.0 and 1.0>}}"
    )

    def __init__(self) -> None:
        super().__init__()
        self._client = None

    def _get_client(self):
        """Lazy-initialize the OpenAI async client."""
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package not installed; LLMJudgePrimitive will return fallback score")
            return None

        kwargs: dict[str, Any] = {}
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE")
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["base_url"] = api_base

        try:
            self._client = OpenAI(**kwargs)
        except Exception as exc:
            logger.warning(f"Failed to create OpenAI client: {exc}")
            return None
        return self._client

    def score(
        self,
        verification_item: VerificationItem,
        evidence: list[EvidenceRecord],
        policy_output: str,
    ) -> float:
        config = verification_item.scoring_config
        criteria = config.get("criteria", "Overall quality and correctness.")
        rubric_text = config.get("rubric", "")
        model = config.get("model", os.environ.get("LLM_JUDGE_MODEL", "gpt-4o"))

        # Build evidence summary
        evidence_lines: list[str] = []
        for e in evidence:
            if e.check_id == verification_item.id:
                status = "success" if e.success else "failure"
                evidence_lines.append(f"- [{status}] {e.tool_name}: {e.tool_output}")
        evidence_summary = "\n".join(evidence_lines) if evidence_lines else "(no evidence collected)"

        rubric_section = f"## Rubric\n{rubric_text}\n\n" if rubric_text else ""

        prompt = self._LLM_JUDGE_PROMPT.format(
            criteria=criteria,
            rubric_section=rubric_section,
            evidence_summary=evidence_summary,
            policy_output=policy_output[:4000],  # Truncate to stay within limits
        )

        return self._call_llm_judge(prompt, model)

    def _call_llm_judge(self, prompt: str, model: str) -> float:
        """Call the LLM judge and parse the numeric score."""
        client = self._get_client()
        if client is None:
            return 0.5  # Fallback when API is unavailable

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            content = response.choices[0].message.content or ""
            return self._parse_score(content)
        except Exception as exc:
            logger.warning(f"LLM judge call failed: {exc}")
            return 0.5  # Fallback on error

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract a float score from the LLM response."""
        import json as _json

        # Try JSON parse first
        try:
            data = _json.loads(text)
            if isinstance(data, dict) and "score" in data:
                return max(0.0, min(1.0, float(data["score"])))
        except (_json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: find a float in the text
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            value = float(match.group(1))
            return max(0.0, min(1.0, value))

        return 0.5


class PerformanceRatioPrimitive(BaseScoringPrimitive):
    """
    Compare model code vs reference code on a single runtime metric.

    Evidence for the same check_id is split by tool_input["role"]:
      - "model": metric value from the model's code
      - "reference": metric value from the reference solution

    scoring_config:
        metric_field: str — field name in tool_output (e.g. "execution_time_ms")
        direction: "lower_is_better" (default) | "higher_is_better"

    Score formula (lower_is_better):
        score = min(1.0, reference_value / model_value)
    Score formula (higher_is_better):
        score = min(1.0, model_value / reference_value)
    """

    def score(
        self,
        verification_item: VerificationItem,
        evidence: list[EvidenceRecord],
        policy_output: str,
    ) -> float:
        config = verification_item.scoring_config
        metric_field = config.get("metric_field", "execution_time_ms")
        direction = config.get("direction", "lower_is_better")

        model_value = None
        reference_value = None

        for e in evidence:
            if e.check_id != verification_item.id or not e.success:
                continue
            role = (e.tool_input or {}).get("role", "")
            if isinstance(e.tool_output, dict):
                metric = e.tool_output.get(metric_field)
            else:
                continue
            if metric is None:
                continue
            try:
                metric = float(metric)
            except (ValueError, TypeError):
                continue

            if role == "model":
                model_value = metric
            elif role == "reference":
                reference_value = metric

        if model_value is None or reference_value is None:
            return 0.0

        # Avoid division by zero
        if direction == "lower_is_better":
            if model_value <= 0:
                return 1.0 if reference_value >= 0 else 0.0
            return min(1.0, reference_value / model_value)
        else:  # higher_is_better
            if reference_value <= 0:
                return 1.0 if model_value >= 0 else 0.0
            return min(1.0, model_value / reference_value)


# Registry of scoring primitives
_SCORING_PRIMITIVES: dict[ScoringPrimitive, type[BaseScoringPrimitive]] = {
    ScoringPrimitive.EXACT_MATCH: ExactMatchPrimitive,
    ScoringPrimitive.CONTAINS: ContainsPrimitive,
    ScoringPrimitive.REGEX: RegexPrimitive,
    ScoringPrimitive.CODE_EXECUTION: CodeExecutionPrimitive,
    ScoringPrimitive.NUMERIC_COMPARISON: NumericComparisonPrimitive,
    ScoringPrimitive.SEMANTIC_SIMILARITY: SemanticSimilarityPrimitive,
    ScoringPrimitive.LLM_JUDGE: LLMJudgePrimitive,
    ScoringPrimitive.PERFORMANCE_RATIO: PerformanceRatioPrimitive,
}


def get_scoring_primitive(primitive_type: ScoringPrimitive) -> BaseScoringPrimitive:
    """
    Get a scoring primitive instance by type.

    Args:
        primitive_type: The type of scoring primitive.

    Returns:
        An instance of the scoring primitive.

    Raises:
        ValueError: If the primitive type is not supported.
    """
    if primitive_type not in _SCORING_PRIMITIVES:
        raise ValueError(f"Unknown scoring primitive: {primitive_type}")

    return _SCORING_PRIMITIVES[primitive_type]()
