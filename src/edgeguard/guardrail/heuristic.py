"""Heuristic (rule-based) guardrail classifier.

Used in Weeks 1–3 before the distilled ML model is trained.
Fast, zero-dependency, and provides a clear baseline to beat.
Each rule is a single pattern with an associated confidence score.
"""

import re

from edgeguard.guardrail.base import BaseClassifier, ClassifierResult, Verdict

# Ordered by severity — first match wins
_INJECTION_PATTERNS: list[tuple[str, float, re.Pattern[str]]] = [
    (
        "ignore_instructions",
        0.97,
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,40}"
            r"\b(previous|prior|above|all|system)\b.{0,20}"
            r"\b(instructions?|prompts?|context|rules?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "new_instructions",
        0.93,
        re.compile(
            r"\b(new|following|actual|real)\b.{0,20}"
            r"\b(instructions?|task|goal|directive)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_jailbreak",
        0.95,
        re.compile(
            r"\b(pretend|act|behave|you are now|from now on)\b.{0,30}"
            r"\b(evil|unrestricted|jailbroken|dan|no restriction)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_command_injection",
        0.90,
        re.compile(
            r"\b(execute|run|eval|system\(|os\.)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_leak",
        0.88,
        re.compile(
            r"\b(repeat|print|output|reveal|show)\b.{0,20}"
            r"\b(system prompt|instructions?|context above)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "delimiter_injection",
        0.85,
        re.compile(
            r"(\[INST\]|\[\/INST\]|<\|im_start\|>|<\|im_end\|>|###)", re.IGNORECASE
        ),
    ),
]


class HeuristicClassifier(BaseClassifier):
    """Rule-based prompt injection classifier.

    O(n * k) where n = prompt length and k = number of patterns.
    Deterministic and interpretable — every block has a named reason.
    """

    def classify(self, prompt: str) -> ClassifierResult:
        for pattern_name, confidence, pattern in _INJECTION_PATTERNS:
            if pattern.search(prompt):
                return ClassifierResult(
                    verdict=Verdict.INJECTED,
                    confidence=confidence,
                    matched_pattern=pattern_name,
                )
        return ClassifierResult(
            verdict=Verdict.CLEAN,
            confidence=0.05,
            matched_pattern=None,
        )

    @property
    def name(self) -> str:
        return "heuristic"
