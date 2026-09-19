"""Two-stage EdgeGuard pipeline.

Responsibility: orchestrate guardrail → LLM in sequence.
Knows nothing about HTTP, serial ports, or model internals.

Stage 1: Guardrail classifier screens the prompt.
Stage 2: LLM generates a response only if Stage 1 passes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from edgeguard.guardrail.base import BaseClassifier, ClassifierResult, Verdict
from edgeguard.llm.base import BaseLLM, InferenceResult

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    prompt: str
    verdict: Verdict
    confidence: float
    matched_pattern: str | None
    llm_response: str | None  # None if blocked
    guardrail_latency_ms: float
    llm_latency_ms: float | None  # None if blocked
    total_latency_ms: float


class EdgeGuardPipeline:
    """Routes prompts through the guardrail then the LLM.

    Constructed once at startup via dependency injection.
    Stateless per-call — safe to call from multiple threads.
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        llm: BaseLLM,
        guardrail_enabled: bool = True,
    ) -> None:
        self._classifier = classifier
        self._llm = llm
        self._guardrail_enabled = guardrail_enabled

    def run(self, prompt: str) -> PipelineResult:
        t0 = time.perf_counter()

        # ── Stage 1: Guardrail ───────────────────────────────────────────────
        if self._guardrail_enabled:
            clf_result: ClassifierResult = self._classifier.classify(prompt)
        else:
            from edgeguard.guardrail.base import ClassifierResult

            clf_result = ClassifierResult(
                verdict=Verdict.CLEAN, confidence=0.0, matched_pattern="guardrail_off"
            )

        guardrail_ms = (time.perf_counter() - t0) * 1000

        log.info(
            "pipeline.guardrail",
            verdict=clf_result.verdict,
            confidence=round(clf_result.confidence, 3),
            pattern=clf_result.matched_pattern,
            latency_ms=round(guardrail_ms, 2),
            classifier=self._classifier.name,
        )

        if clf_result.verdict == Verdict.INJECTED:
            return PipelineResult(
                prompt=prompt,
                verdict=Verdict.INJECTED,
                confidence=clf_result.confidence,
                matched_pattern=clf_result.matched_pattern,
                llm_response=None,
                guardrail_latency_ms=guardrail_ms,
                llm_latency_ms=None,
                total_latency_ms=guardrail_ms,
            )

        # ── Stage 2: LLM inference ───────────────────────────────────────────
        t1 = time.perf_counter()
        llm_result: InferenceResult = self._llm.generate(prompt)
        llm_ms = (time.perf_counter() - t1) * 1000
        total_ms = (time.perf_counter() - t0) * 1000

        log.info(
            "pipeline.llm",
            tokens=llm_result.completion_tokens,
            latency_ms=round(llm_ms, 2),
        )

        return PipelineResult(
            prompt=prompt,
            verdict=clf_result.verdict,
            confidence=clf_result.confidence,
            matched_pattern=clf_result.matched_pattern,
            llm_response=llm_result.text,
            guardrail_latency_ms=guardrail_ms,
            llm_latency_ms=llm_ms,
            total_latency_ms=total_ms,
        )
