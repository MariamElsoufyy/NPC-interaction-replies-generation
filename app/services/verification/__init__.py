"""Content-verification stack for LLM I/O.

Layers, ordered fastest → slowest (= cheapest → most expensive):

    1. regex_checks   — sub-millisecond, local (profanity, anachronism, URLs, emails)
    2. model_checks   — 200ms–3s, OpenAI API (moderation + LLM judge)

Use the :class:`Verifier` orchestrator for a pipeline-friendly API that
runs Tier 1 synchronously before TTS and gathers Tier 2 in parallel
with TTS streaming.
"""

from app.services.verification.base import CheckResult, AggregateResult
from app.services.verification.orchestrator import Verifier

__all__ = ["CheckResult", "AggregateResult", "Verifier"]
