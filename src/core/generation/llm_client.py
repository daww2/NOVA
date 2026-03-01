"""
LLM Client for RAG Pipeline.

Simple OpenAI client with retries and streaming support.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from src.core.observability.tracing import observe, _langfuse_available
from src.core.observability.metrics import METRICS

logger = logging.getLogger(__name__)

_langfuse = None
if _langfuse_available:
    from langfuse import get_client
    _langfuse = get_client()

# ---------------------------------------------------------------------------
# Cost lookup (USD per 1K tokens) — extend as needed
# ---------------------------------------------------------------------------
_COST_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    costs = _COST_PER_1K.get(model, _COST_PER_1K.get("gpt-4o-mini"))
    if not costs:
        return 0.0
    return (prompt_tokens / 1000) * costs["prompt"] + (completion_tokens / 1000) * costs["completion"]


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: dict  # {prompt_tokens, completion_tokens, total_tokens}
    latency_ms: float


class LLMClient:
    """
    Simple OpenAI LLM client.

    Usage:
        client = LLMClient(model="gpt-4o-mini")
        response = await client.generate(prompt, system_prompt)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Generate response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with content, model, usage, latency
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                latency = time.perf_counter() - start
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                content = response.choices[0].message.content

                # Prometheus metrics
                METRICS.LLM_DURATION.observe(latency)
                METRICS.LLM_TOKENS.labels(type="prompt").inc(prompt_tokens)
                METRICS.LLM_TOKENS.labels(type="completion").inc(completion_tokens)
                cost = _estimate_cost(self.model, prompt_tokens, completion_tokens)
                if cost > 0:
                    METRICS.LLM_COST.inc(cost)

                # Langfuse: report generation details
                if _langfuse:
                    try:
                        with _langfuse.start_as_current_observation(
                            as_type="generation",
                            name="llm_generate",
                            model=self.model,
                            input=messages,
                        ) as generation:
                            generation.update(
                                output=content,
                                usage={
                                    "input": prompt_tokens,
                                    "output": completion_tokens,
                                    "total": response.usage.total_tokens,
                                    "unit": "TOKENS",
                                },
                                metadata={"temperature": temperature, "max_tokens": max_tokens},
                            )
                    except Exception as e:
                        logger.debug("Langfuse generation update failed: %s", e)

                return LLMResponse(
                    content=content,
                    model=self.model,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    latency_ms=latency * 1000,
                )

            except Exception as e:
                last_error = e
                METRICS.ERRORS_TOTAL.labels(component="llm").inc()
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")

    @observe(as_type="generation", capture_input=False)
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the LLM one chunk at a time.

        Yields:
            Content delta strings as they arrive.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        collected = []

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        usage_data = None
        async for chunk in stream:
            # Capture usage from the final chunk (OpenAI sends it when include_usage=True)
            if chunk.usage:
                usage_data = chunk.usage

            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                collected.append(delta.content)
                yield delta.content

        # Record metrics after stream completes
        latency = time.perf_counter() - start
        METRICS.LLM_DURATION.observe(latency)

        full_output = "".join(collected)

        if usage_data:
            prompt_tokens = usage_data.prompt_tokens or 0
            completion_tokens = usage_data.completion_tokens or 0
            total_tokens = usage_data.total_tokens or 0

            METRICS.LLM_TOKENS.labels(type="prompt").inc(prompt_tokens)
            METRICS.LLM_TOKENS.labels(type="completion").inc(completion_tokens)
            cost = _estimate_cost(self.model, prompt_tokens, completion_tokens)
            if cost > 0:
                METRICS.LLM_COST.inc(cost)

            # Langfuse: report generation details
            if _langfuse:
                try:
                    with _langfuse.start_as_current_observation(
                        as_type="generation",
                        name="llm_generate_stream",
                        model=self.model,
                        input=messages,
                    ) as generation:
                        generation.update(
                            output=full_output,
                            usage={
                                "input": prompt_tokens,
                                "output": completion_tokens,
                                "total": total_tokens,
                                "unit": "TOKENS",
                            },
                            metadata={"temperature": temperature, "max_tokens": max_tokens},
                        )
                except Exception as e:
                    logger.debug("Langfuse generation update failed: %s", e)

    async def close(self):
        """Close client."""
        await self._client.close()


def create_llm_client(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> LLMClient:
    """Create LLM client. Works with any OpenAI-compatible API."""
    return LLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )
