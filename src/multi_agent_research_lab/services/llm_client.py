"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


# Pricing per 1M tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost based on token counts."""
    pricing = MODEL_PRICING.get(model, {"input": 0.15, "output": 0.60})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


class LLMClient:
    """Provider-agnostic LLM client with OpenAI integration and retry logic."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.openai_model
        self.temperature = temperature
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.settings.openai_api_key:
                raise AgentExecutionError("OPENAI_API_KEY is not configured.")
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=float(self.settings.timeout_seconds),
            )
        return self._client

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> Any:
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            **kwargs,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Return a model completion with token and cost tracking."""
        temp = self.temperature if temperature is None else temperature
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self._call_api(messages=messages, temperature=temp)
            choice = response.choices[0]
            content = choice.message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost = estimate_cost(self.model, input_tokens, output_tokens)

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
        except Exception as exc:
            logger.error("LLM completion failed: %s", exc)
            raise AgentExecutionError(f"LLM call failed: {exc}") from exc
