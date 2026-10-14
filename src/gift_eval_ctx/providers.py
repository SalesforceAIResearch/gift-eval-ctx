# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2
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

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Literal, Protocol

ProviderName = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class RequestSettings:
    """Provider-neutral settings; unsupported model options fail at the API."""

    provider: ProviderName
    model: str
    max_tokens: int = 6000
    timeout: float = 300.0
    reasoning_effort: str | None = None
    temperature: float | None = None
    thinking: Literal["adaptive", "enabled", "disabled"] | None = None
    thinking_budget: int | None = None

    def __post_init__(self) -> None:
        if self.provider not in {"openai", "anthropic"} or not self.model.strip():
            raise ValueError("An official provider and a nonempty model are required")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if self.temperature is not None and not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        if self.provider == "openai" and (
            self.thinking is not None or self.thinking_budget is not None
        ):
            raise ValueError("thinking and thinking_budget are Anthropic-only settings")
        if self.thinking == "enabled":
            if (
                type(self.thinking_budget) is not int
                or not 1024 <= self.thinking_budget < self.max_tokens
            ):
                raise ValueError("Enabled thinking requires 1024 <= thinking_budget < max_tokens")
        elif self.thinking_budget is not None:
            raise ValueError("thinking_budget requires thinking=enabled")
        if self.thinking in {"enabled", "adaptive"} and self.temperature is not None:
            raise ValueError("Omit temperature when enabling Anthropic thinking")
        if self.thinking not in {None, "adaptive", "enabled", "disabled"}:
            raise ValueError("Unknown Anthropic thinking mode")


@dataclass(frozen=True)
class APIResponse:
    """Text-only model output and audit metadata, excluding reasoning content."""

    text: str
    response_id: str | None
    model: str | None
    finish_reason: str | None
    usage: dict[str, Any]


class Provider(Protocol):
    """Interface used by the runner and offline test doubles."""

    async def generate(self, system: str, user: str) -> APIResponse: ...

    async def close(self) -> None: ...


class OpenAIProvider:
    """Use Responses with explicit output limits and SDK retries disabled."""

    def __init__(self, settings: RequestSettings) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ValueError("Install the openai extra: pip install '.[openai]'") from exc
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY before running inference")
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=key,
            base_url="https://api.openai.com/v1",
            timeout=settings.timeout,
            max_retries=0,
        )

    async def generate(self, system: str, user: str) -> APIResponse:
        params: dict[str, Any] = {
            "model": self.settings.model,
            "instructions": system,
            "input": [{"role": "user", "content": user}],
            "max_output_tokens": self.settings.max_tokens,
            "store": False,
        }
        if self.settings.reasoning_effort is not None:
            params["reasoning"] = {"effort": self.settings.reasoning_effort}
        if self.settings.temperature is not None:
            params["temperature"] = self.settings.temperature
        response = await self.client.responses.create(**params)
        reason = response.status
        if response.incomplete_details is not None:
            reason = response.incomplete_details.reason
        return APIResponse(
            response.output_text,
            response.id,
            response.model,
            reason,
            response.usage.model_dump(mode="json") if response.usage is not None else {},
        )

    async def close(self) -> None:
        await self.client.close()


class AnthropicProvider:
    """Stream Messages to support long generations; parse only final text blocks."""

    def __init__(self, settings: RequestSettings) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ValueError("Install the anthropic extra: pip install '.[anthropic]'") from exc
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY before running inference")
        self.settings = settings
        self.client = AsyncAnthropic(
            api_key=key,
            base_url="https://api.anthropic.com",
            timeout=settings.timeout,
            max_retries=0,
        )

    async def generate(self, system: str, user: str) -> APIResponse:
        params: dict[str, Any] = {
            "model": self.settings.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": self.settings.max_tokens,
        }
        if self.settings.reasoning_effort is not None:
            params["output_config"] = {"effort": self.settings.reasoning_effort}
        if self.settings.thinking is not None:
            params["thinking"] = {"type": self.settings.thinking}
            if self.settings.thinking_budget is not None:
                params["thinking"]["budget_tokens"] = self.settings.thinking_budget
        if self.settings.temperature is not None:
            params["temperature"] = self.settings.temperature
        async with self.client.messages.stream(**params) as stream:
            message = await stream.get_final_message()
        text = "".join(block.text for block in message.content if block.type == "text")
        return APIResponse(
            text,
            message.id,
            message.model,
            message.stop_reason,
            message.usage.model_dump(mode="json"),
        )

    async def close(self) -> None:
        await self.client.close()


def create_provider(settings: RequestSettings) -> Provider:
    """Construct only the chosen SDK client, with credentials from its standard environment variable."""
    return (
        OpenAIProvider(settings) if settings.provider == "openai" else AnthropicProvider(settings)
    )


def is_retryable(exc: Exception) -> bool:
    """Retry transient SDK errors only; authentication and invalid requests are fatal."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 400:
        return status in {408, 409, 429} or status >= 500
    # Anthropic can report transient SSE errors after HTTP 200 has begun.
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("type") in {
            "overloaded_error",
            "api_error",
            "rate_limit_error",
            "timeout_error",
        }:
            return True
    return isinstance(exc, TimeoutError) or type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ReadError",
        "WriteError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "RemoteProtocolError",
    }


def retry_delay(exc: Exception, attempt: int) -> float:
    """Respect short Retry-After hints, with bounded exponential backoff otherwise."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    try:
        delay = float(headers.get("retry-after", "nan"))
        if 0 <= delay <= 60:
            return delay
    except (ValueError, TypeError):
        pass
    return min(2 ** min(attempt - 1, 5) + random.uniform(0, 1), 30.0)
