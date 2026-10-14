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

import asyncio
import json
import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from filelock import FileLock, Timeout

from gift_eval_ctx.data import (
    KEY_COLUMNS,
    SYSTEM_PROMPTS,
    Mode,
    Sample,
    SampleKey,
    build_user_message,
    load_samples,
)
from gift_eval_ctx.metrics import mase, seasonality_for_frequency
from gift_eval_ctx.parsing import parse_forecast
from gift_eval_ctx.providers import (
    Provider,
    RequestSettings,
    create_provider,
    is_retryable,
    retry_delay,
)
from gift_eval_ctx.storage import (
    atomic_json,
    check_manifest,
    experiment_settings,
    load_results,
    save_results,
)

logger = logging.getLogger(__name__)


class RunError(RuntimeError):
    """A non-retryable provider error; completed work has already been checkpointed."""


@dataclass(frozen=True)
class RunSettings:
    input: Path
    output: Path
    mode: Mode = "hist_effects"
    workers: int = 8
    save_every: int = 50
    retries: int = 2
    limit: int | None = None
    batch: int | None = None

    def __post_init__(self) -> None:
        if self.workers < 1 or self.save_every < 1 or self.retries < 0:
            raise ValueError("workers/save_every must be positive; retries must be nonnegative")
        if any(value is not None and value < 1 for value in (self.limit, self.batch)):
            raise ValueError("limit and batch must be positive")
        if self.input.resolve() == self.output.resolve():
            raise ValueError("Input and output paths must differ")
        if self.output.suffix != ".parquet":
            raise ValueError("Output must end in .parquet")


@dataclass(frozen=True)
class RunSummary:
    planned: int
    skipped: int
    succeeded: int
    failed: int
    elapsed_seconds: float
    diagnostics: str | None


def _write_event(handle: TextIO, event: dict[str, Any]) -> None:
    handle.write(json.dumps(event, allow_nan=False) + "\n")
    handle.flush()


async def _evaluate_sample(
    sample: Sample,
    mode: Mode,
    provider: Provider,
    request: RequestSettings,
    retries: int,
    audit: TextIO,
) -> dict[str, Any] | None:
    user = build_user_message(sample, mode)
    for attempt in range(1, retries + 2):
        started = time.monotonic()
        event: dict[str, Any] = {
            **dict(zip(KEY_COLUMNS, sample.key, strict=True)),
            "attempt": attempt,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        try:
            # Bound the entire generation, including all Anthropic streaming chunks.
            async with asyncio.timeout(request.timeout):
                response = await provider.generate(SYSTEM_PROMPTS[mode], user)
        except Exception as exc:
            retryable = is_retryable(exc)
            retry = retryable and attempt <= retries
            event.update(
                status="api_error",
                error_type=type(exc).__name__,
                http_status=getattr(exc, "status_code", None),
                elapsed_seconds=time.monotonic() - started,
                retry=retry,
            )
            # SDK exceptions may contain headers, prompts, or credential-bearing URLs.
            # Record their class/status rather than serializing the exception object.
            _write_event(audit, event)
            if not retryable:
                status = event["http_status"]
                raise RunError(
                    f"Provider rejected a request: {type(exc).__name__} (HTTP {status}). "
                    "Check the API key, model, and model-specific options."
                ) from None
            if retry:
                await asyncio.sleep(retry_delay(exc, attempt))
                continue
            return None
        event.update(
            raw_response=response.text,
            response_id=response.response_id,
            returned_model=response.model,
            finish_reason=response.finish_reason,
            usage=response.usage,
            elapsed_seconds=time.monotonic() - started,
        )
        try:
            parsed = parse_forecast(response.text, sample.horizon)
            score = mase(
                sample.future,
                parsed.values,
                sample.history,
                seasonality_for_frequency(sample.frequency),
            )
            if not math.isfinite(score):
                raise ValueError("MASE is not finite")
        except ValueError as exc:
            event.update(status="invalid_forecast", error=str(exc))
            _write_event(audit, event)
            return None
        event.update(
            status="ok",
            original_length=parsed.original_length,
            padded=parsed.padded,
            truncated=parsed.truncated,
            recovered_json=parsed.recovered_json,
            seasonality=seasonality_for_frequency(sample.frequency),
            mase=score,
        )
        _write_event(audit, event)
        return {
            **dict(zip(KEY_COLUMNS, sample.key, strict=True)),
            "forecast": parsed.values,
            "mase": score,
        }
    raise AssertionError("Unreachable retry state")


async def run_evaluation(
    settings: RunSettings, request: RequestSettings, provider: Provider | None = None
) -> RunSummary:
    """Evaluate the next pending samples; a supplied provider is owned by the caller.

    ``limit`` restricts the input prefix, while ``batch`` selects the next N
    incomplete samples within that prefix. Existing results outside the prefix
    are preserved. Failures are logged and omitted from the result Parquet.
    """
    samples = load_samples(settings.input, settings.mode)
    if not samples:
        raise ValueError("Input has no samples")
    experiment = experiment_settings(settings.mode, request)
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(settings.output) + ".lock")
    try:
        lock.acquire(timeout=0)
    except Timeout:
        raise ValueError(
            "Another process is writing this output; use a different --output path"
        ) from None
    try:
        check_manifest(settings.output, experiment)
        results = load_results(settings.output, samples)
        selected = samples[: settings.limit] if settings.limit is not None else samples
        pending = [sample for sample in selected if sample.key not in results]
        skipped = len(selected) - len(pending)
        if settings.batch is not None:
            pending = pending[: settings.batch]
        if not pending:
            logger.info("No pending samples (%d valid results skipped)", skipped)
            return RunSummary(0, skipped, 0, 0, 0.0, None)
        return await _run_pending(
            settings, request, provider, samples, pending, results, skipped, experiment
        )
    finally:
        lock.release()


async def _run_pending(
    settings: RunSettings,
    request: RequestSettings,
    provider: Provider | None,
    samples: list[Sample],
    pending: list[Sample],
    results: dict[SampleKey, dict[str, Any]],
    skipped: int,
    experiment: dict[str, Any],
) -> RunSummary:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    prefix = settings.output.parent / "logs" / f"{settings.output.stem}.{run_id}"
    diagnostics = Path(f"{prefix}.jsonl")
    config = {
        **asdict(settings),
        "input": str(settings.input.resolve()),
        "output": str(settings.output.resolve()),
    }
    atomic_json(
        Path(f"{prefix}.config.json"),
        {
            "run": config,
            "request": asdict(request),
            "experiment": experiment,
            "system_prompt": SYSTEM_PROMPTS[settings.mode],
        },
    )
    started = time.monotonic()
    queue: asyncio.Queue[Sample] = asyncio.Queue()
    for sample in pending:
        queue.put_nowait(sample)
    succeeded = failed = completed = 0
    outcome = "interrupted"
    tasks: list[asyncio.Task[None]] = []
    logger.info(
        "Running %d pending samples with %d workers; skipped %d",
        len(pending),
        settings.workers,
        skipped,
    )
    with diagnostics.open("w", encoding="utf-8") as audit:
        client: Provider | None = provider

        async def worker() -> None:
            nonlocal succeeded, failed, completed
            assert client is not None
            while True:
                try:
                    sample = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                row = await _evaluate_sample(
                    sample, settings.mode, client, request, settings.retries, audit
                )
                if row is None:
                    failed += 1
                else:
                    results[sample.key] = row
                    succeeded += 1
                completed += 1
                if completed % settings.save_every == 0:
                    save_results(settings.output, results, samples)
                if completed % min(settings.save_every, 20) == 0 or completed == len(pending):
                    logger.info(
                        "Completed %d/%d: %d succeeded, %d failed",
                        completed,
                        len(pending),
                        succeeded,
                        failed,
                    )

        try:
            if client is None:
                client = create_provider(request)
            tasks = [
                asyncio.create_task(worker()) for _ in range(min(settings.workers, len(pending)))
            ]
            await asyncio.gather(*tasks)
            outcome = "completed"
        except Exception:
            outcome = "error"
            raise
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                save_results(settings.output, results, samples)
                atomic_json(
                    Path(f"{prefix}.summary.json"),
                    {
                        "outcome": outcome,
                        "planned": len(pending),
                        "skipped": skipped,
                        "succeeded": succeeded,
                        "failed": failed,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
            finally:
                if provider is None and client is not None:
                    await client.close()
    return RunSummary(
        len(pending), skipped, succeeded, failed, time.monotonic() - started, str(diagnostics)
    )
