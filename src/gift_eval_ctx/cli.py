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

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from gift_eval_ctx.data import SYSTEM_PROMPTS
from gift_eval_ctx.providers import RequestSettings
from gift_eval_ctx.runner import RunError, RunSettings, run_evaluation


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate forecasts on GIFT-Eval-CTX using official APIs."
    )
    parser.add_argument("--input", type=Path, required=True, help="Benchmark Parquet file")
    parser.add_argument("--output", type=Path, required=True, help="Result Parquet file")
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    parser.add_argument(
        "--model", required=True, help="Model ID available in your provider account"
    )
    parser.add_argument("--mode", choices=list(SYSTEM_PROMPTS), default="hist_effects")
    parser.add_argument("--workers", type=_positive_int, default=10)
    parser.add_argument(
        "--save-every", type=_positive_int, default=50, help="Checkpoint after N completed samples"
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=6000,
        help="Output token budget (including reasoning where applicable)",
    )
    parser.add_argument(
        "--timeout", type=float, default=300, help="Maximum seconds per API attempt"
    )
    parser.add_argument(
        "--retries",
        type=_nonnegative_int,
        default=2,
        help="Additional attempts for transient API errors",
    )
    parser.add_argument(
        "--reasoning-effort", help="Provider/model-specific effort; omitted by default"
    )
    parser.add_argument(
        "--temperature", type=float, help="Omitted by default; only set for models that support it"
    )
    parser.add_argument(
        "--thinking", choices=["adaptive", "enabled", "disabled"], help="Anthropic only"
    )
    parser.add_argument(
        "--thinking-budget", type=_positive_int, help="Anthropic manual thinking token budget"
    )
    parser.add_argument(
        "--limit", type=_positive_int, help="Consider only the first N input samples"
    )
    parser.add_argument(
        "--batch", type=_positive_int, help="Evaluate the next N incomplete samples"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Return 0 for success, 1 for sample failures, 2 for run errors, or 130 for interruption."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        request = RequestSettings(
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            thinking=args.thinking,
            thinking_budget=args.thinking_budget,
        )
        settings = RunSettings(
            input=args.input,
            output=args.output,
            mode=args.mode,
            workers=args.workers,
            save_every=args.save_every,
            retries=args.retries,
            limit=args.limit,
            batch=args.batch,
        )
        summary = asyncio.run(run_evaluation(settings, request))
    except KeyboardInterrupt:
        logging.info("Interrupted; completed forecasts were checkpointed. Rerun to resume.")
        return 130
    except (ValueError, OSError, RunError) as exc:
        logging.error("%s", exc)
        return 2
    logging.info(
        "Finished in %.1fs: %d succeeded, %d failed, %d skipped. Output: %s",
        summary.elapsed_seconds,
        summary.succeeded,
        summary.failed,
        summary.skipped,
        args.output,
    )
    if summary.diagnostics is not None:
        logging.info("Diagnostics: %s", summary.diagnostics)
    return 1 if summary.failed else 0
