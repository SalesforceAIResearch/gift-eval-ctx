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

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq

from gift_eval_ctx.metrics import seasonality_for_frequency
from gift_eval_ctx.prompts import HIST_ONLY_PROMPT, PRED_HIST_EFFECTS_PROMPT

Mode = Literal["hist_effects", "hist_only"]
SampleKey = tuple[str, str, str, int]
KEY_COLUMNS = ["dataset", "term", "item_id", "window_id"]
SYSTEM_PROMPTS = {"hist_effects": PRED_HIST_EFFECTS_PROMPT, "hist_only": HIST_ONLY_PROMPT}


@dataclass(frozen=True)
class Sample:
    """One sample; ``future`` is used exclusively for local scoring."""

    key: SampleKey
    history: list[float]
    future: list[float]
    horizon: int
    frequency: str
    macro: dict[str, Any]
    micro: list[dict[str, Any]]


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Structured effects must contain finite numbers")
    return number


def _series(values: Any, name: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty numerical list")
    if any(
        isinstance(v, bool) or (v is not None and not isinstance(v, (int, float))) for v in values
    ):
        raise ValueError(f"{name} contains non-numerical values")
    result = [float(v) if v is not None else float("nan") for v in values]
    if any(math.isinf(v) for v in result) or not any(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain finite values and no infinities")
    return result


def load_samples(path: Path, mode: Mode) -> list[Sample]:
    """Load required columns, reject duplicate keys, and validate before any API calls."""
    if mode not in SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported mode: {mode}")
    columns = [*KEY_COLUMNS, "history", "future", "n_horizon", "frequency"]
    if mode == "hist_effects":
        columns.extend(["macro", "micro"])
    missing = set(columns) - set(pq.read_schema(path).names)
    if missing:
        raise ValueError(f"Input is missing columns: {', '.join(sorted(missing))}")
    samples: list[Sample] = []
    seen: set[SampleKey] = set()
    for index, row in enumerate(pq.read_table(path, columns=columns).to_pylist()):
        try:
            if any(not isinstance(row[c], str) or not row[c] for c in KEY_COLUMNS[:3]):
                raise ValueError("dataset, term, and item_id must be nonempty strings")
            if type(row["window_id"]) is not int:
                raise ValueError("window_id must be an integer")
            key = (row["dataset"], row["term"], row["item_id"], row["window_id"])
            if key in seen:
                raise ValueError(f"Duplicate sample key: {key}")
            seen.add(key)
            horizon = row["n_horizon"]
            if type(horizon) is not int or horizon < 1:
                raise ValueError("n_horizon must be a positive integer")
            history, future = _series(row["history"], "history"), _series(row["future"], "future")
            if len(future) != horizon:
                raise ValueError("Future length does not match n_horizon")
            frequency = row["frequency"]
            if not isinstance(frequency, str):
                raise ValueError("frequency must be a string")
            seasonality_for_frequency(frequency)
            macro, micro = {}, []
            if mode == "hist_effects":
                macro = json.loads(
                    row["macro"], parse_constant=_reject_constant, parse_float=_finite_json_float
                )
                micro = json.loads(
                    row["micro"], parse_constant=_reject_constant, parse_float=_finite_json_float
                )
                if not isinstance(macro, dict) or not isinstance(micro, list):
                    raise ValueError("macro must encode an object and micro must encode a list")
                if not all(isinstance(event, dict) for event in micro):
                    raise ValueError("Every micro entry must be an object")
            samples.append(Sample(key, history, future, horizon, frequency, macro, micro))
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Invalid input row {index}: {exc}") from exc
    return samples


def build_user_message(sample: Sample, mode: Mode) -> str:
    """Format history at four significant digits and reuse the original effects message."""
    if mode not in SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported mode: {mode}")
    history = ", ".join("null" if not math.isfinite(v) else f"{v:.4g}" for v in sample.history)
    message = f"Forecast horizon: {sample.horizon} steps\nHistorical values: [{history}]"
    if mode == "hist_effects":
        effects = json.dumps({"macro": sample.macro, "micro": sample.micro}, indent=2)
        message += f"\n\nStructured effects (describing future behavior):\n{effects}"
        bounds = sample.macro.get("level_range")
        finite = [v for v in sample.history if math.isfinite(v)]
        if (
            isinstance(bounds, list)
            and len(bounds) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in bounds)
            and finite
        ):
            low, span = min(finite), max(finite) - min(finite)
            raw_low, raw_high = [low + v / 100 * span for v in bounds]
            message += (
                "\n\nFor this sample, the macro level_range corresponds to the inclusive "
                f"raw-value range [{raw_low:.10g}, {raw_high:.10g}] over the forecast "
                "horizon. Every forecast value "
                f"must stay within this range: no value below {raw_low:.10g} or above "
                f"{raw_high:.10g}."
            )
    message += (
        '\n\nRespond with only the JSON object {"forecast": [...]}. The forecast array '
        f"must contain exactly {sample.horizon} numerical values—no more and no fewer. "
        "Count and verify the values before finishing. No analysis or prose."
    )
    return message
