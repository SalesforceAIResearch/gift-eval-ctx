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
import re
from dataclasses import dataclass

_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")


@dataclass(frozen=True)
class ParsedForecast:
    """Normalized values and the repairs applied to the response."""

    values: list[float]
    original_length: int
    padded: int
    truncated: int
    recovered_json: bool


def _numeric_prefix(text: str) -> list[float]:
    """Recover only a contiguous array prefix with explicitly delimited numbers.

    A number at EOF may itself be truncated (e.g. 12 instead of 123 or 1e).
    Discard that final token. Never skip malformed entries inside the array.
    """
    start = re.match(r'\s*\{\s*"forecast"\s*:\s*\[', text)
    if start is None:
        raise ValueError("Response must be a JSON object containing forecast")
    position = start.end()
    values: list[float] = []
    while True:
        tail = text[position:].lstrip()
        if not tail:
            return values
        if tail.startswith("]"):
            if values:
                raise ValueError("Trailing comma in forecast")
            if not re.fullmatch(r"\]\s*}?\s*", tail):
                raise ValueError("Unexpected content after the forecast array")
            return values
        match = _NUMBER.match(tail)
        if match is None:
            if tail == "-":
                return values
            raise ValueError("Non-numerical value in forecast")
        number = float(match.group())
        if not math.isfinite(number):
            raise ValueError("Forecast values must be finite")
        rest = tail[match.end() :].lstrip()
        if not rest:
            return values
        if rest[0] not in ",]":
            # A partially written exponent or decimal is recoverable only at EOF.
            if re.fullmatch(r"(?:\.|[eE][+-]?)\s*", rest):
                return values
            raise ValueError("Malformed numerical value in forecast")
        values.append(number)
        if rest[0] == "]":
            if not re.fullmatch(r"\]\s*}?\s*", rest):
                raise ValueError("Unexpected content after the forecast array")
            return values
        position = len(text) - len(rest)
        if rest[0] == ",":
            position += 1


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("Duplicate keys in forecast JSON")
    return obj


def parse_forecast(raw: str, horizon: int) -> ParsedForecast:
    """Parse JSON, validate finite numbers, and pad/truncate to the horizon.

    Complete JSON objects, outer Markdown fences, and a truncated numerical
    array are supported. Short forecasts repeat their final value. Long forecasts
    retain their first ``horizon`` values. No forecast values are clipped.
    """
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", text, flags=re.DOTALL)
    if fence is not None:
        text = fence.group(1).strip()
    recovered = False
    try:
        obj = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError:
        # Repair the narrow, unambiguous case of swapped final object/array closers.
        swapped = re.sub(r"}\s*]\s*$", "]}", text)
        if swapped != text:
            try:
                obj = json.loads(swapped, object_pairs_hook=_unique_object)
            except json.JSONDecodeError as exc:
                raise ValueError("Malformed forecast JSON") from exc
            if not isinstance(obj, dict) or set(obj) != {"forecast"}:
                raise ValueError("Response must contain exactly one key: forecast") from None
            values = obj["forecast"]
        else:
            values = _numeric_prefix(text)
        recovered = True
    else:
        if not isinstance(obj, dict) or set(obj) != {"forecast"}:
            raise ValueError("Response must contain exactly one key: forecast") from None
        values = obj["forecast"]
    if not isinstance(values, list) or not values:
        raise ValueError("Forecast must be a nonempty numerical array")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError("Forecast values must be numbers, without strings, booleans, or nulls")
    try:
        result = [float(v) for v in values]
    except OverflowError as exc:
        raise ValueError("Forecast values must be finite") from exc
    if not all(math.isfinite(v) for v in result):
        raise ValueError("Forecast values must be finite")
    original_length = len(result)
    padded, truncated = max(0, horizon - original_length), max(0, original_length - horizon)
    result = result[:horizon]
    result.extend([result[-1]] * padded)
    return ParsedForecast(result, original_length, padded, truncated, recovered)
