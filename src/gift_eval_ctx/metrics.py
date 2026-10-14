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

import re
from collections.abc import Sequence
from functools import cache

import numpy as np
from pandas.tseries.frequencies import to_offset

_SEASONALITIES = {"S": 3600, "MIN": 1440, "H": 24, "D": 1, "W": 1, "M": 12, "B": 5, "Q": 4}
_ALIASES = {"S": "s", "T": "min", "H": "h", "M": "ME", "Q": "QE", "Y": "YE"}


@cache
def seasonality_for_frequency(frequency: str) -> int:
    """Return the GIFT-Eval seasonal lag, independent of Pandas alias spelling.

    Examples: H -> 24, 10T -> 144, 15T -> 96, 5T -> 288, 10S -> 360, D -> 1.
    Multiples that do not divide the base lag fall back to 1.
    """
    match = re.fullmatch(r"(\d*)([A-Za-z]+)(-.*)?", frequency)
    if match is None:
        raise ValueError(f"Invalid frequency: {frequency!r}")
    multiple, unit, anchor = match.groups()
    normalized = f"{multiple}{_ALIASES.get(unit, unit)}{anchor or ''}"
    offset = to_offset(normalized)
    if offset.n <= 0:
        raise ValueError("Frequency must be positive")
    base = offset.name.split("-")[0].upper()
    if len(base) > 1 and base.endswith(("S", "E")) and base not in {"MS", "US", "NS"}:
        base = base[:-1]
    # Month-start is MS; millisecond is lowercase ms before normalization.
    if offset.name.split("-")[0] == "MS":
        base = "M"
    lag, remainder = divmod(_SEASONALITIES.get(base, 1), offset.n)
    return int(lag) if remainder == 0 else 1


def seasonal_error(history: Sequence[float], seasonality: int, epsilon: float = 1e-8) -> float:
    """Compute the seasonal-naive scale after removing missing history values.

    This retains the existing benchmark convention: remove NaNs before applying
    the lag, fall back to lag 1 for short histories, and floor the scale at epsilon.
    """
    if seasonality < 1:
        raise ValueError("Seasonality must be positive")
    values = np.asarray(history, dtype=float)
    values = values[~np.isnan(values)]
    lag = seasonality if seasonality < len(values) else 1
    if len(values) <= lag:
        return epsilon
    return max(float(np.mean(np.abs(values[lag:] - values[:-lag]))), epsilon)


def mase(
    future: Sequence[float], forecast: Sequence[float], history: Sequence[float], seasonality: int
) -> float:
    """Score equal-length forecasts; ignore missing targets, without imputation."""
    actual, predicted = np.asarray(future, dtype=float), np.asarray(forecast, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("Future and forecast must be one-dimensional and have equal length")
    valid = ~np.isnan(actual) & ~np.isnan(predicted)
    if not valid.any():
        return float("nan")
    error = float(np.mean(np.abs(actual[valid] - predicted[valid])))
    return error / seasonal_error(history, seasonality)
