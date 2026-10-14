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
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from gift_eval_ctx.data import KEY_COLUMNS, Mode, Sample, SampleKey
from gift_eval_ctx.providers import RequestSettings

OUTPUT_SCHEMA = pa.schema(
    [(c, pa.string()) for c in KEY_COLUMNS[:3]]
    + [("window_id", pa.int64()), ("forecast", pa.list_(pa.float64())), ("mase", pa.float64())]
)


def experiment_settings(mode: Mode, request: RequestSettings) -> dict[str, Any]:
    """Compare explicit model settings when appending to an existing output.

    max_tokens and timeout are operational settings. They do not invalidate
    completed samples, and remain recorded in the per-run configuration.
    """
    parameters = asdict(request)
    parameters.pop("max_tokens")
    parameters.pop("timeout")
    return {"mode": mode, "request": parameters}


def atomic_json(path: Path, value: Any) -> None:
    """Replace a JSON file only after the complete document has reached disk."""
    _atomic_write(
        path,
        lambda temporary: temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        ),
    )


def _atomic_write(path: Path, write: Callable[[Path], Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        write(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def check_manifest(output: Path, experiment: dict[str, Any]) -> Path:
    """Prevent mixing explicit model settings or modes in one output file."""
    path = output.parent / "logs" / f"{output.name}.manifest.json"
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid output manifest: {path}") from exc
        if not isinstance(stored, dict):
            raise ValueError(f"Invalid output manifest: expected a JSON object in {path}")
        previous = {key: stored.get(key) for key in experiment}
        if previous != experiment:
            changed = [key for key in experiment if experiment[key] != previous[key]]
            raise ValueError(
                f"Output configuration differs ({', '.join(changed)}). Use a different --output path."
            )
        if stored != experiment:
            atomic_json(path, experiment)
    elif output.exists():
        raise ValueError("Existing output has no manifest; use a different --output path")
    else:
        atomic_json(path, experiment)
    return path


def load_results(output: Path, samples: list[Sample]) -> dict[SampleKey, dict[str, Any]]:
    """Keep valid completed rows; invalid forecasts or scores remain retryable."""
    if not output.exists():
        return {}
    table = pq.read_table(output)
    if not table.schema.equals(OUTPUT_SCHEMA, check_metadata=False):
        raise ValueError("Existing output must match the six-column evaluation schema and types")
    by_key = {sample.key: sample for sample in samples}
    seen: set[SampleKey] = set()
    results: dict[SampleKey, dict[str, Any]] = {}
    for row in table.to_pylist():
        key = tuple(row[c] for c in KEY_COLUMNS)
        if key not in by_key or key in seen:
            raise ValueError(f"Unknown or duplicate result key: {key}")
        seen.add(key)
        values, score = row["forecast"], row["mase"]
        valid = (
            isinstance(values, list)
            and len(values) == by_key[key].horizon
            and all(type(v) in (int, float) and math.isfinite(v) for v in values)
            and type(score) in (int, float)
            and math.isfinite(score)
            and score >= 0
        )
        if valid:
            results[key] = row
    return results


def save_results(
    output: Path, results: dict[SampleKey, dict[str, Any]], samples: list[Sample]
) -> None:
    """Write only the public six columns, preserving the original input order."""
    rows = [results[sample.key] for sample in samples if sample.key in results]
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    _atomic_write(output, lambda temporary: pq.write_table(table, temporary, compression="zstd"))
