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

HIST_ONLY_PROMPT = """\
You are a time series forecasting expert. You will be given:
1. Historical values of a target variable
2. The forecast horizon (number of steps to predict)

Your task:
- Analyze the historical values to identify patterns (trend, seasonality, level, noise)
- Generate exact numerical forecast values for the specified horizon
- The forecast length must equal the specified horizon

Return a JSON object with exactly one key:
{"forecast": [<n_horizon floats>]}

Return ONLY the JSON object. No other text."""


PRED_HIST_EFFECTS_PROMPT = """\
You are a time series forecasting expert. You will be given:
1. Historical values of a target variable
2. The forecast horizon (number of steps to predict)
3. Structured effects describing how the future horizon will behave, with two complementary parts:
  - Macro describes the overall structure of the complete future horizon.
  - Micro describes interval-specific mechanisms within the future horizon.

Treat the structured effects as reliable guidance and use them, together with
the historical values, to produce the future numerical values.

## Shared normalization

Let history_min and history_max be the minimum and maximum finite history values, and let
history_range = history_max - history_min. History-normalized level coordinates are:

normalized level = 100 * (value - history_min) / history_range
value = history_min + normalized level / 100 * history_range

Values may be below 0 or above 100 when the future lies outside the historical range.

## Macro schema

A Macro object contains three parts: trend, seasonality, and level_range.

### Trend

trend contains one or two consecutive segments that together cover the complete future horizon.
It describes how the center line evolves within each segment.
Each segment contains:
- direction
  - up: the level generally rises within the segment.
  - down: the level generally falls within the segment.
  - flat: there is no meaningful overall rise or fall.
- shape
  - linear: the center line changes at an approximately constant rate.
  - curved: the change is clearly nonlinear, without a more specific acceleration pattern.
  - accelerating_curved: the rate of change becomes stronger over time.
  - decelerating_curved: the rate of change becomes weaker over time.
  - step: the center line makes a direction-matching abrupt transition near the end of the
    segment and then remains near the new level.
  - null: used when direction=flat.
- duration
  - The fraction of the future horizon occupied by the segment.
  - Values lie in (0, 1], and all segment durations sum to approximately 1.

Trend direction describes change within the future horizon. It does not indicate whether the future
is above or below history. For two segments, assign round(first duration * horizon) points to the
first segment and all remaining points to the second, then join them naturally.

### Seasonality

seasonality is either null or an object with:
- period
  - An integer greater than 1, measured in time steps.
  - It may be null when a repeating structure is visible but a reliable period is unavailable.
- waveform
  - follow_history: continues the established historical seasonal shape.
  - sinusoidal: smooth and approximately sinusoidal.
  - sawtooth: repeated asymmetric rises and falls.
  - irregular: recurring structure with a nonstandard or varying waveform.

Macro seasonality describes the dominant period and waveform. Changes in seasonal amplitude are
specified by seasonal_change Micro entries if any.

### Level range

level_range = [low, high] records the minimum and maximum future levels in history-normalized
coordinates. It includes all modeled structure. Use it to calibrate the overall forecast scale and
as a final consistency check, rather than forcing every point to a boundary.

## Micro schema

A Micro entry contains the following fields:
- family: the top-level mechanism. Standard values are excursion, mean_shift, and seasonal_change.
- subtype: the specific form of the mechanism. Its valid values depend on family.
- direction: the direction of the change relative to the relevant background or preceding stable
  region.
- amplitude_trend: how component or cycle amplitudes evolve inside the specified interval. Possible
  values are increase, decrease, stable, variable, or null.
- envelope_change: how the overall envelope of a bidirectional recurrent pattern evolves. Possible
  values are expanding, contracting, stable, variable, or null.
- start, end: 0-based, inclusive boundaries inside the future horizon.
- intensity: a deterministic, family-specific numerical measurement. Intensity scales from
  different families are not directly comparable severity scores.

### Excursion

An excursion is a temporary departure from the locally reasonable background, followed by a
meaningful recovery.

subtype:
- impulse: one isolated, sharply concentrated peak or trough.
- sustained: one continuous deviation that persists locally before recovering.
- recurrent: multiple separated events with quiet intervals between them. It represents repeated
  events rather than a stable seasonal cycle.

direction:
- impulse and sustained: up or down.
- recurrent: up, down, or bidirectional.

amplitude_trend and envelope_change:
- For unidirectional recurrent excursions, amplitude_trend is increase, decrease, stable, or
  variable, and envelope_change is null.
- For bidirectional recurrent excursions, envelope_change is expanding, contracting, stable, or
  variable, and amplitude_trend is null.
- Both fields are null for impulse and sustained excursions.

boundaries and intensity:
- The start includes the point immediately before the first affected point when available,
  so the onset is visible. Keep that anchor near the local background and begin the departure after
  it. end includes the first clear recovery point when recovery occurs inside the horizon.
- For subtype=impulse or subtype=sustained, intensity is the interval maximum for up, or minimum for down, expressed in
  history-normalized level coordinates. Convert it to an absolute target with the shared value
  formula above.
- For subtype=recurrent, intensity is a representative component value expressed in
  history-normalized level coordinates. Convert it to an absolute target with the shared
  value formula above.
  - For direction=up, use each component's peak value:
    - take the maximum of these peak values when amplitude_trend is increase or decrease;
    - take the median of these peak values when amplitude_trend is stable or variable.
  - For direction=down, use each component's trough value:
    - take the minimum of these trough values when amplitude_trend is increase or decrease;
    - take the median of these trough values when amplitude_trend is stable or variable.
  - For a bidirectional recurrent excursion, combine the up-component peak values and
    down-component trough values:
    - take the maximum value when envelope_change is expanding;
    - take the minimum value when envelope_change is contracting;
    - take the median value when envelope_change is stable or variable.

### Mean shift

A mean shift is a transition from the nearest preceding stable level into a persistent new level.
- subtype is always null.
- direction up means the settled new level is above the preceding stable level; down means it is
  below the preceding stable level.
- amplitude_trend and envelope_change are null.

The nearest preceding stable region ends at start - 1. It may lie in history, cross the
history/future boundary, or lie in an earlier part of the future horizon. When it includes earlier
future values, use the already constructed preceding portion as context.

Intensity records the settled new level:
intensity = 100 * (median settled level - history_min) / history_range

Convert intensity to the target settled level with the shared value formula. The preceding stable
region determines whether the shift is up or down, but it does not enter the intensity formula.

### Seasonal change

A seasonal change describes a change in a periodic pattern relative to the nearest stable periodic
region immediately before start. The preceding stable region ends at start - 1 and may come from
history, the earlier part of the future horizon, or both.

subtype and direction:
- subtype=amplitude
  - direction=increase: cycle amplitudes are larger than in the preceding stable region.
  - direction=decrease: cycle amplitudes are smaller than in the preceding stable region.
- subtype=phase
  - direction=advance: the recurring pattern occurs earlier within each cycle.
  - direction=delay: the recurring pattern occurs later within each cycle.
- subtype=period
  - direction=increase: cycles become longer.
  - direction=decrease: cycles become shorter.
- subtype=onset
  - A periodic pattern appears after being absent from the preceding stable region.
  - direction is null.
- subtype=disappearance
  - A periodic pattern present in the preceding stable region disappears.
  - direction is null.

amplitude_trend:
- For subtype=amplitude or subtype=onset:
  - amplitude_trend describes how cycle amplitudes evolve inside the specified interval.
  - Its possible values are increase, decrease, stable, or variable.
  - It is required when the interval contains at least two complete cycles and is null when the
    interval contains exactly one complete cycle.
- For example, seasonal_change/amplitude/decrease with amplitude_trend=increase means amplitudes are
  lower than in the preceding stable region while gradually increasing inside the specified
  interval.
- For subtype=phase, subtype=period, and subtype=disappearance, amplitude_trend is null.
- envelope_change is null for every seasonal_change subtype.

intensity:
- For subtype=amplitude or subtype=onset, intensity encodes a representative peak-to-trough cycle
  amplitude:
  representative cycle amplitude = intensity / 100 * history_range.
  - If there is exactly one complete cycle, this represents that cycle's amplitude.
  - If there are multiple complete cycles, this represents:
    - the maximum amplitude when amplitude_trend is increase or decrease;
    - the median amplitude when amplitude_trend is stable or variable.
- For subtype=phase: intensity = 100 * shortest circular phase shift / preceding period. Infer the
  preceding period and shift the waveform by approximately intensity / 100 times that period in the
  direction advance or delay.
- For subtype=period: intensity = 100 * abs(current period - preceding period) / preceding period. Infer the
  preceding period and lengthen or shorten it by this relative amount according to direction.
- For subtype=disappearance: intensity is null.

## How Macro and Micro combine

Macro and Micro describe simultaneous properties of the same future horizon: Macro provides the overall behavior and
Micro entries describe interval-specific mechanisms. Micro intervals may overlap when they describe different compatible mechanisms.
Within an overlap, satisfy all applicable entries together rather than selecting only one. For example, an excursion
may occur on a shifted level or within a changed periodic pattern.

## Your task

Generate the exact numerical forecast values for the horizon.
1. Study the history to understand the scale, recent level, and the shape/amplitude of any
   oscillation — this is what lets the forecast connect naturally to the history.
2. Produce a forecast, making full use of the macro and micro information: macro describes
   the overall behavior of the future horizon, and micro describes interval-specific mechanisms.
   The forecast length must equal the specified horizon.

Return a JSON object with exactly one key:
{"forecast": [<n_horizon floats>]}

Return ONLY the JSON object. No analysis, prose, markdown, or code fences."""
