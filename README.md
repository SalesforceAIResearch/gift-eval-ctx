# GIFT-Eval-CTX

We present GIFT-Eval-CTX, a benchmark for evaluating whether forecasting methods can convert verified future information into accurate numerical trajectories. GIFT-Eval-CTX contains 10,000 instances drawn from 20 GIFT-Eval datasets, covering 7 domains and 6 frequencies, with prediction horizons ranging from 30 to 900 steps. Each instance provides structured effects derived from the realized future: Macro effects describe horizon-wide behavior, while Micro effects specify localized changes.

Useful links: 📄 [Paper](https://arxiv.org/abs/2610.17794), 📔 [Blog](), 📊 [Data](https://huggingface.co/datasets/Salesforce/GiftEvalCTX), and 🏆 [Leaderboard](https://huggingface.co/spaces/Salesforce/GIFT-Eval-CTX).

## Installation

Python 3.11 or later is required. Install from a checkout:

```bash
git clone https://github.com/SalesforceAIResearch/gift-eval-ctx.git
cd gift-eval-ctx
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'
```

For a single provider, replace `all` with `openai` or `anthropic`.

## Downloading Data

To load all benchmark instances:

```python
from datasets import load_dataset

dataset = load_dataset("Salesforce/GiftEvalCTX", split="test")
```

To download the complete repository, including the benchmark data and all methods' results:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Salesforce/GiftEvalCTX",
    repo_type="dataset",
    local_dir="./GiftEvalCTX",
)
```

## Running Evaluations

Create a `.env` file in the directory where you run the evaluator and fill in the API key for your provider:

```dotenv
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

Replace the model ID in the examples below with a model available in your provider account.

OpenAI:

```bash
python -m gift_eval_ctx \
  --input GiftEvalCTX/data/structured_effects.parquet \
  --output results/openai_hist_effects.parquet \
  --provider openai --model "your-openai-model-id" \
  --mode hist_effects --workers 10 --save-every 50 --max-tokens 6000
```

Anthropic:

```bash
python -m gift_eval_ctx \
  --input GiftEvalCTX/data/structured_effects.parquet \
  --output results/anthropic_hist_effects.parquet \
  --provider anthropic --model "your-anthropic-model-id" \
  --mode hist_effects --workers 10 --save-every 50 --max-tokens 6000
```

For the history-only comparison, use `--mode hist_only` and a separate output path.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--mode` | `hist_effects` | `hist_effects` or `hist_only` |
| `--workers` | `10` | Maximum concurrent samples |
| `--save-every` | `50` | Checkpoint after this many completed samples |
| `--max-tokens` | `6000` | Provider output budget, including reasoning where applicable |
| `--timeout` | `300` | Wall-clock seconds per API attempt, including streaming |
| `--retries` | `2` | Additional attempts for transient API errors |
| `--limit` | unset | Consider the first N input samples |
| `--batch` | unset | Run the next N incomplete samples within the selected input |
| `--reasoning-effort` | unset | Provider/model-specific effort setting |
| `--temperature` | unset | Optional sampling temperature in `[0, 1]` |
| `--thinking` | unset | Anthropic `adaptive`, `enabled`, or `disabled` |
| `--thinking-budget` | unset | Anthropic manual thinking budget; requires `--thinking enabled` |

Sampling and reasoning options are omitted unless requested. Their availability depends on the selected model. With Anthropic manual thinking, the thinking budget must be at least 1024 and less than `--max-tokens`. Do not set temperature when enabling thinking.

For provider-specific settings, see the [OpenAI API documentation](https://developers.openai.com/api/reference/python/resources/responses/methods/create) and [Anthropic SDK documentation](https://platform.claude.com/docs/en/api/sdks/python).

## Output Format

The output Parquet contains six columns:

| Field | Type | Meaning |
| --- | --- | --- |
| `dataset` | string | Dataset/configuration identifier |
| `term` | string | Forecast term |
| `item_id` | string | Series identifier |
| `window_id` | int64 | Forecast window identifier |
| `forecast` | list of float64 | Numerical forecast with exactly `n_horizon` values |
| `mase` | float64 | Mean Absolute Scaled Error |

Forecasts shorter than the horizon are padded by repeating the final value; longer forecasts are truncated to the horizon. Samples with invalid forecast responses are recorded as failures.

## Resume and Logs

Rerun the same command with the same output path to resume. Completed samples are skipped; failed or unfinished samples can be processed in later runs.

`--batch 1000` processes the next 1000 unfinished samples per run. `--limit 1000` restricts evaluation to the first 1000 input samples.

The following settings are checked when resuming: `--provider`, `--model`, `--mode`, `--reasoning-effort`, `--temperature`, `--thinking`, or `--thinking-budget`.

Run configurations, model responses, failure details, and summaries are saved in `logs/` beside the output Parquet. For example, logs for `results/model.parquet` are stored in `results/logs/`.

## Submitting Results

To submit your results, refer to the instructions in the [GIFT-Eval-CTX dataset repository](https://huggingface.co/datasets/Salesforce/GiftEvalCTX).

## Citation

If you find this benchmark helpful, please cite our paper:

```bibtex
@article{liu2026giftevalctx,
  title={GIFT-Eval-CTX: A Benchmark For Contextual Forecasting Model Evaluation},
  author={Liu, Xu and Yan, Hanshu and Aksu, Taha and Pham, Quang and Qiao, Zhongzheng and Liew, Jun Hao and Savarese, Silvio and Sahoo, Doyen and Li, Junnan},
  journal={arXiv preprint arXiv:2610.17794},
  year={2026}
}
```
