# Structured annotation, a small trained model against prompted API models

Code, saved predictions, and cost logs for the article "I Fine Tuned a 0.6B Model to Beat Claude Haiku and GPT-6 Sol at Labeling. Better Instructions Recovered Only 9 of 22 Points" (`draft.md`).

The task labels 2,974 test commands from MASSIVE English (en-US) with a scenario, an intent, and a list of slots. A record counts only if every field matches the human label exactly. Systems compared on the same records:

| System | Whole record exact (percent) |
|---|---:|
| Qwen3 0.6B with LoRA, 3 seeds | 70.35 mean |
| Claude Haiku 5.5, label definitions in the prompt, 3 passes | 56.66 mean |
| GPT-6 Sol, original prompt, 1 pass | 52.66 |
| Claude Haiku 5.5, original prompt, 3 passes | 47.86 mean |

The GPT-6 Sol and label definition runs were added after the original test results were known. Read `RESULTS.md` for the integrity checks, costs, and limits.

## Folders

- `colab_scripts/` data loading, strict scoring (`schema.py`), prompted runs, LoRA training, and local model runs.
- `results/full/` predictions and summaries for the main test runs. `results/defs/` holds the label definition prompt versions and runs. `results/gpt6_sol/` holds the GPT-6 Sol run. `results/archive/` holds logs of failed attempts.
- `FROZEN_FULL.json` the settings and script hashes fixed before the main test run. `PLAN.md` the plan and decisions.
- `figures/` the images used in the article. `make_figures.py` redraws the tables from `draft.md`.
- The trained adapters (about 40 MB each) are not included. `colab_scripts/train_lora.py` retrains them in about 20 minutes on a Colab T4.

## Running it

The experiment ran on Google Colab through the Colab command line tool. Set `ANTHROPIC_API_KEY` (and `OPENAI_API_KEY` for the GPT-6 Sol script) in your environment first. The drivers upload the key to the Colab session as a temporary file and delete it at once. API calls cost money, so read `RESULTS.md` for what each run cost.

## Data and license notes

The prediction files contain MASSIVE texts and labels. MASSIVE is released under CC BY 4.0, copyright Amazon.com Inc. or its affiliates, see https://huggingface.co/datasets/AmazonScience/massive. Qwen3 0.6B is released under Apache 2.0. No license file is chosen for the code yet.
