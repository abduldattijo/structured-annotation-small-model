# Full experiment results — 2026-10-22

The experiment completed on a Colab T4. The session was stopped after the final download; `colab --auth=adc sessions` reported no active sessions. Raw predictions, summaries, and three saved LoRA adapters are under `results/full/`. The predeclared settings are in `FROZEN_FULL.json`.

## Task and outcome

MASSIVE English contains short **voice-assistant commands**, not support tickets. The task is to return an exact scenario, intent, and ordered list of slot/value pairs. The held-out test set has 2,974 records. Whole-record accuracy requires every field to match the dataset label exactly.

| System | Full-test runs | Whole-record exact accuracy | Scenario | Intent | Slots exact |
|---|---:|---:|---:|---:|---:|
| Qwen3 0.6B with LoRA, seeds 0–2 | 3 | 69.91%, 70.85%, 70.31% (mean **70.35%**) | 91.22%–91.46% | 88.16%–88.50% | 76.50%–77.64% |
| Claude Haiku 5.5, three prompted passes | 3 | 47.78%, 47.92%, 47.88% (mean **47.86%**) | 87.39%–87.49% | 82.48%–82.78% | 55.88%–56.42% |
| Claude Haiku 5.5 with prompt caching | 1 | 47.88% | 87.32% | 82.62% | 55.85% |
| GPT-6 Sol, frozen prompt, reasoning effort `none` | 1 | **52.66%** | 90.22% | 85.78% | 59.58% |
| Claude Haiku 5.5 with label definitions and prompt caching | 3 | 56.72%, 56.35%, 56.89% (mean **56.66%**) | 89.41%–89.68% | 85.27%–85.54% | 64.12%–64.49% |

The LoRA model's mean whole-record score is 22.50 percentage points above this prompted Haiku baseline. This is a comparison on one narrow annotation contract, not evidence that a 0.6B model generally outperforms larger models. Scenario names can be inferred from intent names in this dataset, so the fields are not independent.

The additional GPT-6 Sol pass used the same frozen prompt and 2,974 test items. It scored 4.80 percentage points above the mean Haiku pass and 17.69 points below the mean LoRA seed. This is one GPT-6 Sol pass, so the comparison does not measure its run-to-run variation. The model used `reasoning_effort="none"`; results with other reasoning settings are unknown.

The later label-definition test added one-line meanings for every intent and slot type to Haiku's prompt. Definitions were drafted from training rows. Version 1 scored 52.0% on the same 300 development rows where the base prompt scored 44.7%; one dev-error-driven revision scored 53.3% and was frozen. Three cached test passes with that frozen version averaged 56.66%, 8.80 points above the original Haiku mean and 13.69 below the LoRA mean. This is a **post hoc follow-up**: the test set and earlier model results had already been seen before this question was posed. It supports a useful prompt improvement on these records, not an independent confirmation of a causal explanation. The changed prompt is also longer, so definitions and added context were not isolated from each other.

## Cost and speed

| Measurement | Result |
|---|---:|
| Haiku uncached API usage, per 2,974-record pass | $0.3703, or about $1.25 per 10,000 records |
| Haiku cached API usage, 2,974 records | $0.0933, or about $0.31 per 10,000 records |
| GPT-6 Sol API usage, 2,974 records | $4.5635, or about $15.34 per 10,000 records at the observed token mix |
| Definition-prompt Haiku cached API usage, per 2,974-record pass | $0.1360–$0.1362, about $0.46 per 10,000 records, excluding a cache-priming request |
| Haiku mean cache-read tokens per cached record | 1,036 |
| Haiku median single-request latency, 100-record dev samples | 0.997 s uncached; 0.993 s cached |
| Qwen batched T4 inference, three seeds | 38.31–39.93 records/s; 74.5–77.6 s for 2,974 records after loading |
| Qwen median single-record latency, three seeds | 0.963–1.018 s |
| Qwen training, seeds 1 and 2 | 1,231 s and 1,156 s; peak 7.42 GB GPU memory |
| Colab balance, start of successful resumed run to stop | 597.89 to 596.56 compute units (1.33 units, including session setup and all remaining work) |

Haiku's cached usage figure omits the **one priming request** used to create the cache: the script made that call but did not add it to its cost counter. Its exact charge was not retained. Do not present $0.0933 as the complete bill. The Colab balance is measured in compute units; converting it into money requires the actual purchase price and an allocation rule for training and idle time. The seed 0 training metadata file was lost during an earlier Colab restart, although its adapter, dev check, and full-test predictions were saved. Its training duration should not be quoted from the seed 1 and 2 metadata.

The definition follow-up used the same cache-priming code, so its recorded $0.408356 across three test passes also excludes three priming requests. The two 300-row development calls recorded $0.080179 and $0.080889. No reply in the three definition test passes hit the 400-token cutoff.

## Integrity checks and limits

- Each of the seven full-test prediction files contains exactly 2,974 records with 2,974 distinct IDs, matching its summary. The seven files have the same IDs and gold labels. Both 100-record latency files have 100 distinct IDs. The frozen experiment scripts still match the manifest hashes.
- The additional GPT-6 Sol file also has 2,974 distinct IDs, identical gold labels, zero API errors, zero incomplete replies, and zero parse failures. Its system prompt hash matches the earlier frozen prompt. The 50-record development pilot cost $0.0764; the complete test pass cost $4.5635 based on recorded token usage. The Colab CPU session was stopped afterward.
- Each of the three definition-prompt test files has 2,974 distinct IDs and the same gold labels as the original Haiku cached pass. Their prompt hashes match frozen SHA-256 `3150ce689c93ccbc96eacc4d8efd84f3adca117e7dcc51e56a68b4893fd1b5eb`; independently rescoring their raw replies reproduced the summary scores. The runs had 11, 9, and 11 unparseable replies, zero transport errors, and zero cutoff replies. On pass 1 against the original cached Haiku pass, definitions alone corrected 422 records, while the original prompt alone got 159 records right.
- The three uncached Haiku passes had 9, 12, and 7 unparseable responses. They had zero transport errors; one pass had one cutoff reply. The cached pass had 11 unparseable responses and one cutoff reply.
- The prompt was selected on 300 development records before the full test. The chosen three-example prompt scored 44.7% whole-record exact on that sample; the 18- and 36-example versions cost more and scored 42.7% and 43.3%. This does not establish that no better Haiku prompt exists.
- The JSON parser and reply token limit were adjusted after development results and before the full test. The full-run manifest records later operational refreezes after memory, upload, and Colab-session failures; describe those in any article. The test results were not used to change the model or scoring settings.
- The three Qwen seeds vary training randomness; the three Haiku passes vary response randomness on the **same** test items. Their narrow ranges do not represent uncertainty over new datasets or businesses.
- MASSIVE's labels include annotation conventions. Exact match rewards reproducing those conventions, not necessarily the best user-facing interpretation.

## Files

- `FROZEN_FULL.json`: frozen prompt hash, model settings, script hashes, and refreeze notes.
- `results/full/*_summary.json`: metrics and API usage.
- `results/full/*.jsonl`: per-record predictions and gold labels.
- `results/full/adapters/s*/adapter_model.safetensors`: saved LoRA adapters.
- `results/archive/`: failed-run logs and recovery history.
- `results/gpt6_sol/GPT6_RUN_MANIFEST.json`: additional model settings, prompt and input hashes, pilot result, and spending cap.
- `results/gpt6_sol/full.jsonl` and `full_summary.json`: GPT-6 Sol predictions, recorded API usage, and scored results.
- `results/defs/defs_v2_test*.jsonl` and matching summaries: frozen definition-prompt predictions, usage, and scores; `PLAN.md` records the development versions and their selection.

The next writing step is to build a Colab notebook and article from these saved results, with the voice-command scope and cost accounting caveats visible to readers.
