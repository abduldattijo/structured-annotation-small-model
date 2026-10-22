# Structured annotation: a small fine-tuned model against a prompted API model

Status: plan v1 (2026-10-22). Nothing has been run. No claim in this plan is a result.

## Reader question

When a team labels thousands of records with fixed fields, is it cheaper and as accurate to train a small model once, or to keep prompting a large API model on every record?

## Why this topic (from the research check)

- The three ideas in the pasted list cited arXiv links that point to other papers. Their numbers were not found. This plan uses only sources I opened or confirmed.
- The ICML 2026 listing "Reasoning Models Lose their Way with Long Deduction" covers a different problem, so it is not a source for this plan.

## Sources checked (2026-10-22)

| Source | What I confirmed | Use in the article |
|---|---|---|
| [propella-1, arXiv 2602.12414](https://arxiv.org/abs/2602.12414) | Document annotation, 18 properties in six categories, models of 0.6B, 1.7B and 4B parameters. The 4B model "achieves higher agreement than much larger general-purpose models". The page gives no score, no comparison model name, and no cost figure. | Background only. The 0.729 score from a search snippet is NOT on the page and must not be quoted. |
| [Merchant extraction with LoRA, arXiv 2606.08051](https://arxiv.org/pdf/2606.08051) | 270M to 8B models with LoRA on financial transaction extraction. The snippet reports 0.8B Qwen 3.5 at 94.75% F1 and mentions latency trade-offs. | Background. Read the full paper before citing any number. |
| [ETLCH, arXiv 2509.08381](https://arxiv.org/pdf/2509.08381) | 1B LLaMA with LoRA on 100 to 1000 examples per task. Authors say it outperforms baselines on most metrics. Baselines not confirmed. | Background. Confirm the baselines first. |
| [InsLogicBench, ACL 2026](https://aclanthology.org/2026.acl-long.1035/) | A real ACL 2026 long paper (2026.acl-long.1035, pages 22592 to 22619). Eight LLMs tested. Models give correct verdicts with weak justifications. Dataset size and scores not retrieved. | Only if the insurance idea is picked later. Read the full paper first. |
| Claude Haiku 5.5 prices | $0.10 per million input tokens and $0.50 per million output tokens, from the project's `config.py`, checked 2026-10-15. | Cost estimate. Re-check the pricing page before any spend. |

## Design

**Task.** A fixed-field extraction from short text, such as support tickets into category, urgency and product, or policy text into rule tags. The task must have a human-labelled test set.

**Dataset gate.** Before drafting, the dataset's source page must show its license, and the license must allow this use. Quote the license name in the article and in Selected Sources. Choose the dataset only after this check passes.

**Systems compared.**
1. Prompted API model: Claude Haiku 5.5, one fixed prompt and one fixed output schema.
2. LoRA-tuned small model: Qwen3 0.6B, tuned on a training split. The 4B model is optional and later.

**Splits.** Training, validation and test, fixed before any model is run. The test set is not opened until the final run. Record the split seed in the plan.

**Metrics.**
- Strict field-level precision and recall. A field counts as correct only when its value matches exactly after the normalisation written in the plan.
- Malformed output rate, meaning replies that fail the schema.
- Latency per record and peak memory.
- Cost per 10,000 records, computed from billed tokens for the API model and from measured compute time for the local model.

**Estimate, to be replaced by the pilot.** Assume 400 input tokens and 80 output tokens per record for the API model, which is a guess until the prompt is written. That gives about $0.40 for input and $0.40 for output per 10,000 records, about $0.80 in total. The LoRA training and inference cost is not estimated yet, because it depends on the hardware.

## Pilot (before any full run)

- Use 50 test records and one run per system, so cost stays under about $0.10.
- Confirm the API model returns valid output on its exact settings, with thinking off if that setting is used, and that no reply is cut off.
- Confirm the local model trains, saves, and reloads without error.
- Check that the labels are consistent. If two annotators disagree on more than a set share of records, stop and fix the labels before the full run.
- Show the pilot numbers to the author before spending on the full run.

## Stop rules

- Stop the full run if any reply hits the token limit, is empty, or has cost zero. Count such replies as errors, never as results.
- Stop if the pilot shows the task is too easy for the API model (near 100% on every field). Then the comparison cannot separate the two systems, and the task must be made harder.
- Hard cap: $5 total for the project, checked after every new spend. Ask before going over.

## Claims the article must not make

- No cost ratio such as "16 times" unless it comes from our own measured run.
- No claim that the small model is better in general. Report the one task and dataset.
- Report any error or dropped record in the main text.

## Outline

1. The problem: labels that repeat the same instructions on every record.
2. The annotation contract: the fields, the allowed values, and the labelling rules.
3. The two systems, with the prompt and the training setup.
4. The metrics, with a plain explanation of strict matching.
5. Results by field, with the error types.
6. Cost per 10,000 records, shown as a table with the price basis named.
7. Where the small model fails, and what it does not show.
8. A practical checklist for choosing between the two.

## Decisions made (2026-10-22)

- **Dataset (chosen 2026-10-22):** [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive), English (en-US) part. License CC BY 4.0, shown in the repository's LICENSE file (copyright Amazon.com Inc. or its affiliates). The card gives 11,514 train, 2,033 dev and 2,974 test utterances for en-US. Annotations are marked expert generated. Each row has a scenario, an intent, and slot annotations.
- **Task:** read one short utterance and return fixed fields as JSON. The fields are scenario, intent, and the list of slots with their values. This is voice assistant text and not support tickets, and the article must say so plainly in the opening.
- **How to load it:** the Hub copy uses a loading script that downloads a tarball from Amazon's S3 bucket (`amazon-massive-dataset-1.1.tar.gz`). Newer `datasets` versions do not run such scripts, so the Colab notebook downloads the tarball and reads the English JSON lines itself. Check the tarball's file layout when writing the notebook, and do not assume it.
- **Where the experiment runs (updated 2026-10-22):** the whole experiment runs in Google Colab on a paid plan, chosen by the author. That covers training, the local model runs, the timing, and the calls to the prompted API model. The API key goes in Colab Secrets and never in the notebook text. The author buys the plan, and the Mac stays free.
- **Training hardware (updated 2026-10-22):** Google Colab on a paid plan, chosen by the author. The author buys the plan. Prices and GPU access were not confirmed on Google's own page, and third party sources disagree, so check the pricing page before buying. The Mac stays as a fallback. Earlier Mac note: test the Mac first. The Mac is an Apple M4 with 16 GB of unified memory, 77 GB free disk, PyTorch 2.8 with the MPS backend. The LoRA package `peft` and Apple's MLX are not installed. If the test fails or is too slow, rent a hosted GPU instead.
- **Model sizes:** one small model in the first run. The 4B model is added later only if the first results justify it.

## Mac test result (2026-10-22, random token batches, no dataset)

| Model | Training speed | Peak memory | Verdict |
|---|---|---|---|
| Qwen3.5 0.8B (Apache 2.0) | First step 25 s, then over 130 s per step at batch 4 (1,024 tokens per step) | 13.6 GB | Too slow. Its newer attention layers fall back to slow Python code on a Mac. Run stopped at step 5. Log in `results/archive/`. |
| Qwen3 0.6B (Apache 2.0) | 2.8 s per step at batch 4, about 364 tokens per second | 9.1 GB of 16 GB | Works. 20 steps took under a minute after loading. |

For Qwen3 0.6B on this Mac, generating 80 tokens took 5.6 s per record one at a time and 0.59 s per record in batches of 8. First load took about 4 minutes because of the download.

**Estimates, not results.** For 3,000 tickets of about 256 tokens each, one pass is about 35 minutes, so three passes are about 1.7 hours. Scoring 10,000 records in batches of 8 would take about 1.6 hours. Real tickets may differ in length.

**Plan change.** The small model is now Qwen3 0.6B, and not Qwen3.5 0.8B. The article must say it is 0.6B. The Qwen3 0.6B license tag on its Hugging Face page is Apache 2.0.

## Dataset search (2026-10-22, licenses read from each Hugging Face dataset page or API)

| Dataset | License | Where labels come from | Verdict |
|---|---|---|---|
| [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) | CC BY 4.0 | Card says expert generated annotations, original source | Candidate. Real style customer queries, 10K to 100K rows. One field only (intent, 77 values). |
| [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive) | CC BY 4.0 | Card says expert generated annotations, original source | Candidate. Three kinds of annotation (scenario, intent, slots), which suits field level scoring. Text is voice assistant commands, not support tickets. |
| [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) | CC BY 3.0 | Expert generated | Candidate for a single field task. Older license version. |
| [cngchis/Support-Ticket-Router-12K-Cleaned](https://huggingface.co/datasets/cngchis/Support-Ticket-Router-12K-Cleaned) | Apache 2.0 | Machine generated with GPT-4 class models, per its own card | Rejected as the main set. Synthetic, only 6 classes, likely too easy, and the labels are not human. |
| [tasksource/help-desk-tickets](https://huggingface.co/datasets/tasksource/help-desk-tickets) | CC BY 4.0 | Real tickets, masked, from a Mendeley dataset | Rejected. Only 357 tickets with text and no category field to extract. The original Mendeley license page could not be read, because it needs a browser. |
| [lvogelFH/de-en-support-tickets](https://huggingface.co/datasets/lvogelFH/de-en-support-tickets) | Says CC BY 4.0 | Derived from Tobi-Bueck/customer-support-tickets, which is CC BY NC 4.0 | Rejected. The license conflicts with its source. TDS needs commercial use. |
| Tobi-Bueck/customer-support-tickets and many similar | CC BY NC 4.0 | | Rejected. Non commercial. |

Only a few support ticket datasets on the Hub have a permissive license, and most of those are synthetic. Real, human labelled, permissive ticket data with several fields was not found.

## Running on Colab from the terminal (set up 2026-10-22)

The completed setup, recovery steps, and failure history are documented in [COLAB_CLI_RUNBOOK.md](COLAB_CLI_RUNBOOK.md).

- The official Colab CLI (`google-colab-cli` 0.7.5) is installed in `~/.local/share/colab-cli-venv`, linked as `~/.local/bin/colab`. Google Cloud SDK is in `~/.local/share/google-cloud-sdk`, with a wrapper at `~/.local/bin/gcloud` that uses Python 3.12.
- Sign in used `gcloud auth application-default login` with four scopes (openid, cloud-platform, userinfo.email, colaboratory), approved by the author in the browser. Always pass `--auth=adc` to `colab`, because the CLI's real default is `oauth2`, not `adc` as its docs say.
- Smoke test passed: `colab --auth=adc run --gpu T4 colab_smoke_test.py` started a Tesla T4 (15 GB), ran for about 15 seconds, and shut down by itself. Balance was 500.00 compute units before and after, with no session left running.
- Rules for every run: use `colab run` for one off jobs, or `colab stop` as soon as a session ends. Check `colab --auth=adc usage` before and after, and report the change to the author.
- Anthropic key (decided 2026-10-22): the author chose to run everything on Colab with the existing project key. The key goes to the remote machine only by the tested upload method (temp file with umask 077, uploaded, local copy deleted, remote script reads it into memory and deletes the file). Never put the key in code passed to `colab exec`, because the CLI records executed code in `~/.config/colab-cli/history/` and `colab.log`. The $5 project cap applies.

## What the English data contains (inspected on Colab, 2026-10-22, `colab_scripts/inspect_data.py`)

- 11,514 train, 2,033 dev, 2,974 test rows, matching the card.
- 18 scenarios, 60 intents, 55 slot types.
- Texts are very short, 6.9 words on average, longest 35 words.
- 32.6% of training rows have no slots. Rows have 0.99 slots on average, at most 7.
- Every intent name starts with its scenario name, so scenario can be read off the intent. Scenario and intent are therefore not independent fields, and the article must say so.
- 3.1% of rows repeat a slot type inside one text, so slots are scored as a list of (type, value) pairs, not a dictionary.
- In every training row each slot value appears word for word in the text, so slots can be scored by exact string match.
- Because texts are so short, the prompted model's cost will be dominated by the instructions that list the labels, repeated on every request. That is the cost the article is about, and the pilot must measure it.
- Colab cost of the inspection: under 0.01 compute units. Balance 499.99.

## Pilot results (2026-10-22, 50 random dev rows, T4, files in `results/pilot/`)

This is a pipeline check, not a result. The test split was not opened.

| | Prompted Haiku 5.5 (3 examples) | LoRA Qwen3 0.6B (1,000 rows, 1 epoch, 32 steps) |
|---|---|---|
| Scenario accuracy | 0.88 | 0.74 |
| Intent accuracy | 0.78 | 0.52 |
| Slots exact | 0.58 | 0.24 |
| Whole record exact | 0.46 | 0.14 |
| Slot precision, recall, F1 | 0.63, 0.65, 0.64 | 0.34, 0.46, 0.39 |
| Malformed or cut off replies | 0 | 0 |
| Cost | $0.0062 for 50 rows, about $1.24 per 10,000, 1,052 input tokens per row | training 23.5 s, peak 12.6 GB on the GPU |
| Speed | median 0.98 s per call, one at a time | 28.4 rows per second in batches, 1.1 s per row one at a time, 1.6 GB on the GPU when running |

- The first Haiku run on the same 50 rows scored scenario 0.86 and the second 0.88, so the API model is not repeatable run to run. The final run needs repeats.
- Several gold labels are conventions of the dataset, for example short or odd phrases labelled general_quirky with no slots. Some prompt "errors" are these conventions.
- Failed attempt 1 (before the first fix): the connection dropped at the start of a command and the CLI waited. The driver now has timeouts and retries. Failed attempt 2: Colab's old `torchao` package blocked the LoRA library, so the driver removes it. Both stopped cleanly. Compute units used by the failed attempts and the pilot: about 0.2.
- Balance after the pilot: 599.79 compute units (the author added a Colab Pro purchase of 100 units).
- Cost basis idea, not final: a T4 session showed a burn rate of 1.07 units per hour. The price per unit must come from what the author paid.
- Open: whether the 1,052 token instruction block can be prompt cached. The cache read price is $0.01 per million tokens, but a minimum prompt length may apply. Check the caching page before claiming a cached cost.

## Prompt comparison on dev (2026-10-22, 300 random dev rows, seed 1, test split closed; files in `results/prompts/`)

Selection rule, written before the run: highest whole record accuracy wins, and a tie goes to the cheaper prompt.

| Prompt | Examples | Scenario | Intent | Slots exact | Whole record | Slot F1 | Unparseable replies | Input tokens per row | Cost for 300 rows |
|---|---|---|---|---|---|---|---|---|---|
| base | 3 | 0.870 | 0.810 | 0.523 | **0.447** | 0.583 | 3 | 1,052 | $0.037 |
| s1 | 18 (one per scenario) | 0.847 | 0.797 | 0.513 | 0.427 | 0.600 | 7 | 1,852 | $0.062 |
| s2 | 36 (two per scenario) | 0.827 | 0.780 | 0.513 | 0.433 | 0.587 | 9 | 2,835 | $0.091 |

- **Result:** more examples did not help and cost up to 2.4 times more. The base prompt is frozen for the final run. Differences of 2 points are within the noise of 300 rows and one run each.
- **Unparseable replies:** the prompted model sometimes answers garbled text such as "lets solve the puzzle" with a chat reply and no JSON. This is a real failure mode of prompting that a trained model does not have, and it counts as wrong in the score.
- **Scoring rule fixed after seeing results (disclose in the article):** the first parser rejected a reply that held a valid JSON object followed by a note with braces in it. The parser now reads the first complete JSON object. It applies to both systems, and the saved raw replies were rescored. The ranking did not change.
- **Other changes after seeing results:** the token limit for replies was raised from 200 to 400, because one earlier run had one empty or cut off reply and was stopped. The rerun had none. The driver was rebuilt to start jobs in the background after a dropped connection queued a duplicate job. About 8 US cents of API calls were wasted on the duplicate and the aborted run.
- **Cached cost, estimate only:** the instruction block is about 1,050 tokens, above the 512 token minimum for Haiku 5.5 (Anthropic prompt caching page, read 2026-10-22). Cache reads cost $0.01 per million tokens. If the block is cached, input cost per row could fall from about $0.000105 to about $0.000015, so the total might fall from about $1.24 to about $0.34 per 10,000 rows. This is arithmetic, not a measurement. The final run must measure it with real cache usage fields.

## Next steps

The full run and file audit are complete. See [RESULTS.md](RESULTS.md) for the held-out results, cost accounting, and limitations. Colab was stopped after the run.

1. Write the Colab notebook in pieces: data download and parsing, prompt and schema, the API model run, the LoRA training, the local model run, strict scoring, and the cost table.
2. Run a 50 record pilot in Colab, and show the pilot numbers before any full run.
3. Freeze the splits, the schema, the scoring rules and the prompt in a manifest before the full run.
4. Full run, audit, and then the article draft under the guidelines.

## Open decisions (need the author)

- None right now. The Colab plan is bought by the author. Check Google's pricing page first.

## Label definition test (registered 2026-10-23, before any run)

Question: how much of the 22 point gap does a prompt close when every intent and slot type carries a one line meaning?

- **What changes:** only the label lists in the Haiku 5.5 prompt (`colab_scripts/definitions.py`, `run_api_defs.py`). Model, settings (thinking off, 400 reply tokens), the same three examples, the parser and the scoring are the frozen ones.
- **How the definitions were written:** from six sample values per slot type and three sample texts per intent, all from the TRAINING split. No additional dev or test row was read to draft version 1, but the decision to try definitions followed analysis of the original test errors. The article must say both.
- **Version 1 prompt hash:** 0f6c5ac43de378a0a9257125850ef599b9d8ab447be019a8fa3eb7b2ed67b4b0.
- **Dev check:** the same 300 dev rows as the earlier prompt comparison (seed 1), one uncached run. The base prompt scored 0.447 whole record on these rows.
- **Revisions:** at most two, each written from the dev errors only, each version and its dev score recorded here. No new test row is inspected during these revisions; the original test split had already been opened. Rule: the version with the highest dev whole record accuracy is frozen, and a tie goes to the earlier version.
- **Test:** after freezing, three passes on the 2,974 test records, run with prompt caching to cut cost. Caching did not change accuracy in the earlier single cached pass (47.88 against 47.78 to 47.92), and the article must state that this is the only check.
- **Reporting:** the result goes in the article whatever it is, with the dev scores of all versions tried.
- **Budget:** Anthropic spend cap for this test is 2 US dollars. Stop and ask if it is reached.

### Definition test log
- **v1 dev (300 rows, uncached):** whole record 0.520 against 0.447 for the base prompt on the same rows. Scenario 0.840, intent 0.803, slots exact 0.607, slot F1 0.658. 2 unparseable replies. Input 2,481 tokens per row, cost 0.08 US dollars. Files `results/defs/defs_v1_dev*`.
- **What v1 got wrong:** most remaining slot errors look like inconsistent human labels (for example the same word labeled differently in similar texts). One pattern is systematic: 6 texts whose label is `qa_definition` went to `general_quirky`.
- **v2 (revision 1 of 2):** only the `general_quirky` line changes, to say that word meaning questions go to `qa_definition` and fact questions to `qa_factoid`.
- **v2 dev (300 rows, uncached):** whole record 0.533, intent 0.830, slots exact 0.610, slot F1 0.662, 1 unparseable reply, cost 0.08 US dollars. Higher than v1 (0.520), so v2 is frozen. No further revision, although one was allowed.
- **Frozen on 2026-10-23, before any definition-prompt test run:** prompt sha256 3150ce689c93ccbc96eacc4d8efd84f3adca117e7dcc51e56a68b4893fd1b5eb. Script hashes in `results/defs/FROZEN_DEFS_scripts.txt`. Next, three cached test passes.
- **Completed follow-up test:** three cached passes on all 2,974 test records scored 0.5672, 0.5636, and 0.5689 whole record (mean 0.5666). All had the frozen prompt hash, matching IDs and gold labels, and zero transport or cutoff errors. Recorded API usage was 0.408356 US dollars for the three passes, excluding three cache-priming requests. The Colab session was stopped. See `RESULTS.md` for the full comparison and limits.
- **Interpretation:** definitions improved the original Haiku mean by 8.80 points and left a 13.69-point gap to the LoRA mean. The idea and revision arose after the original test set was opened, so report this as exploratory follow-up, not a second pristine held-out test.
