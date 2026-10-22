# Colab CLI runbook for this experiment

Recorded 2026-10-22 from the completed MASSIVE annotation run. This documents the setup on this Mac and the changes that let the run finish. The official CLI is `google-colab-cli` 0.7.5; commands and behavior may change with later versions.

## Local setup and authentication

1. The CLI lives in an isolated Python 3.12 environment at `~/.local/share/colab-cli-venv`; `~/.local/bin/colab` links to it. Google Cloud SDK lives at `~/.local/share/google-cloud-sdk`. The `~/.local/bin/gcloud` wrapper sets `CLOUDSDK_PYTHON` to `~/.local/bin/python3.12`, because the SDK installer hit an error using the Mac's older Python 3.9.
2. The author approved Google sign-in in the browser. The working command was entered as **one shell line**:

   ```sh
   gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
   ```

   A previous paste split this command before `--scopes`, so Google saved credentials without the Colab scope. The login was repeated correctly. Do not read or print `~/.config/gcloud/application_default_credentials.json`.
3. Every Colab call in this project explicitly uses `--auth=adc`, before the subcommand. The CLI's observed default was a separate OAuth flow. Check access with `colab --auth=adc whoami`, and redact the email when sharing output.
4. Before any paid run, check `colab --auth=adc usage` and `colab --auth=adc sessions`. Afterward, check both again. The smoke test used `colab --auth=adc run --gpu T4 colab_smoke_test.py`; `run` created, executed, and stopped its short-lived session. The full experiment used an explicit session because it had several dependent jobs.

## The full-run workflow

The controller is [run_on_colab.py](run_on_colab.py). It uses session name `exp` and a T4. Its `full` mode checks [FROZEN_FULL.json](FROZEN_FULL.json), then skips steps whose local summary or adapter file already exists. It installs `anthropic` and `peft`, removes Colab's preinstalled incompatible `torchao` 0.10.0, and uploads the five experiment scripts. Run it from this folder with `~/.local/bin` on `PATH`:

```sh
PATH="$HOME/.local/bin:$PATH" python3 run_on_colab.py full
```

The full run is already complete. This command is for recovery or a deliberately new run; inspect `results/full/` first to avoid unnecessary GPU time or API calls. The script's `pilot` and `prompts` modes were used earlier for the 50-record pipeline check and development prompt selection.

Each long operation starts once on the remote kernel, writing `/content/started_<step>`, `/content/log_<step>.txt`, and `/content/status_<step>.json`. The local controller polls the status file through `colab download`, which still worked while the kernel was busy. Retrying a **poll** is safe. Blindly repeating the Python job after a connection error caused a duplicate API run in an earlier version, so the start marker now prevents a second launch in the same session. The controller also has command timeouts and checks whether the session still exists. A small `colab exec` call every two minutes coincided with a successful long run, but it has not been proven to prevent Colab from reclaiming a machine.

After each training seed, the driver downloads `adapter_model.safetensors` to `results/full/adapters/s<seed>/`. After scoring, it downloads predictions and summary immediately. A later session can upload a saved adapter to a flat `/content/adapter_in_<seed>.safetensors` path and rebuild its small config, then resume without retraining. `colab upload` did **not** create a missing remote folder in this run; the earlier nested upload failed. On success, the driver saves extra remote logs under `results/rescued/` and stops the session. On a `full`-mode failure, it deliberately **keeps** the session so files can be recovered; check `colab --auth=adc sessions` and stop it explicitly when recovery is done. The top docstring of `run_on_colab.py` predates this behavior and incorrectly says every error stops the session.

## API key handling

The author chose to run the Anthropic calls on Colab using the existing project key. The controller reads the key locally, writes a temporary file with mode `0600`, uploads that file, and removes the local copy immediately. A remote command reads it into `os.environ` and deletes the remote file. The key therefore exists briefly in files and remains accessible to code in that remote kernel until the session stops. Keep that session private and stop it when finished.

Do **not** put a key literal in code passed to `colab exec` or on its command line. A dummy-key experiment showed that the CLI writes executed code in plain text under `~/.config/colab-cli/history/` and `colab.log`. The dummy trace was removed. The upload method was then checked with a second dummy value; its content did not appear in those logs. Do not print the credential file, the project key file, or remote environment variables. The driver redacts Anthropic-shaped keys from its printed output, but redaction is a backup, not the primary control.

## Failures and what changed

| Observed failure | Change made | Evidence or remaining limit |
|---|---|---|
| A dropped connection led the first driver to queue a duplicate API job. | Remote start marker plus status-file polling; local timeouts. | The duplicate spent about 8 US cents before it was stopped. A marker only protects one live session. |
| Colab's `torchao` 0.10.0 conflicted with `peft`. | Uninstall `torchao` during session setup; this project does not use it. | The pilot then trained successfully. |
| A full batch of 32 ran out of the T4's 14.56 GiB GPU memory. | Keep effective batch 32 but compute it in microbatches of 8 with gradient accumulation. | Seeds 1 and 2 trained through 1,080 steps with 7.42 GB peak allocated GPU memory. |
| A driver crash stopped the session and lost the first trained adapter. | Save adapter weights locally immediately; keep the session on a full-run failure. | Saved adapters later survived restarts. Seed 0's small training metadata file was lost, though its weights and predictions remain. |
| A nested restore upload path was assumed to exist remotely. | Upload to a flat `/content/` path and check the upload return code. | Adapter restore and dev sanity check succeeded. |
| Colab sessions disappeared after long jobs. | Resume from local files; add periodic activity and faster lost-session checks. | The exact reason Colab reclaimed those machines is unknown. The final run succeeded; that does not prove which change prevented another loss. |

The frozen model and scoring scripts were not altered to handle the connection failures. Operational changes to the controller and the manifest refreezes are recorded in [RESULTS.md](RESULTS.md) and `FROZEN_FULL.json`.

## Additional GPT-6 Sol pass

The later GPT-6 Sol comparison used [run_gpt6_sol_colab.py](run_gpt6_sol_colab.py) and a **CPU** Colab session named `gpt6sol`. It reused the frozen prompt and the same 2,974 test items. The controller staged the existing OpenAI key with the same restricted temporary-file upload pattern, installed the `openai` package, launched a background job once, and downloaded the growing JSONL every 20 seconds. It stored the prompt and input hashes, model settings, pricing assumptions, pilot result, and spending cap in `results/gpt6_sol/GPT6_RUN_MANIFEST.json`. A 50-record development pilot cost $0.0764; the complete test pass recorded $4.5635 in API usage. Both sessions stopped, and `colab --auth=adc sessions` reported no active sessions afterward. The balance moved from 596.56 to 596.55 compute units across this extra work.

The first two pilot launches stopped before scoring because the controller assumed `colab exec` always prints a final line. In this case a successful launch returned an empty output string. The controller now accepts that response, while the remote status file and complete local result count establish whether the job actually finished. Those attempts made no recorded API calls. The full pass saved all 2,974 distinct results locally and had no API errors or incomplete replies.

The later Haiku label-definition check used [run_defs_on_colab.py](run_defs_on_colab.py) on a Colab CPU session. It staged the Anthropic key through the restricted temporary-file upload, ran two 300-record development versions, then three cached passes with the frozen second version. Each pass saved its JSONL and summary to `results/defs/`. The final `colab --auth=adc sessions` check showed no active sessions and a zero compute-unit usage rate. The three test summaries recorded 0.408356 US dollars in API usage, excluding one cache-priming request per pass.

## Completion and cost checks

The final controller log ended with `finished` and `session stopped`. `colab --auth=adc sessions` then reported no active sessions. The successful resumed run took the Colab balance from 597.89 to 596.56 compute units, including session setup and all remaining work. Recorded Anthropic usage across its saved full-run summaries totaled $1.22002, below the $2 manifest cap. The cached API tally omits one cache-priming request, whose exact charge was not captured.

For a future run, confirm that each result JSONL has the expected number of unique IDs and matches its summary, compare IDs and gold labels across systems, verify frozen script hashes, and confirm the final Colab usage rate is zero. This run passed those checks; details are in [RESULTS.md](RESULTS.md).
