"""Freezes the full run: hashes of every script, the exact prompt, seeds, and settings. Writes FROZEN_FULL.json.
Refuses to overwrite a frozen file whose contents would change.

    python3 freeze_full.py
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "FROZEN_FULL.json")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


prompt = json.load(open(os.path.join(HERE, "results/prompts/prompt_base_summary.json")))["system_prompt"]
manifest = {
    "frozen_on": "2026-10-22",
    "scripts": {f: sha(os.path.join(HERE, "colab_scripts", f"{f}.py"))
                for f in ("massive_data", "schema", "run_api", "train_lora", "run_local")},
    "driver": sha(os.path.join(HERE, "run_on_colab.py")),
    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    "prompt_selection_rule": "highest whole record accuracy on 300 dev rows (seed 1); a tie goes to the cheaper prompt. Winner: base (3 examples)",
    "data": "MASSIVE 1.1 en-US, train 11514, dev 2033, test 2974, from amazon-massive-dataset-1.1.tar.gz",
    "prompted_model": {"id": "claude-haiku-5-5", "thinking": "disabled", "max_tokens": 400, "price_in_per_mtok": 0.10,
                       "price_out_per_mtok": 0.50, "cache_read_per_mtok": 0.01, "cache_write_5m_per_mtok": 0.125,
                       "uncached_test_runs": 3, "cached_test_runs": 1, "workers": 6},
    "small_model": {"id": "Qwen/Qwen3-0.6B", "method": "LoRA, rank 16, alpha 32, dropout 0.05, all linear layers",
                    "train_rows": 11514, "epochs": 3, "batch": 32, "micro_batch": 8, "lr": 2e-4, "seeds": [0, 1, 2],
                    "precision": "float32 weights with float16 autocast for training, float16 for serving",
                    "decoding": "greedy, 96 new tokens at most", "serving_batch": 64},
    "dev_sanity_check": "300 dev rows after training seed 0. Only to catch a broken run. No tuning on the test split.",
    "spend_cap_usd": 2.0,
    "test_split": "opened once per system for scoring. No change to any setting after seeing test results",
    "refrozen": "2026-10-22, before any test result. (1) after the first full training attempt ran out of GPU memory, batches of 32 are computed in pieces of 8, with the same optimizer steps. (2) after a driver crash lost a trained adapter, the driver retries downloads, saves adapters locally, and keeps the session on failure. (3) after Colab reclaimed the machine right after training, the driver became resumable and restores a saved adapter into a fresh machine. (4) after Colab reclaimed the machine again, the driver sends a keep alive command every two minutes and stops at once if the session is lost. No model, prompt, data, or scoring change.",
}
new = json.dumps(manifest, indent=1, sort_keys=True)
if os.path.exists(OUT) and open(OUT).read() != new:
    sys.exit("FROZEN_FULL.json already exists with different contents. Refusing to overwrite.")
open(OUT, "w").write(new)
print("frozen. prompt sha256 =", manifest["prompt_sha256"][:16], "...")
