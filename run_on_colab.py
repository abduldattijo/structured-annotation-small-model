"""Runs the experiment steps on a Colab GPU through the Colab CLI, with timeouts and safe retries.

    python3 run_on_colab.py pilot

The Anthropic key reaches the remote machine only as an uploaded temp file that is deleted at once.
It is never printed, never put in code, and never passed on a command line. All output is scrubbed.
The session is always stopped at the end, even after an error.
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
S = "exp"
SCRUB = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")


def c(*args, stdin=None, timeout=300):
    try:
        r = subprocess.run(["colab", "--auth=adc", *args], input=stdin, text=True, capture_output=True,
                           timeout=timeout, cwd=HERE)
        return r.returncode, SCRUB.sub("<key hidden>", (r.stdout or "") + (r.stderr or ""))
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout} s"


def say(msg):
    print(SCRUB.sub("<key hidden>", msg), flush=True)


def tail(text, n=25):
    return "\n".join(text.strip().splitlines()[-n:])


def exec_code(code, timeout, retries=2):
    """Runs code in the kernel. Retries only when the connection failed before the code could start."""
    for attempt in range(retries + 1):
        rc, out = c("exec", "-s", S, "--timeout", str(timeout), stdin=code, timeout=timeout + 60)
        lost = "Connection was lost" in out or rc == 124
        if not lost:
            return rc, out
        say(f"  connection problem (attempt {attempt + 1}). Waiting, then retrying.")
        time.sleep(15)
        c("status", "-s", S)
    return 1, out


def stage_key():
    """Writes ANTHROPIC_API_KEY from your environment to a private temp file for upload. Never printed."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set ANTHROPIC_API_KEY in your environment first")
    path = f"/tmp/.expkey_{os.getpid()}"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, key.encode())
    os.close(fd)
    return path


def setup(gpu="T4", packages=("anthropic", "peft")):
    say("balance before: " + c("usage")[1].splitlines()[0])
    rc, out = c("new", "-s", S, "--gpu", gpu, timeout=400)
    say(tail(out, 2))
    if rc:
        raise SystemExit("could not start the session")
    say(tail(c("install", "-s", S, *packages, timeout=400)[1], 1))
    # Colab ships an old torchao that the LoRA library rejects. We do not use it, so remove it.
    rc, out = exec_code("import subprocess, sys\nr = subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], "
                        "capture_output=True, text=True)\nprint((r.stdout or r.stderr).strip().splitlines()[-1:])\n", 180)
    say("  torchao: " + tail(out, 1))
    for f in ("massive_data", "schema", "run_api", "train_lora", "run_local"):
        c("upload", "-s", S, f"colab_scripts/{f}.py", f"/content/{f}.py")
    path = stage_key()
    try:
        say("  " + tail(c("upload", "-s", S, path, "/content/.k")[1], 1).replace(path, "<temp file>"))
    finally:
        os.remove(path)
    rc, out = exec_code(
        "import os\nv=open('/content/.k').read().strip()\nos.environ['ANTHROPIC_API_KEY']=v\nos.remove('/content/.k')\n"
        "print('key in memory only | file removed:', not os.path.exists('/content/.k'), '| length ok:', len(v) > 20)\n"
        "import transformers, peft, torch\n"
        "print('transformers', transformers.__version__, '| peft', peft.__version__, '| torch', torch.__version__)\n", 120)
    say(tail(out, 3))
    if rc:
        raise SystemExit("key step failed")


def fetch(name):
    """Downloads a small remote file with the file API, which works even while the kernel is busy."""
    local = f"/tmp/_colab_{os.getpid()}_{name}"
    for attempt in range(4):
        rc, _ = c("download", "-s", S, f"/content/{name}", local, timeout=120)
        if rc == 0 and os.path.exists(local):
            text = open(local, errors="ignore").read()
            os.remove(local)
            return text
        time.sleep(5 * (attempt + 1))
    return None


def step(title, code, max_wait):
    """Starts the job in a background thread on the remote machine, then polls a status file.

    A dropped connection can neither lose the job nor start it twice: the job writes a marker file
    when it starts and skips if the marker already exists.
    """
    import json
    import textwrap
    name = re.sub(r"\W+", "_", title).strip("_")
    body = textwrap.indent(code, " " * 16)
    wrapper = (
        "import os, sys, json, threading, traceback, contextlib\n"
        "sys.path.insert(0, '/content')\n"
        f"if os.path.exists('/content/started_{name}'):\n"
        f"    print('already started: {name}')\n"
        "else:\n"
        f"    open('/content/started_{name}', 'w').write('1')\n"
        "    def _job():\n"
        f"        with open('/content/log_{name}.txt', 'w', buffering=1) as _log, contextlib.redirect_stdout(_log), contextlib.redirect_stderr(_log):\n"
        "            try:\n"
        + body
        + "\n"
        f"                json.dump({{'ok': True}}, open('/content/status_{name}.json', 'w'))\n"
        "            except BaseException:\n"
        f"                json.dump({{'ok': False, 'error': traceback.format_exc()[-3000:]}}, open('/content/status_{name}.json', 'w'))\n"
        "    threading.Thread(target=_job, daemon=True).start()\n"
        f"    print('started: {name}')\n"
    )
    say(f"=== {title}")
    rc, out = exec_code(wrapper, 120)
    say("  " + tail(out, 1))
    t0 = time.time()
    status = None
    polls = 0
    while time.time() - t0 < max_wait:
        time.sleep(15)
        polls += 1
        text = fetch(f"status_{name}.json")
        if text:
            status = json.loads(text)
            break
        if polls % 8 == 0:        # every two minutes: a tiny command keeps the session active and shows if it was lost
            rc, out = c("exec", "-s", S, "--timeout", "30", stdin="print('alive')", timeout=90)
            if "not found" in out or "No active" in out:
                raise SystemExit(f"the Colab session was lost during: {title}. Rerun the driver to resume.")
    log = fetch(f"log_{name}.txt") or ""
    say(tail(re.sub(r"\x1b\[[0-9;]*m", "", log), 30))
    if status is None:
        raise SystemExit(f"step did not finish within {max_wait} s: {title}")
    if not status.get("ok"):
        say(SCRUB.sub("<key hidden>", status.get("error", "")))
        raise SystemExit(f"step failed: {title}")
    say(f"  finished in {time.time() - t0:.0f} s")


def download(files, outdir):
    os.makedirs(os.path.join(HERE, outdir), exist_ok=True)
    for f in files:
        say("  " + tail(c("download", "-s", S, f"/content/{f}", f"{outdir}/{f}")[1], 1))


def rescue():
    """Saves any result files from the remote machine before it is stopped, even after a failure."""
    rc, out = c("ls", "-s", S, "/content", timeout=60)
    if rc:
        return
    names = [l.strip() for l in out.splitlines() if re.search(r"\.(jsonl|json|txt)$", l.strip())]
    names = [n for n in names if not n.startswith("started_")]
    if names:
        os.makedirs(os.path.join(HERE, "results/rescued"), exist_ok=True)
        for n in names:
            c("download", "-s", S, f"/content/{n}", f"results/rescued/{n}", timeout=120)
        say(f"saved {len(names)} files from the remote machine to results/rescued/")


def pilot():
    setup()
    step("1. prompted model, 50 dev rows",
         "import run_api; run_api.main(split='dev', n=50, seed=0, out='/content/api_pilot.jsonl', max_usd=0.25)", 900)
    step("2. LoRA training, 1000 rows, 1 epoch",
         "import train_lora; train_lora.main(n_train=1000, epochs=1, out_dir='/content/adapter_pilot')", 1500)
    step("3. tuned model on the same 50 dev rows",
         "import run_local; run_local.main(adapter='/content/adapter_pilot', split='dev', n=50, out='/content/local_pilot.jsonl')", 1500)
    download(["api_pilot.jsonl", "api_pilot_summary.json", "adapter_pilot_train_info.json",
              "local_pilot.jsonl", "local_pilot_summary.json"], "results/pilot")


def prompts():
    """Compares three prompt versions on 300 dev rows. The test split stays closed.
    Selection rule, written before the run: highest whole record accuracy wins, a tie goes to the cheaper prompt."""
    setup()
    for v in ("base", "s1", "s2"):
        step(f"prompt variant {v}, 300 dev rows",
             f"import run_api; run_api.main(split='dev', n=300, seed=1, out='/content/prompt_{v}.jsonl', max_usd=0.6, workers=4, variant='{v}')", 1500)
    download([f"prompt_{v}{x}" for v in ("base", "s1", "s2") for x in (".jsonl", "_summary.json")], "results/prompts")


def save_adapter(seed):
    """Copies the trained adapter weights (about 40 MB) to this Mac. The small config file is rebuilt
    from the frozen settings when needed, so only the weights have to be saved."""
    out = f"results/full/adapters/s{seed}"
    os.makedirs(os.path.join(HERE, out), exist_ok=True)
    for attempt in range(4):
        rc, _ = c("download", "-s", S, f"/content/adapter_s{seed}/adapter_model.safetensors",
                  f"{out}/adapter_model.safetensors", timeout=300)
        if rc == 0:
            say(f"  adapter weights for seed {seed} saved to {out}")
            return
        time.sleep(10)
    raise SystemExit(f"could not save the adapter weights for seed {seed}")


RESTORE = """
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
_base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32)
_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules='all-linear', task_type='CAUSAL_LM')
_m = get_peft_model(_base, _cfg)
_res = set_peft_model_state_dict(_m, load_file('/content/adapter_in_SEED.safetensors'))
print('unexpected keys:', len(getattr(_res, 'unexpected_keys', []) or []))
_m.save_pretrained('/content/adapter_sSEED')
del _m, _base
torch.cuda.empty_cache()
"""


def check_frozen():
    """Aborts if any frozen script changed since FROZEN_FULL.json was written."""
    import hashlib
    import json
    m = json.load(open(os.path.join(HERE, "FROZEN_FULL.json")))

    def sha(path):
        return hashlib.sha256(open(os.path.join(HERE, path), "rb").read()).hexdigest()

    for name, h in m["scripts"].items():
        if sha(f"colab_scripts/{name}.py") != h:
            raise SystemExit(f"frozen file changed: colab_scripts/{name}.py. Stopping.")
    return m


def full():
    """The full run. Needs FROZEN_FULL.json. The test split is scored once per system, with no tuning.

    Resumable: every step saves its results to results/full/ at once, and a rerun skips finished steps.
    """
    import json
    m = check_frozen()
    psha = m["prompt_sha256"]
    cap = m["spend_cap_usd"]
    outdir = os.path.join(HERE, "results/full")
    os.makedirs(outdir, exist_ok=True)

    def have(name):
        return os.path.exists(os.path.join(outdir, name))

    def api_spent():
        total = 0.0
        for nme in ("api_test_run1", "api_test_run2", "api_test_run3", "api_test_cached", "api_latency_uncached", "api_latency_cached"):
            if have(f"{nme}_summary.json"):
                total += json.load(open(os.path.join(outdir, f"{nme}_summary.json")))["summary"]["cost_usd"]
        return total

    state = {"up": False, "remote": set()}

    def ensure_session():
        if not state["up"]:
            setup()
            state["up"] = True

    def ensure_adapter_remote(seed):
        ensure_session()
        if seed in state["remote"]:
            return
        local = f"results/full/adapters/s{seed}/adapter_model.safetensors"
        rc, out = c("upload", "-s", S, local, f"/content/adapter_in_{seed}.safetensors", timeout=400)
        if rc:
            raise SystemExit("could not upload the saved adapter: " + tail(out, 2))
        step(f"restore adapter seed {seed}", RESTORE.replace("SEED", str(seed)), 900)
        state["remote"].add(seed)

    def api_step(name, split, n, workers, cache, seed=0, max_usd=0.6):
        if have(f"{name}_summary.json"):
            say(f"skip {name}: already done")
            return
        if api_spent() >= cap:
            raise SystemExit("spend cap reached")
        ensure_session()
        code = (f"import run_api; run_api.main(split='{split}', n={n}, seed={seed}, out='/content/{name}.jsonl', "
                f"max_usd={max_usd}, workers={workers}, variant='base', cache={cache}, expected_prompt_sha='{psha}')")
        step(f"api {name}", code, 2400)
        download([f"{name}.jsonl", f"{name}_summary.json"], "results/full")
        say(f"  API spend so far: ${api_spent():.3f} of ${cap:.2f}")

    for seed in m["small_model"]["seeds"]:
        if not have(f"adapters/s{seed}/adapter_model.safetensors"):
            ensure_session()
            step(f"train seed {seed}",
                 f"import train_lora; train_lora.main(n_train=None, epochs=3, batch=32, lr=2e-4, seed={seed}, out_dir='/content/adapter_s{seed}')", 3600)
            save_adapter(seed)
            download([f"adapter_s{seed}_train_info.json"], "results/full")
            state["remote"].add(seed)
        if seed == 0 and not have("local_dev_s0_summary.json"):
            ensure_adapter_remote(0)
            step("dev sanity check seed 0",
                 "import run_local; run_local.main(adapter='/content/adapter_s0', split='dev', n=300, seed=1, out='/content/local_dev_s0.jsonl', latency_rows=10)", 1500)
            download(["local_dev_s0.jsonl", "local_dev_s0_summary.json"], "results/full")
        if seed == 0:
            dev = json.load(open(os.path.join(outdir, "local_dev_s0_summary.json")))
            say(f"  dev whole record accuracy {dev['whole_record_accuracy']:.3f}")
            if dev["whole_record_accuracy"] < 0.30:
                raise SystemExit("the trained model looks broken on dev. Stopping to investigate before touching the test split.")
        if not have(f"local_test_s{seed}_summary.json"):
            ensure_adapter_remote(seed)
            step(f"local test seed {seed}",
                 f"import run_local; run_local.main(adapter='/content/adapter_s{seed}', split='test', n=None, out='/content/local_test_s{seed}.jsonl', latency_rows=100)", 2400)
            download([f"local_test_s{seed}.jsonl", f"local_test_s{seed}_summary.json"], "results/full")
    for run in (1, 2, 3):
        api_step(f"api_test_run{run}", "test", None, 6, False)
    api_step("api_test_cached", "test", None, 6, True)
    api_step("api_latency_uncached", "dev", 100, 1, False, seed=2, max_usd=0.1)
    api_step("api_latency_cached", "dev", 100, 1, True, seed=2, max_usd=0.1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    failed = False
    try:
        {"pilot": pilot, "prompts": prompts, "full": full}[what]()
        say("finished")
    except BaseException:
        failed = True
        raise
    finally:
        rescue()
        if failed and what == "full":
            say("FAILED. The session is KEPT so nothing is lost. Stop it with: colab --auth=adc stop -s exp. " + c("usage")[1].splitlines()[0])
        else:
            c("stop", "-s", S)
            say("session stopped. " + c("usage")[1].splitlines()[0])
