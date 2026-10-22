"""Drive one resumable GPT-6 Sol comparison through the official Colab CLI."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
OUT = ROOT / "results" / "gpt6_sol"
SESSION = "gpt6sol"
REDACT = re.compile(r"sk-[A-Za-z0-9_-]{15,}")
PROMPT_SHA = json.loads((ROOT / "FROZEN_FULL.json").read_text())["prompt_sha256"]


def cli(*args, code=None, timeout=180):
    try:
        run = subprocess.run([str(Path.home() / ".local/bin/colab"), "--auth=adc", *args],
                             input=code, text=True, capture_output=True, timeout=timeout, cwd=ROOT)
        return run.returncode, REDACT.sub("<key hidden>", (run.stdout or "") + (run.stderr or ""))
    except subprocess.TimeoutExpired:
        return 124, f"Colab command timed out after {timeout} seconds"


def log(message):
    print(REDACT.sub("<key hidden>", str(message)), flush=True)


def prepare_inputs():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads((ROOT / "results/full/api_test_run1_summary.json").read_text())
    system = source["system_prompt"]
    assert hashlib.sha256(system.encode()).hexdigest() == PROMPT_SHA
    prompt_path = OUT / "system_prompt.txt"
    prompt_path.write_text(system, encoding="utf-8")
    for name, src, count in (
        ("pilot", ROOT / "results/pilot/api_pilot.jsonl", 50),
        ("full", ROOT / "results/full/api_test_run1.jsonl", 2974),
    ):
        rows = []
        for line in src.open(encoding="utf-8"):
            row = json.loads(line)
            rows.append({k: row[k] for k in ("id", "text", "gold")})
        assert len(rows) == count and len({r["id"] for r in rows}) == count
        (OUT / f"{name}_inputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def stage_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if len(key) < 20:
        raise RuntimeError("Set OPENAI_API_KEY in your environment first")
    fd, path = tempfile.mkstemp(prefix=".gpt6key_")
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key)
    return path


def checked(*args, code=None, timeout=180):
    rc, text = cli(*args, code=code, timeout=timeout)
    if rc:
        raise RuntimeError(f"colab {' '.join(args[:2])} failed: {text[-450:]}")
    return text


def setup():
    log(checked("usage").splitlines()[0])
    log(checked("new", "-s", SESSION, timeout=400).strip().splitlines()[-1])
    checked("install", "-s", SESSION, "openai", timeout=400)
    for local, remote in (
        (ROOT / "colab_scripts/gpt6_sol_api.py", "/content/gpt6_sol_api.py"),
        (OUT / "system_prompt.txt", "/content/system_prompt.txt"),
        (OUT / "pilot_inputs.jsonl", "/content/pilot_inputs.jsonl"),
        (OUT / "full_inputs.jsonl", "/content/full_inputs.jsonl"),
    ):
        checked("upload", "-s", SESSION, str(local), remote, timeout=300)
    key_path = stage_key()
    try:
        checked("upload", "-s", SESSION, key_path, "/content/.openai_key", timeout=120)
    finally:
        os.remove(key_path)
    code = ("import os\n"
            "os.environ['OPENAI_API_KEY'] = open('/content/.openai_key').read().strip()\n"
            "os.remove('/content/.openai_key')\n"
            "print('key ready; remote file removed:', not os.path.exists('/content/.openai_key'))\n")
    launch_output = checked("exec", "-s", SESSION, "--timeout", "60", code=code, timeout=120)
    log(launch_output.strip().splitlines()[-1] if launch_output.strip() else "Key transfer accepted")


def snapshot(name):
    remote = f"/content/gpt6_{name}.jsonl"
    temp = OUT / f".{name}_download.jsonl"
    rc, _ = cli("download", "-s", SESSION, remote, str(temp), timeout=90)
    if rc or not temp.exists():
        return 0
    good = []
    for line in temp.open(encoding="utf-8", errors="ignore"):
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            break
    temp.unlink(missing_ok=True)
    if good:
        local = OUT / f"{name}.jsonl"
        staged = OUT / f".{name}_saved.jsonl"
        staged.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in good), encoding="utf-8")
        staged.replace(local)
    return len(good)


def run_job(name, max_usd):
    local = OUT / f"{name}.jsonl"
    if local.exists():
        checked("upload", "-s", SESSION, str(local), f"/content/gpt6_{name}.jsonl", timeout=300)
    code = (
        "import os, sys, json, threading, traceback, contextlib\n"
        "sys.path.insert(0, '/content')\n"
        f"if os.path.exists('/content/started_gpt6_{name}'):\n"
        " print('already started')\n"
        "else:\n"
        f" open('/content/started_gpt6_{name}', 'w').write('1')\n"
        " def job():\n"
        f"  with open('/content/log_gpt6_{name}.txt', 'w', buffering=1) as lg, contextlib.redirect_stdout(lg), contextlib.redirect_stderr(lg):\n"
        "   try:\n"
        "    import gpt6_sol_api\n"
        f"    gpt6_sol_api.main('/content/{name}_inputs.jsonl', '/content/system_prompt.txt', '/content/gpt6_{name}.jsonl', {max_usd}, workers=4, max_output_tokens=200)\n"
        f"    json.dump({{'ok': True}}, open('/content/status_gpt6_{name}.json', 'w'))\n"
        "   except BaseException:\n"
        f"    json.dump({{'ok': False, 'error': traceback.format_exc()[-2000:]}}, open('/content/status_gpt6_{name}.json', 'w'))\n"
        " threading.Thread(target=job, daemon=True).start()\n"
        " print('started')\n"
    )
    launch_output = checked("exec", "-s", SESSION, "--timeout", "60", code=code, timeout=120)
    log(launch_output.strip().splitlines()[-1] if launch_output.strip() else f"{name} launch accepted")
    start = time.time()
    last_report = 0
    while time.time() - start < 7200:
        time.sleep(20)
        count = snapshot(name)
        if count != last_report and (count % 50 < 4 or count == (50 if name == "pilot" else 2974)):
            log(f"{name}: {count} saved locally")
            last_report = count
        status_path = OUT / f".{name}_status.json"
        rc, _ = cli("download", "-s", SESSION, f"/content/status_gpt6_{name}.json", str(status_path), timeout=90)
        if rc == 0 and status_path.exists():
            status = json.loads(status_path.read_text())
            status_path.unlink(missing_ok=True)
            snapshot(name)
            if not status.get("ok"):
                raise RuntimeError(f"{name} failed: {status.get('error', '')[-700:]}")
            return
        if int(time.time() - start) % 120 < 21:
            cli("exec", "-s", SESSION, "--timeout", "30", code="print('alive')", timeout=75)
    raise RuntimeError(f"{name} did not finish within two hours")


def summarize(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location("schema", ROOT / "colab_scripts/schema.py")
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    rows = [json.loads(line) for line in (OUT / f"{name}.jsonl").open(encoding="utf-8")]
    expected = 50 if name == "pilot" else 2974
    if len(rows) != expected or len({r["id"] for r in rows}) != expected:
        raise RuntimeError(f"{name} has {len(rows)} rows, expected {expected} distinct IDs")
    metrics = schema.summarize(rows)
    metrics.update({
        "model": "gpt-6-sol", "reasoning_effort": "none", "max_output_tokens": 200,
        "cost_usd": round(sum(r["usd"] for r in rows), 6),
        "input_tokens_mean": sum(r["in"] for r in rows) / expected,
        "output_tokens_mean": sum(r["out"] for r in rows) / expected,
        "cache_read_tokens_mean": sum(r["cache_read"] for r in rows) / expected,
        "cache_write_tokens_mean": sum(r["cache_write"] for r in rows) / expected,
        "incomplete_rows": sum(bool(r["incomplete_reason"]) for r in rows),
    })
    (OUT / f"{name}_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    log(f"{name}: {expected} rows, ${metrics['cost_usd']:.4f}, exact {metrics['whole_record_accuracy']:.3f}, incomplete {metrics['incomplete_rows']}")
    return metrics


def main():
    prepare_inputs()
    requested = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if requested not in ("pilot", "full"):
        raise SystemExit("Use pilot or full")
    failed = False
    try:
        setup()
        if requested == "pilot":
            run_job("pilot", 0.50)
            summarize("pilot")
        else:
            pilot = json.loads((OUT / "pilot_summary.json").read_text())
            if pilot["incomplete_rows"] or pilot["errors"]:
                raise RuntimeError("Pilot did not finish cleanly")
            projected = pilot["cost_usd"] / 50 * 2974
            log(f"Pilot-based full cost projection: ${projected:.2f}")
            if projected > 10.25:
                raise RuntimeError("Projected full cost exceeds the $10.25 limit")
            run_job("full", 10.50)
            summarize("full")
    except BaseException:
        failed = True
        raise
    finally:
        if failed:
            snapshot(requested)
        rc, stop_output = cli("stop", "-s", SESSION, timeout=120)
        log("Colab stop: " + (stop_output.strip().splitlines()[-1] if stop_output.strip() else str(rc)))
        log(cli("usage")[1].splitlines()[0])


if __name__ == "__main__":
    main()
