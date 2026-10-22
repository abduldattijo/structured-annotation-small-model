"""Runs the label definition test on a Colab CPU session (API calls only, no GPU).

    python3 run_defs_on_colab.py dev VERSION      one uncached dev run of definitions version VERSION (1, 2, 3)
    python3 run_defs_on_colab.py test VERSION     three cached test passes with the frozen version

The Anthropic key reaches the session only as an uploaded temp file that is deleted at once.
"""
import os
import sys

import run_on_colab as R
from run_on_colab import S, c, say, tail, exec_code, stage_key, step

FILES = ("massive_data", "schema", "run_api", "definitions", "run_api_defs")


def setup_cpu():
    say("balance before: " + c("usage")[1].splitlines()[0])
    rc, out = c("new", "-s", S, timeout=400)
    say(tail(out, 2))
    if rc:
        raise SystemExit("could not start the session")
    say(tail(c("install", "-s", S, "anthropic", timeout=400)[1], 1))
    for f in FILES:
        c("upload", "-s", S, f"colab_scripts/{f}.py", f"/content/{f}.py")
    path = stage_key()
    try:
        say("  " + tail(c("upload", "-s", S, path, "/content/.k")[1], 1).replace(path, "<temp file>"))
    finally:
        os.remove(path)
    rc, out = exec_code(
        "import os\nv=open('/content/.k').read().strip()\nos.environ['ANTHROPIC_API_KEY']=v\nos.remove('/content/.k')\n"
        "print('key in memory only | file removed:', not os.path.exists('/content/.k'), '| length ok:', len(v) > 20)\n", 120)
    say(tail(out, 2))
    if rc:
        raise SystemExit("key step failed")


def main():
    mode, version = sys.argv[1], sys.argv[2]
    try:
        setup_cpu()
        if mode == "dev":
            step(f"definitions v{version} dev 300",
                 f"import run_api_defs; run_api_defs.main(split='dev', n=300, seed=1, out='/content/defs_v{version}_dev.jsonl', max_usd=0.5, workers=4)", 1500)
            R.download([f"defs_v{version}_dev.jsonl", f"defs_v{version}_dev_summary.json"], "results/defs")
        else:
            sha = sys.argv[3]
            for k in (1, 2, 3):
                step(f"definitions v{version} test pass {k}",
                     f"import run_api_defs; run_api_defs.main(split='test', n=None, out='/content/defs_v{version}_test{k}.jsonl', max_usd=0.8, workers=6, cache=True)", 3000)
                R.download([f"defs_v{version}_test{k}.jsonl", f"defs_v{version}_test{k}_summary.json"], "results/defs")
    finally:
        say(tail(c("stop", "-s", S)[1], 1))
        say("balance after: " + c("usage")[1].splitlines()[0])


if __name__ == "__main__":
    main()
