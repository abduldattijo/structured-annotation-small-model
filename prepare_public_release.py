"""Stages a public copy of this project next to it, without private key file lookups or the large adapter files.

    python3 prepare_public_release.py
Refuses to overwrite an existing folder. The key lookups in two scripts are replaced by environment variables.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parent / "structured-annotation-small-model-public"
FILES = ["PLAN.md", "RESULTS.md", "COLAB_CLI_RUNBOOK.md", "FROZEN_FULL.json", "draft.md", "requirements.txt",
         "freeze_full.py", "mac_test.py", "run_mac_test.sh", "colab_smoke_test.py", "run_on_colab.py",
         "run_defs_on_colab.py", "run_gpt6_sol_colab.py", "make_figures.py", "prepare_public_release.py"]


def main():
    if DEST.exists():
        raise SystemExit(f"Refusing to overwrite {DEST}")
    DEST.mkdir()
    for f in FILES:
        shutil.copy2(ROOT / f, DEST / f)
    for d in ("colab_scripts", "analysis", "figures"):
        shutil.copytree(ROOT / d, DEST / d, ignore=shutil.ignore_patterns("__pycache__", "ALT_TEXT.md"))
    shutil.copytree(ROOT / "results", DEST / "results",
                    ignore=shutil.ignore_patterns("adapters", "rescued", "__pycache__", "*.safetensors"))
    # Anthropic key: read from the environment instead of a private file
    p = DEST / "run_on_colab.py"
    s = p.read_text()
    a = s.index("def stage_key():")
    b = s.index("def setup(")
    s = s[:a] + '''def stage_key():
    """Writes ANTHROPIC_API_KEY from your environment to a private temp file for upload. Never printed."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set ANTHROPIC_API_KEY in your environment first")
    path = f"/tmp/.expkey_{os.getpid()}"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, key.encode())
    os.close(fd)
    return path


''' + s[b:]
    p.write_text(s)
    # OpenAI key: same change
    p = DEST / "run_gpt6_sol_colab.py"
    s = p.read_text()
    a = s.index("def stage_key():")
    b = s.index("    fd, path = tempfile.mkstemp")
    s = s[:a] + '''def stage_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if len(key) < 20:
        raise RuntimeError("Set OPENAI_API_KEY in your environment first")
''' + s[b:]
    p.write_text(s)
    (DEST / ".gitignore").write_text(".env*\n*.safetensors\n__pycache__/\nlocal-secrets/\n")
    print("staged", DEST)


if __name__ == "__main__":
    main()
