"""Downloads MASSIVE 1.1 and returns the English (en-US) rows with parsed slots.

Source: https://huggingface.co/datasets/AmazonScience/massive (CC BY 4.0, copyright Amazon.com Inc.
or its affiliates). The Hub copy is loaded by a script that downloads this tarball, so we read the
tarball directly.
"""
import json
import re
import tarfile
import urllib.request
from pathlib import Path

URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
SLOT_RE = re.compile(r"\[([^\[\]:]+?)\s*:\s*([^\[\]]+?)\]")


def download(dest="/content/massive"):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "massive.tar.gz"
    if not tar_path.exists():
        urllib.request.urlretrieve(URL, tar_path)
    return tar_path


def load_en_us(dest="/content/massive"):
    """Returns {'train': [...], 'dev': [...], 'test': [...]} of dicts for the en-US file."""
    tar_path = download(dest)
    with tarfile.open(tar_path) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("en-US.jsonl"))
        lines = tf.extractfile(member).read().decode("utf-8").splitlines()
    out = {"train": [], "dev": [], "test": []}
    for line in lines:
        r = json.loads(line)
        slots = [(t.strip(), v.strip()) for t, v in SLOT_RE.findall(r["annot_utt"])]
        out[r["partition"]].append({"id": r["id"], "text": r["utt"], "scenario": r["scenario"],
                                    "intent": r["intent"], "slots": slots, "annot": r["annot_utt"]})
    return out
