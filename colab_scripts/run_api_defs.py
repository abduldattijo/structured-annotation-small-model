"""Haiku run with a one line meaning for every intent and slot type added to the frozen base prompt.

Same model, settings, three examples, and scoring as run_api.py. Only the label lists change: each label
now carries a meaning from definitions.py. Use:
    import run_api_defs; run_api_defs.main(split="dev", n=300, seed=1, out="/content/defs_dev.jsonl", ...)
"""
import hashlib

import run_api
import schema
from definitions import INTENT_DEFS, SLOT_DEFS


def build_system(inv, examples):
    assert set(INTENT_DEFS) == set(inv["intents"]), "intent definitions do not match the label list"
    assert set(SLOT_DEFS) == set(inv["slot_types"]), "slot definitions do not match the label list"
    lines = [
        "You label short commands given to a voice assistant.",
        "Reply with exactly one line of JSON and nothing else. The JSON has three keys, scenario, intent and slots.",
        "scenario must be one of: " + ", ".join(inv["scenarios"]) + ".",
        "intent must be one of the following. Each one is shown with its meaning.",
    ]
    lines += [f"{i}: {INTENT_DEFS[i]}" for i in inv["intents"]]
    lines += [
        "slots is a list of [slot_type, value] pairs in the order the values appear in the text.",
        "Each value must be copied word for word from the text. Use [] when the text has no slots.",
        "slot_type must be one of the following. Each one is shown with its meaning.",
    ]
    lines += [f"{s}: {SLOT_DEFS[s]}" for s in inv["slot_types"]]
    lines.append("Examples.")
    for e in examples:
        lines.append("Text: " + e["text"])
        lines.append("JSON: " + schema.dumps(schema.target(e)))
    return "\n".join(lines)


def main(**kw):
    run_api.build_system = build_system
    return run_api.main(**kw)


def prompt_sha(train):
    inv = schema.build_inventory(train)
    s = build_system(inv, run_api.pick_examples(train, "base"))
    return s, hashlib.sha256(s.encode("utf-8")).hexdigest()
