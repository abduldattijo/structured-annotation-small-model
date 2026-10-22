"""Shared label contract, output parsing, and strict scoring. Pure Python, no model code."""
import json
import re
from collections import Counter


def build_inventory(train):
    return {"scenarios": sorted({r["scenario"] for r in train}),
            "intents": sorted({r["intent"] for r in train}),
            "slot_types": sorted({t for r in train for t, _ in r["slots"]})}


def target(row):
    return {"scenario": row["scenario"], "intent": row["intent"], "slots": [[t, v] for t, v in row["slots"]]}


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def parse(text):
    """Returns (pred, clean). pred is {'scenario','intent','slots':[(type,value),..]} or None.

    clean is True only when the whole reply, ignoring surrounding whitespace, is the JSON object.
    A reply with a code fence or extra words is still scored when the first JSON object can be read
    from it, but it is counted as not clean. Slots must be two element lists of strings.
    """
    t = (text or "").strip()
    clean = True
    obj = None
    try:
        obj = json.loads(t)
    except Exception:
        clean = False
        i = t.find("{")
        if i >= 0:
            try:
                obj, _ = json.JSONDecoder().raw_decode(t[i:])        # the first complete JSON object
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None, False
    if not isinstance(obj.get("scenario"), str) or not isinstance(obj.get("intent"), str):
        return None, False
    slots = obj.get("slots")
    if not isinstance(slots, list):
        return None, False
    out = []
    for s in slots:
        if isinstance(s, list) and len(s) == 2 and all(isinstance(x, str) for x in s):
            out.append((s[0].strip(), s[1].strip()))
        else:
            return None, False
    return {"scenario": obj["scenario"].strip(), "intent": obj["intent"].strip(), "slots": out}, clean


def score_record(gold, pred):
    """gold is a target() dict. pred is a parse() result or None. A missing prediction scores zero."""
    g = Counter((t, v) for t, v in gold["slots"])
    if pred is None:
        return {"scenario": 0, "intent": 0, "slots_exact": 0, "whole": 0,
                "slot_tp": 0, "slot_gold": sum(g.values()), "slot_pred": 0}
    p = Counter(pred["slots"])
    tp = sum((g & p).values())
    sc = int(pred["scenario"] == gold["scenario"])
    it = int(pred["intent"] == gold["intent"])
    se = int(g == p)
    return {"scenario": sc, "intent": it, "slots_exact": se, "whole": int(sc and it and se),
            "slot_tp": tp, "slot_gold": sum(g.values()), "slot_pred": sum(p.values())}


def summarize(rows):
    """rows: dicts with 'gold' (target dict), 'raw' (reply text) and optional 'error'."""
    n = len(rows)
    tot = Counter()
    parse_fail = not_clean = errors = 0
    for r in rows:
        if r.get("error"):
            errors += 1
        pred, clean = parse(r.get("raw", ""))
        parse_fail += pred is None
        not_clean += (not clean)
        for k, v in score_record(r["gold"], pred).items():
            tot[k] += v
    prec = tot["slot_tp"] / tot["slot_pred"] if tot["slot_pred"] else 0.0
    rec = tot["slot_tp"] / tot["slot_gold"] if tot["slot_gold"] else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": n, "errors": errors, "parse_failures": parse_fail, "not_clean_replies": not_clean,
            "scenario_accuracy": tot["scenario"] / n, "intent_accuracy": tot["intent"] / n,
            "slots_exact_accuracy": tot["slots_exact"] / n, "whole_record_accuracy": tot["whole"] / n,
            "slot_precision": prec, "slot_recall": rec, "slot_f1": f1}
