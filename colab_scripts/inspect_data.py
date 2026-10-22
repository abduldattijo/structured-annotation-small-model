"""Run on Colab. Prints what the English MASSIVE data actually contains. No keys, no model."""
import collections
import json
import sys

sys.path.insert(0, "/content")
from massive_data import load_en_us

d = load_en_us()
print({k: len(v) for k, v in d.items()})
tr = d["train"]
print("scenarios:", len({r["scenario"] for r in tr}), "| intents:", len({r["intent"] for r in tr}),
      "| slot types:", len({t for r in tr for t, _ in r["slots"]}))
print("rows with no slots (train): %.1f%%" % (100 * sum(1 for r in tr if not r["slots"]) / len(tr)))
print("slots per row (train): mean %.2f, max %d" % (sum(len(r["slots"]) for r in tr) / len(tr), max(len(r["slots"]) for r in tr)))
print("words per text (train): mean %.1f, max %d" % (sum(len(r["text"].split()) for r in tr) / len(tr), max(len(r["text"].split()) for r in tr)))
print("intent name starts with scenario name: %.1f%%" % (100 * sum(1 for r in tr if r["intent"].startswith(r["scenario"])) / len(tr)))
print("rows where a slot type repeats in one text: %.1f%%" % (100 * sum(1 for r in tr if len({t for t, _ in r['slots']}) != len(r['slots'])) / len(tr)))
mis = [r for r in tr if r["slots"] and any(v not in r["text"] for _, v in r["slots"])]
print("rows where a slot value is not found verbatim in the text:", len(mis), "of", len(tr))
for r in tr[:4]:
    print(json.dumps({k: r[k] for k in ("text", "scenario", "intent", "slots", "annot")}))
print("top intents (train):", collections.Counter(r["intent"] for r in tr).most_common(3))
json.dump({"counts": {k: len(v) for k, v in d.items()}}, open("/content/data_counts.json", "w"))
