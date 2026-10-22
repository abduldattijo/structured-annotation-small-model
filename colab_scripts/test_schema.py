"""Quick checks of the scoring rules. Run with: python3 test_schema.py"""
import schema as S

gold = {"scenario": "alarm", "intent": "alarm_set", "slots": [["time", "nine am"], ["date", "friday"]]}
good = S.dumps(gold)
p, c = S.parse(good); assert c and S.score_record(gold, p)["whole"] == 1
p, c = S.parse("```json\n" + good + "\n```"); assert p is not None and not c            # fenced: scored but not clean
p, c = S.parse("Here: " + good); assert p is not None and not c
p, c = S.parse(good + "\n\nNote: {braces} in a later explanation"); assert p is not None and not c   # first object wins
assert S.parse("not json")[0] is None and S.parse("")[0] is None and S.parse(None)[0] is None
assert S.parse('{"scenario":"a","intent":"b","slots":[{"type":"x","value":"y"}]}')[0] is None   # wrong slot shape
assert S.parse('{"scenario":"a","intent":"b"}')[0] is None
swapped = S.dumps({"scenario": "alarm", "intent": "alarm_set", "slots": [["date", "friday"], ["time", "nine am"]]})
assert S.score_record(gold, S.parse(swapped)[0])["slots_exact"] == 1                          # slot order does not matter
wrong = S.dumps({"scenario": "alarm", "intent": "alarm_query", "slots": [["time", "nine am"]]})
r = S.score_record(gold, S.parse(wrong)[0]); assert r["intent"] == 0 and r["whole"] == 0 and r["slot_tp"] == 1 and r["slot_pred"] == 1
dup = {"scenario": "x", "intent": "x_y", "slots": [["a", "1"], ["a", "1"]]}
assert S.score_record(dup, S.parse(S.dumps({"scenario": "x", "intent": "x_y", "slots": [["a", "1"]]}))[0])["slots_exact"] == 0   # repeated slots count
s = S.summarize([{"gold": gold, "raw": good}, {"gold": gold, "raw": "oops"}])
assert s["n"] == 2 and s["parse_failures"] == 1 and s["whole_record_accuracy"] == 0.5
print("all schema checks passed")
