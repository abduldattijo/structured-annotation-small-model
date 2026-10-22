"""Prompted model run: Claude Haiku 5.5 labels each text with one fixed prompt.

Run inside the Colab kernel after the key is in the environment:
    import run_api; run_api.main(split="dev", n=50, seed=0, out="/content/api_pilot.jsonl")

The key is read from ANTHROPIC_API_KEY. It is never printed or written.
Prices are USD per million tokens from Anthropic's pricing page, checked 2026-10-22.
"""
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import schema
from massive_data import load_en_us

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10, 0.50           # per million tokens, prompts up to 100,000 tokens
PRICE_CACHE_WRITE, PRICE_CACHE_READ = 0.125, 0.01   # 5 minute cache; Haiku 5.5 minimum cacheable prompt is 512 tokens
THINKING = {"type": "disabled"}            # no hidden reasoning, so billed output is the answer
MAX_TOKENS = 400                           # a correct reply is about 40 tokens, so a reply that reaches this is runaway text


def pick_examples(train, variant="base"):
    """Fixed examples chosen by rule before any run, from training rows only.

    base: the first train row with 0, 1 and 2 slots.
    s1:   the first train row of each scenario (18 examples).
    s2:   the first two train rows of each scenario (36 examples).
    """
    if variant == "base":
        return [next(r for r in train if len(r["slots"]) == k) for k in (0, 1, 2)]
    per = {"s1": 1, "s2": 2}[variant]
    seen, out = {}, []
    for r in train:
        if seen.get(r["scenario"], 0) < per:
            seen[r["scenario"]] = seen.get(r["scenario"], 0) + 1
            out.append(r)
    return sorted(out, key=lambda r: (r["scenario"], r["id"]))


def build_system(inv, examples):
    lines = [
        "You label short commands given to a voice assistant.",
        "Reply with exactly one line of JSON and nothing else. The JSON has three keys, scenario, intent and slots.",
        "scenario must be one of: " + ", ".join(inv["scenarios"]) + ".",
        "intent must be one of: " + ", ".join(inv["intents"]) + ".",
        "slots is a list of [slot_type, value] pairs in the order the values appear in the text.",
        "Each value must be copied word for word from the text. Use [] when the text has no slots.",
        "slot_type must be one of: " + ", ".join(inv["slot_types"]) + ".",
        "Examples.",
    ]
    for e in examples:
        lines.append("Text: " + e["text"])
        lines.append("JSON: " + schema.dumps(schema.target(e)))
    return "\n".join(lines)


def one_call(client, system, row, cache=False):
    import anthropic
    t0 = time.time()
    last = None
    for attempt in range(5):
        try:
            sysarg = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}] if cache else system
            resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=sysarg, thinking=THINKING,
                                          messages=[{"role": "user", "content": row["text"]}])
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            u = resp.usage
            tin, tout = u.input_tokens, u.output_tokens
            cw = getattr(u, "cache_creation_input_tokens", 0) or 0
            cr = getattr(u, "cache_read_input_tokens", 0) or 0
            usd = (tin * PRICE_IN + cw * PRICE_CACHE_WRITE + cr * PRICE_CACHE_READ + tout * PRICE_OUT) / 1e6
            return {"raw": text, "in": tin, "out": tout, "cache_write": cw, "cache_read": cr, "stop": resp.stop_reason,
                    "usd": usd, "s": time.time() - t0, "error": None}
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            last = type(e).__name__
            time.sleep(2 ** attempt)
    return {"raw": "", "in": 0, "out": 0, "cache_write": 0, "cache_read": 0, "stop": None, "usd": 0.0, "s": time.time() - t0, "error": last}


def main(split="dev", n=50, seed=0, out="/content/api_pilot.jsonl", max_usd=0.5, workers=1, variant="base", cache=False, expected_prompt_sha=None):
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set in this kernel")
    data = load_en_us()
    inv = schema.build_inventory(data["train"])
    examples = pick_examples(data["train"], variant)
    system = build_system(inv, examples)
    if expected_prompt_sha:
        import hashlib
        got = hashlib.sha256(system.encode("utf-8")).hexdigest()
        if got != expected_prompt_sha:
            raise RuntimeError("The prompt differs from the frozen prompt. Stopping before any API call.")
    rows = data[split]
    if n:
        rows = random.Random(seed).sample(rows, n)
    client = anthropic.Anthropic()
    spent = 0.0
    done = []
    print(f"{MODEL} | {split} | {len(rows)} rows | example ids {[e['id'] for e in examples]} | "
          f"variant {variant} | cache {cache} | system prompt {len(system.split())} words", flush=True)

    def work(r):
        return r, one_call(client, system, r, cache)

    if cache and rows:                      # write the cache once before parallel calls, so the rest are reads
        one_call(client, system, rows[0], True)

    with open(out, "w") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for r, res in ex.map(work, rows):
            spent += res["usd"]
            rec = {"id": r["id"], "text": r["text"], "gold": schema.target(r), **res}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.append(rec)
            if spent >= max_usd:
                raise RuntimeError(f"Spend cap reached: ${spent:.4f} >= ${max_usd:.2f}. Stopped after {len(done)} rows.")
            if len(done) % 25 == 0:
                print(f"  {len(done)} rows, ${spent:.4f}", flush=True)
    bad = [d for d in done if d["error"]]                      # transport failures: never results
    cut = [d for d in done if d["stop"] == "max_tokens"]
    empty = [d for d in done if not d["error"] and not d["raw"].strip()]
    for d in (cut + empty)[:3]:
        print("  unusual reply:", d["id"], "| stop:", d["stop"], "|", repr(d["raw"][:160]), flush=True)
    summ = schema.summarize(done)
    summ.update({"cost_usd": round(spent, 6), "input_tokens_mean": sum(d["in"] for d in done) / len(done),
                 "output_tokens_mean": sum(d["out"] for d in done) / len(done),
                 "seconds_per_call_median": sorted(d["s"] for d in done)[len(done) // 2],
                 "error_rows": len(bad), "cut_off_rows": len(cut), "empty_rows": len(empty), "model": MODEL, "split": split, "workers": workers,
                 "variant": variant, "cache": cache, "n_examples": len(examples),
                 "cache_read_tokens_mean": sum(d["cache_read"] for d in done) / len(done),
                 "cache_write_tokens_mean": sum(d["cache_write"] for d in done) / len(done)})
    print(json.dumps(summ, indent=1), flush=True)
    json.dump({"summary": summ, "example_ids": [e["id"] for e in examples], "system_prompt": system},
              open(out.replace(".jsonl", "_summary.json"), "w"), indent=1)
    if bad:
        raise RuntimeError(f"{len(bad)} rows had an API error. Treat as errors, not results.")
    return summ
