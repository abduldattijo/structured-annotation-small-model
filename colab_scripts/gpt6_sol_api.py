"""Run the frozen MASSIVE annotation prompt with GPT-6 Sol on a Colab CPU.

Inputs and the system prompt are prepared from the saved Haiku run. Results are
appended and flushed one record at a time so a restarted session can resume.
"""
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

MODEL = "gpt-6-sol"
INPUT_PRICE = 2.0
CACHED_INPUT_PRICE = 0.2
CACHE_WRITE_PRICE = 2.5
OUTPUT_PRICE = 10.0


def annotate(client, system, row, max_output_tokens):
    for attempt in range(3):
        start = time.monotonic()
        try:
            response = client.responses.create(
                model=MODEL,
                instructions=system,
                input=row["text"],
                reasoning={"effort": "none"},
                max_output_tokens=max_output_tokens,
                store=False,
            )
            usage = response.usage
            if usage is None:
                raise RuntimeError("OpenAI response had no usage record")
            details = getattr(usage, "input_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            cache_write = getattr(details, "cache_write_tokens", 0) or 0
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            ordinary = max(0, input_tokens - cached - cache_write)
            usd = (ordinary * INPUT_PRICE + cached * CACHED_INPUT_PRICE
                   + cache_write * CACHE_WRITE_PRICE + output_tokens * OUTPUT_PRICE) / 1_000_000
            return {
                "id": row["id"], "text": row["text"], "gold": row["gold"],
                "raw": response.output_text or "", "in": input_tokens,
                "out": output_tokens, "cache_read": cached, "cache_write": cache_write,
                "usd": usd, "s": round(time.monotonic() - start, 3),
                "stop": response.status,
                "incomplete_reason": getattr(getattr(response, "incomplete_details", None), "reason", None),
                "error": None,
            }
        except (RateLimitError, APIConnectionError) as exc:
            if attempt == 2:
                raise RuntimeError(f"API call failed for record {row['id']}: {type(exc).__name__}") from None
            time.sleep(2 ** attempt)
        except APIStatusError as exc:
            if exc.status_code not in (500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"API status {exc.status_code} for record {row['id']}") from None
            time.sleep(2 ** attempt)


def main(inputs, system_file, output, max_usd, workers=4, max_output_tokens=200):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is absent")
    rows = [json.loads(line) for line in open(inputs, encoding="utf-8")]
    system = open(system_file, encoding="utf-8").read()
    existing = [json.loads(line) for line in open(output, encoding="utf-8")] if os.path.exists(output) else []
    done_ids = {r["id"] for r in existing}
    spent = sum(r["usd"] for r in existing)
    remaining = [r for r in rows if r["id"] not in done_ids]
    print(f"{MODEL} | {len(rows)} total | {len(done_ids)} saved | {len(remaining)} remaining | "
          f"cap ${max_usd:.2f} | reasoning none | output cap {max_output_tokens}", flush=True)
    client = OpenAI(max_retries=0, timeout=90.0)
    pending = {}
    source = iter(remaining)
    first_error = None

    def submit_next(pool):
        if spent >= max_usd or first_error is not None:
            return False
        try:
            row = next(source)
        except StopIteration:
            return False
        pending[pool.submit(annotate, client, system, row, max_output_tokens)] = row["id"]
        return True

    with open(output, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(workers):
            submit_next(pool)
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                record_id = pending.pop(future)
                try:
                    rec = future.result()
                except Exception as exc:
                    first_error = f"record {record_id}: {type(exc).__name__}: {exc}"
                    print(first_error, flush=True)
                    continue
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                os.fsync(out.fileno())
                spent += rec["usd"]
                done_ids.add(rec["id"])
                if len(done_ids) % 25 == 0 or len(done_ids) == len(rows):
                    print(f"  {len(done_ids)}/{len(rows)} records, ${spent:.4f}", flush=True)
            while len(pending) < workers and submit_next(pool):
                pass
    if first_error:
        raise RuntimeError(first_error)
    if len(done_ids) < len(rows):
        raise RuntimeError(f"spend cap reached: ${spent:.4f}; saved {len(done_ids)}/{len(rows)} records")
    print(f"complete: {len(done_ids)} records, ${spent:.4f}", flush=True)

