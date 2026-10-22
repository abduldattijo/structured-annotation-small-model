"""Runs the fine tuned small model on a split, with timing and memory, and scores it.

Run inside the Colab kernel:
    import run_local; run_local.main(adapter="/content/adapter_pilot", split="dev", n=50, out="/content/local_pilot.jsonl")
"""
import json
import random
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import schema
from massive_data import load_en_us
from train_lora import MODEL, prompt_of


def generate(model, tok, prompts, max_new=96):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    cut = [bool((row != tok.pad_token_id).all()) and tok.eos_token_id not in row.tolist() for row in new]
    return [tok.decode(r, skip_special_tokens=True).strip() for r in new], cut


def main(adapter, split="dev", n=50, seed=0, out="/content/local_pilot.jsonl", batch=64, dtype="float16",
         latency_rows=50):
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, dtype)).to("cuda")
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload().eval()
    data = load_en_us()
    rows = data[split]
    if n:
        rows = random.Random(seed).sample(rows, n)
    prompts = [prompt_of(r["text"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))      # similar lengths together
    generate(model, tok, prompts[:4])                                    # warm up, not timed
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    raws, cuts = [None] * len(rows), [None] * len(rows)
    t0 = time.time()
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        r, c = generate(model, tok, [prompts[j] for j in idx])
        for j, a, b in zip(idx, r, c):
            raws[j], cuts[j] = a, b
    torch.cuda.synchronize()
    batch_seconds = time.time() - t0
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    lat = []
    for p in prompts[:latency_rows]:                                     # one record at a time
        s = time.time()
        generate(model, tok, [p])
        torch.cuda.synchronize()
        lat.append(time.time() - s)
    recs = [{"id": r["id"], "text": r["text"], "gold": schema.target(r), "raw": raws[i], "cut_off": cuts[i]}
            for i, r in enumerate(rows)]
    with open(out, "w") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summ = schema.summarize(recs)
    summ.update({"model": MODEL, "adapter": adapter, "split": split, "dtype": dtype, "batch_size": batch,
                 "gpu": torch.cuda.get_device_name(0), "records_per_second_batched": round(len(rows) / batch_seconds, 2),
                 "batched_total_seconds": round(batch_seconds, 1), "peak_gpu_memory_gb": peak,
                 "latency_one_record_median_s": round(sorted(lat)[len(lat) // 2], 3),
                 "cut_off_replies": sum(cuts)})
    print(json.dumps(summ, indent=1), flush=True)
    json.dump(summ, open(out.replace(".jsonl", "_summary.json"), "w"), indent=1)
    del model, base
    torch.cuda.empty_cache()
    return summ
