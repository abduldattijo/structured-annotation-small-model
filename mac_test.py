"""Ten minute check: can an Apple M4 with 16 GB train and run Qwen3.5 0.8B with LoRA?

Uses random token ids, so no dataset is involved. It measures training speed, peak memory, and
generation speed. Run from this folder:  .venv/bin/python mac_test.py
Usage: mac_test.py BATCH_SIZE [MODEL_ID]. Results are written to results/mac_test_batchN.json.
"""
import json
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.5-0.8B"
DEVICE = "mps"
SEQ, STEPS = 256, 20
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 4

def mem_gb():
    return round(torch.mps.driver_allocated_memory() / 1e9, 2)

out = {"model": MODEL, "device": DEVICE, "seq_len": SEQ, "batch": BATCH, "steps": STEPS}
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE)
out["load_seconds"] = round(time.time() - t0, 1)
out["loaded_class"] = type(model).__name__
out["params_millions"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
out["memory_after_load_gb"] = mem_gb()

cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
model = get_peft_model(model, cfg)
out["trainable_params_millions"] = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
vocab = tok.vocab_size
steps = []
for i in range(STEPS):
    ids = torch.randint(100, vocab - 100, (BATCH, SEQ), device=DEVICE)
    s = time.time()
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    opt.step(); opt.zero_grad()
    torch.mps.synchronize()
    steps.append(round(time.time() - s, 2))
    if i in (0, 4, 9, STEPS - 1):
        print(f"step {i+1}: {steps[-1]} s, loss {loss.item():.2f}, mem {mem_gb()} GB", flush=True)
out["step_seconds_first5"] = steps[:5]
out["step_seconds_median_after_warmup"] = sorted(steps[5:])[len(steps[5:]) // 2]
out["tokens_per_second_training"] = round(BATCH * SEQ / out["step_seconds_median_after_warmup"])
out["peak_memory_training_gb"] = mem_gb()

model.eval()
prompt = tok("Classify this ticket: the app crashes when I open settings. Answer in JSON.", return_tensors="pt").to(DEVICE)
gen = {}
for bs in (1, 8):
    batch = {k: v.repeat(bs, 1) for k, v in prompt.items()}
    s = time.time()
    with torch.no_grad():
        model.generate(**batch, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.mps.synchronize()
    sec = time.time() - s
    gen[f"batch_{bs}"] = {"seconds_for_80_new_tokens": round(sec, 1), "seconds_per_record": round(sec / bs, 2)}
out["generation"] = gen
out["peak_memory_after_generation_gb"] = mem_gb()
out["total_minutes"] = round((time.time() - t0) / 60, 1)
json.dump(out, open(f"results/mac_test_{MODEL.split('/')[-1]}_batch{BATCH}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
