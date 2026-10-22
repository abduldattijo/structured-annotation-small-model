"""LoRA fine tuning of a small model on the MASSIVE English training rows.

The label contract is learned into the adapter, so the prompt at run time is only the text and an arrow.
Run inside the Colab kernel:
    import train_lora; train_lora.main(n_train=1000, epochs=1, out_dir="/content/adapter_pilot")
"""
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import schema
from massive_data import load_en_us

MODEL = "Qwen/Qwen3-0.6B"
ARROW = "\n=>"


def prompt_of(text):
    return text + ARROW


def encode(tok, row, max_len=160):
    p = tok(prompt_of(row["text"]), add_special_tokens=False)["input_ids"]
    t = tok(" " + schema.dumps(schema.target(row)), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids, labels = p + t, [-100] * len(p) + t
    if len(ids) > max_len:
        raise RuntimeError(f"example {row['id']} is {len(ids)} tokens, over the {max_len} limit")
    return ids, labels


def batches(examples, size, pad_id, rng):
    order = list(range(len(examples)))
    rng.shuffle(order)
    for i in range(0, len(order), size):
        chunk = [examples[j] for j in order[i:i + size]]
        m = max(len(c[0]) for c in chunk)
        ids = torch.tensor([c[0] + [pad_id] * (m - len(c[0])) for c in chunk])
        lab = torch.tensor([c[1] + [-100] * (m - len(c[1])) for c in chunk])
        att = torch.tensor([[1] * len(c[0]) + [0] * (m - len(c[0])) for c in chunk])
        yield ids, lab, att


def main(n_train=None, epochs=3, batch=32, lr=2e-4, seed=0, out_dir="/content/adapter", rank=16, micro=8):
    random.seed(seed)
    torch.manual_seed(seed)
    data = load_en_us()
    rows = data["train"]
    if n_train:
        rows = random.Random(seed).sample(rows, n_train)
    tok = AutoTokenizer.from_pretrained(MODEL)
    examples = [encode(tok, r) for r in rows]
    # fp32 weights with fp16 autocast: stable on a T4, which has no fast bf16
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to("cuda")
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps_total = epochs * math.ceil(len(examples) / batch)
    warm = max(1, int(0.05 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps_total - s) / max(1, steps_total - warm)))
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(seed)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    log, step, t0 = [], 0, time.time()
    print(f"{MODEL} | {len(examples)} examples | {epochs} epochs | batch {batch} | {steps_total} steps | "
          f"{trainable/1e6:.1f}M trainable", flush=True)
    for ep in range(epochs):
        for ids, lab, att in batches(examples, batch, tok.pad_token_id, rng):
            # one optimizer step per batch of 32, computed in pieces of `micro` rows to fit the GPU memory
            nb, total = ids.shape[0], 0.0
            for i in range(0, nb, micro):
                ci, cl, ca = ids[i:i + micro].cuda(), lab[i:i + micro].cuda(), att[i:i + micro].cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    part = model(input_ids=ci, attention_mask=ca, labels=cl).loss * (ci.shape[0] / nb)
                if not torch.isfinite(part):
                    raise RuntimeError(f"loss is {part.item()} at step {step}. Stop and investigate.")
                scaler.scale(part).backward()
                total += part.item()
            loss = torch.tensor(total)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 10 == 0 or step == steps_total:
                log.append({"step": step, "loss": round(loss.item(), 4), "seconds": round(time.time() - t0, 1)})
                print(log[-1], flush=True)
    model.save_pretrained(out_dir)
    info = {"model": MODEL, "examples": len(examples), "epochs": epochs, "batch": batch, "lr": lr, "rank": rank,
            "micro_batch": micro, "steps": step, "train_seconds": round(time.time() - t0, 1),
            "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "trainable_params_millions": round(trainable / 1e6, 2), "loss_log": log}
    json.dump(info, open(out_dir.rstrip("/") + "_train_info.json", "w"), indent=1)
    del model, opt
    torch.cuda.empty_cache()
    return info
