"""Qwen3-0.6B + LoRA 事件抽取（生成 JSON）。"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import DATA, evaluate, get_device, load_jsonl, load_schema, row_event

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def system_prompt(schema):
    lines = ["你是金融公告事件抽取助手。从句子中抽取一个事件，只输出 JSON：",
             '{"type": "事件类型", "trigger": "触发词原文", "arguments": [{"role": "角色", "argument": "论元原文"}]}',
             "事件类型和角色："]
    for t in sorted(schema):
        lines.append(f"- {t}：{'、'.join(schema[t])}")
    return "\n".join(lines)


def target_json(ev):
    return json.dumps({"type": ev["type"], "trigger": ev["trigger"]["text"],
                       "arguments": [{"role": a["role"], "argument": a["text"]}
                                     for a in ev.get("args") or []]}, ensure_ascii=False)


def chat_prompt(tok, system, text):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class SFTDataset(Dataset):
    def __init__(self, rows, tok, system, max_len=1536):
        self.items = []
        for row in rows:
            ev = row_event(row)
            if not ev:
                continue
            prompt = chat_prompt(tok, system, row["text"])
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            a = tok(target_json(ev), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            if len(p) + len(a) > max_len or len(p) >= max_len:
                continue
            self.items.append((p + a, [-100] * len(p) + a))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, labels = self.items[i]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def parse_event(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(obj.get("events"), list):
        obj = obj["events"][0] if obj["events"] else None
    if not isinstance(obj, dict):
        return None
    trig = obj.get("trigger")
    trig = trig.get("text", "") if isinstance(trig, dict) else (trig or "")
    args = []
    for a in obj.get("arguments") or obj.get("args") or []:
        role = a.get("role", "")
        val = a.get("argument") or a.get("text") or ""
        if role and val:
            args.append({"role": role, "text": val})
    return {"type": obj.get("type", ""), "trigger": {"text": trig}, "args": args}


@torch.no_grad()
def predict(model, tok, system, rows, device, max_new=256):
    model.eval()
    preds, ok = [], 0
    for row in rows:
        prompt = chat_prompt(tok, system, row["text"])
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        ev = parse_event(text)
        ok += ev is not None
        preds.append({"id": row.get("id"), "events": [ev] if ev else []})
    return preds, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT / "pretrain_models" / "Qwen3-0.6B"))
    ap.add_argument("--train", default=str(DATA / "train_extract.jsonl"))
    ap.add_argument("--dev", default=str(DATA / "dev_extract.jsonl"))
    ap.add_argument("--num_train", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", default=str(OUT / "lora_adapter"))
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"找不到 Qwen3-0.6B：{model_dir}\n"
                         f"请把模型放到 week7/pretrain_models/Qwen3-0.6B，或用 --model_dir 指定。")

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = get_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    schema = load_schema()
    system = system_prompt(schema)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train = load_jsonl(args.train)
    dev = load_jsonl(args.dev)[: args.n_eval]
    if args.num_train:
        train = train[: args.num_train]
    ds = SFTDataset(train, tok, system, args.max_len)
    print(f"device={device} train={len(ds)} eval={len(dev)} prompt_len={len(tok(system)['input_ids'])}")

    from functools import partial
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=partial(collate, pad_id=tok.pad_token_id), drop_last=True)

    base = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype)
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"])).to(device)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = min(len(dl), args.max_steps) if args.max_steps else len(dl)
    best, log = -1.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(dl):
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            (out.loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            running += out.loss.item()
            n += 1
            if i % 20 == 0:
                print(f"  e{epoch} step {i}/{steps} loss={out.loss.item():.4f}", flush=True)
            if args.max_steps and n >= args.max_steps:
                break

        preds, valid = predict(model, tok, system, dev, device)
        m = evaluate(dev, preds)
        score = m["event"]["f1"]
        log.append({"epoch": epoch, "loss": round(running / max(1, n), 4),
                    "json_valid": round(valid / max(1, len(dev)), 4),
                    "trigger_f1": m["trigger"]["f1"], "argument_f1": m["argument"]["f1"],
                    "event_f1": score, "sec": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: json={log[-1]['json_valid']} trigger={m['trigger']['f1']} "
              f"arg={m['argument']['f1']} event={score} ({log[-1]['sec']}s)")

        if score > best:
            best = score
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.save_dir)
            tok.save_pretrained(args.save_dir)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lora_metrics.json").write_text(
        json.dumps({"best_event_f1": best, "log": log}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best event_f1={best} -> {args.save_dir}")


if __name__ == "__main__":
    main()
